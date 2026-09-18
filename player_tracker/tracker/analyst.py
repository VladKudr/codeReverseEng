"""ИИ-аналитик действий игрока по итогам слежения (DeepSeek, OpenAI-совместимый API).

Модель не видит видео (deepseek-chat — текстовая), поэтому ей даётся не картинка, а структурированная
выжимка того, что измерено:

  * профиль из опроса перед слежением (`profile.PlayerProfile`): возраст, позиция, формат игры, рост — и
    вытекающие ориентиры: размер поля и характер игры для формата, по чему судить позицию, возрастные
    зоны скорости; плюс цвет формы и «на что смотреть»;
  * сводка метрик (`player_metrics.compute` без посекундной ленты): время в кадре, дистанция, зоны
    скорости, спринты, ускорения, единоборства, «мяч рядом», положение в сцене;
  * качество данных: доля отслеженного времени, поправки оператора, оговорки метрик — модель обязана
    их учитывать и не делать выводов там, где данных нет;
  * игровой контекст с тактической доски (`board.game_context`): где мяч (найден / оценён в разрыве),
    у какой команды (ближайший игрок), расстояние игрока до мяча, положение относительно своей команды и
    последней линии обороны;
  * инструменты, которыми модель сама добирает подробности: посекундная лента за отрезок, эпизоды
    (спринты, единоборства, мяч рядом), события слежения, расстановка всех игроков и мяча на момент.

Ответ — разбор с отметками времени [мм:сс] (веб делает их переходом по видео) и чат для уточняющих
вопросов с тем же контекстом. Видео и кадры никуда не отправляются — только числа и текст.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .board import track_off
from .profile import PlayerProfile

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

SYSTEM_PROMPT = """Ты — тренер-аналитик детско-юношеского футбола. Тебе дают результаты компьютерного анализа видео:
одного игрока отследили на видео, посчитали его перемещения, скорости, близость к соперникам и к мячу.
Видео ты не видишь — только эти данные.

Правила:
1. Опирайся только на данные. Не придумывай передачи, удары, отборы, голы и их качество — таких данных нет.
   «Мяч рядом» значит лишь, что мяч был в пределах ~1 м от ног игрока (владение или борьба за мяч).
   Игровой мяч отслежен по всему ролику (раздел «ball»: доля кадров, где найден, и где оценён в коротком
   разрыве); где мяч неизвестен — не делай выводов о владении и участии в эпизоде.
   Раздел «game»: possession_estimate — у кого мяч по ближайшему игроку (own — своя команда, opp —
   соперник, loose — ничей/в борьбе), это оценка; focus_dist_to_ball_m — насколько игрок был близко к мячу;
   focus_last_line_share — доля времени, когда игрок был последним полевым своей команды (для защитника —
   держал линию). В ленте: ball_third — треть поля с мячом, ball_with — у кого мяч, dist_to_ball_m,
   vs_team_m — насколько игрок впереди (+) или позади (−) центра своей команды, vs_last_line_m — впереди
   последней линии своей команды. Связывай действия игрока с фазой: мяч у своих — атака, у соперника —
   оборона (где был игрок, возвращался ли, страховал ли). В расстановке officials — судья, тренеры и
   взрослые у поля: это не игроки, их в счёт соперников не бери; вратари даны отдельно с командой.
2. Метры и скорости — оценка по росту игрока в кадре (±20–30 %). Сравнивай с возрастными ориентирами
   осторожно и говори об этом.
3. Учитывай качество данных: если игрок был в кадре малую долю времени, итоги — только за это время.
   Если были поправки оператора — слежение там исправлено вручную.
4. Каждое наблюдение про конкретный момент подкрепляй отметкой времени в формате [мм:сс].
   Если нужно подробнее — вызови инструменты (посекундная лента, эпизоды, события) и не гадай.
5. Разделяй «что измерено» и «как это можно интерпретировать». Если вывода сделать нельзя — так и скажи
   и подскажи, что снять или отметить, чтобы в следующий раз ответить.
6. Учитывай профиль: возраст (возрастные зоны скорости и нагрузки), позицию (оценивай по задачам этой позиции,
   см. «Ориентиры») и формат игры (размер поля и плотность: расстояния, темп, частота единоборств
   на маленьком поле и на большом означают разное). Если фактическое положение игрока не похоже на
   заявленную позицию — отметь это как наблюдение, без упрёка.
7. Схема поля (раздел «pitch» в метриках) задана оператором: где камера, куда смотрит, где свои ворота.
   Говори о своей и чужой половине, третях поля и флангах относительно атаки СВОЕЙ команды игрока;
   рывки «к чужим воротам» — атакующие, «к своим воротам» — возврат в оборону. Положение на поле —
   оценка (±несколько метров): делай выводы о тенденциях, а не о точке. Если outside_share большой —
   предупреди, что схема, вероятно, отмечена неточно.
8. Пиши по-русски, дружелюбно и конкретно, как тренер родителю или юному игроку, без воды.

Формат первого разбора (markdown):
## Коротко — 2–3 предложения.
## Физическая работа — дистанция, темп, зоны, спринты, ускорения; ориентиры для возраста.
## Участие в игре — близость к мячу, владение своей команды и соперника, единоборства, открывания; ключевые моменты со временем.
## Позиционирование — трети поля и фланги относительно своей атаки, положение относительно своей команды и последней линии в обороне и в атаке, направления рывков; насколько это похоже на задачи его позиции в этом формате.
## Что улучшить — 2–4 конкретные рекомендации с упражнениями.
## Ограничения данных — что из-за видео/слежения сказать нельзя."""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_timeline",
        "description": "Посекундная лента за отрезок видео: в кадре ли игрок (доля), средняя и максимальная скорость (м/с), "
                       "зона скорости, отметка времени time (мм:сс — её и указывай в ответе), положение x (поперёк, м) и "
                       "z (от камеры, м), треть поля (third) и фланг (channel) "
                       "относительно атаки своей команды, to_goal — 0 у своих ворот … 1 у чужих, расстояние до ближайшего "
                       "игрока (м), доля секунды с мячом рядом, где мяч (ball_third, ball_to_goal), у кого он (ball_with), "
                       "расстояние до мяча (dist_to_ball_m), положение относительно своей команды (vs_team_m) и её последней "
                       "линии (vs_last_line_m), события слежения. Не больше 120 секунд за вызов.",
        "parameters": {"type": "object", "properties": {
            "start_s": {"type": "number", "description": "начало, секунды от начала видео"},
            "end_s": {"type": "number", "description": "конец, секунды"}},
            "required": ["start_s", "end_s"]}}},
    {"type": "function", "function": {
        "name": "get_episodes",
        "description": "Эпизоды: sprints — спринты (начало, конец, пиковая скорость, дистанция, direction — к чужим/своим "
                       "воротам или поперёк, towards_goal_m — на сколько метров ближе к чужим воротам); close_contacts — единоборства/"
                       "плотная опека (ближайший игрок < 1.5 м дольше 0.5 с, peak — минимальное расстояние); ball_episodes — "
                       "мяч рядом с игроком (distance_m — сколько пробежал с мячом рядом).",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["sprints", "close_contacts", "ball_episodes"]}},
            "required": ["kind"]}}},
    {"type": "function", "function": {
        "name": "get_positions",
        "description": "Расстановка на момент видео (тактическая доска): игрок в фокусе, партнёры и соперники (полевые), "
                       "вратари с командой, число officials (судья, тренеры, взрослые — не игроки) — координаты "
                       "на поле в метрах (x вдоль поля от левой линии ворот схемы, y поперёк от дальней бровки) и "
                       "to_goal (0 у своих ворот … 1 у чужих), мяч (найден/оценён) и у кого он. Можно до 5 моментов за вызов.",
        "parameters": {"type": "object", "properties": {
            "times_s": {"type": "array", "items": {"type": "number"}, "description": "моменты, секунды от начала видео"}},
            "required": ["times_s"]}}},
    {"type": "function", "function": {
        "name": "get_tracking_events",
        "description": "События слежения со временем: захват, потеря (игрок ушёл из кадра или закрыт), повторный захват, "
                       "поправки оператора. Нужны, чтобы понять, где данные надёжны.",
        "parameters": {"type": "object", "properties": {}}}},
]


def fmt_time(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def load_api_key(data_dir: Optional[Path] = None) -> Optional[str]:
    """DEEPSEEK_API_KEY из окружения, иначе data/secrets.json {"deepseek_api_key": "..."} (каталог data в git не попадает)."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    if data_dir is not None:
        p = Path(data_dir) / "secrets.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")).get("deepseek_api_key") or None
            except json.JSONDecodeError:
                return None
    return None


class DeepSeekClient:
    def __init__(self, api_key: str, model: str = "deepseek-chat", url: str = DEEPSEEK_URL, timeout: float = 180.0):
        self.api_key, self.model, self.url, self.timeout = api_key, model, url, timeout

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> dict:
        import httpx

        body = {"model": self.model, "messages": messages, "temperature": 0.4}
        if tools:
            body["tools"] = tools
        r = httpx.post(self.url, json=body, timeout=self.timeout,
                       headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"DeepSeek {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]


@dataclass
class PlayerContext:
    profile: PlayerProfile = field(default_factory=PlayerProfile)
    team_color: Optional[str] = None
    focus: Optional[str] = None          # на что обратить внимание

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PlayerContext":
        d = d or {}
        return cls(PlayerProfile.from_dict(d), d.get("team_color") or None, d.get("focus") or None)

    def as_text(self) -> str:
        p = self.profile
        parts = []
        for label, v in (("возраст", p.age and f"{p.age:g}"), ("позиция", p.position_label()),
                         ("формат игры", p.format_label()), ("рост, м", p.height_m),
                         ("форма", self.team_color), ("на что смотреть", self.focus)):
            if v not in (None, ""):
                parts.append(f"{label}: {v}")
        return "; ".join(parts) or "не указан"


def build_context(metrics: dict, job: dict, player: PlayerContext) -> str:
    """Выжимка для модели: всё, кроме посекундной ленты и тепловой карты (их модель берёт инструментами)."""
    brief = {k: v for k, v in metrics.items() if k != "timeline"}
    pos = dict(brief.get("position") or {})
    heat = pos.pop("heatmap", None)
    if heat:
        pos["heatmap_note"] = heat.get("note")
        pos["x_range_m"], pos["z_range_m"] = heat.get("x_range_m"), heat.get("z_range_m")
    brief["position"] = pos
    if brief.get("pitch"):
        pt = dict(brief["pitch"])
        pt.pop("heatmap", None)
        pt.pop("setup", None)
        brief["pitch"] = pt
    mv = dict(brief.get("movement") or {})
    mv["sprints"] = len(mv.get("sprints") or [])
    brief["movement"] = mv
    inv = dict(brief.get("involvement") or {})
    inv["close_contacts"] = len(inv.get("close_contacts") or [])
    inv["ball_episodes"] = len(inv.get("ball_episodes") or [])
    brief["involvement"] = inv
    video = {"длительность": fmt_time(metrics.get("duration_s", 0)), "поправок оператора": len(job.get("corrections") or []),
             "выбор цели": job.get("init", {}).get("number") and f"номер {job['init']['number']}" or "кликом по игроку"}
    return (f"Игрок: {player.as_text()}\n"
            f"Ориентиры:\n{player.profile.guide()}\n"
            f"Видео: {json.dumps(video, ensure_ascii=False)}\n"
            f"Метрики (эпизоды — количеством; подробности — инструментами):\n{json.dumps(brief, ensure_ascii=False)}")


class Analyst:
    """Разбор и чат с инструментами. `client` — объект с методом chat(messages, tools) -> message."""

    def __init__(self, client, metrics: dict, job: dict, max_tool_rounds: int = 6, board: Optional[dict] = None,
                 pitch=None):
        self.client = client
        self.metrics = metrics
        self.job = job
        self.max_tool_rounds = max_tool_rounds
        self.board = board
        self.pitch = pitch

    # --- инструменты -----------------------------------------------------------------------------------
    def _tool(self, name: str, args: dict) -> str:
        if name == "get_timeline":
            a = float(args.get("start_s", 0))
            b = min(float(args.get("end_s", a + 60)), a + 120)
            rows = [{"time": fmt_time(r["t"]), **r} for r in self.metrics.get("timeline", []) if a <= r["t"] <= b]
            return json.dumps(rows, ensure_ascii=False)
        if name == "get_episodes":
            kind = args.get("kind")
            src = self.metrics.get("movement", {}) if kind == "sprints" else self.metrics.get("involvement", {})
            eps = src.get(kind)
            if eps is None:
                return json.dumps({"error": f"нет эпизодов вида {kind}"}, ensure_ascii=False)
            return json.dumps([{**e, "start": fmt_time(e["start_s"]), "end": fmt_time(e["end_s"])} for e in eps],
                              ensure_ascii=False)
        if name == "get_positions":
            return json.dumps([self._positions(float(t)) for t in (args.get("times_s") or [])[:5]], ensure_ascii=False)
        if name == "get_tracking_events":
            fps = self.job.get("fps") or 30.0
            ev = [{"time": fmt_time(e["time"]), "event": e["event"]} for e in self.job.get("events", [])]
            corr = [{"time": fmt_time(c["frame"] / fps), "correction": "цели нет" if c.get("absent") else "цель указана"}
                    for c in self.job.get("corrections", [])]
            return json.dumps({"events": ev, "corrections": corr}, ensure_ascii=False)
        return json.dumps({"error": f"неизвестный инструмент {name}"}, ensure_ascii=False)

    def _positions(self, t: float) -> dict:
        b = self.board
        if not b or not b.get("frames") or self.pitch is None:
            return {"time": fmt_time(t), "error": "тактическая доска не построена или схема поля не отмечена"}
        fr = min(b["frames"], key=lambda f: abs(f["t"] - t))
        own = b.get("own_team")
        tracks = b.get("tracks", {})
        focus = fr.get("focus")

        def pt(x, y):
            return {"x": x, "y": y, "to_goal": round(float(self.pitch.progress(x)), 2)}

        mates, opps, keepers, officials = [], [], [], 0
        for tid, x, y in fr["p"]:
            if focus and tid == focus.get("id"):
                continue
            meta = tracks.get(str(tid), {})
            if track_off(meta, fr.get("f", 0)):       # вне игры на этом кадре (состав): не участник
                continue
            team = meta.get("team")
            side = "свои" if own is not None and team == own else "соперник" if team in (0, 1) else "неизвестно"
            if meta.get("gk"):
                keepers.append({**pt(x, y), "team": side})
            elif own is not None and team == own:
                mates.append(pt(x, y))
            elif team in (0, 1):
                opps.append(pt(x, y))
            else:
                officials += 1
        out = {"time": fmt_time(fr["t"]), "focus": pt(focus["x"], focus["y"]) if focus else None,
               "teammates": mates, "opponents": opps, "goalkeepers": keepers, "officials": officials}
        if fr.get("ball"):
            bx, by, st = fr["ball"]
            out["ball"] = {**pt(bx, by), "state": "найден" if st == "d" else "оценка в разрыве"}
            if focus:
                out["focus_to_ball_m"] = round(float(((focus["x"] - bx) ** 2 + (focus["y"] - by) ** 2) ** 0.5), 1)
        else:
            out["ball"] = "неизвестно"
        return out

    # --- диалог ------------------------------------------------------------------------------------------
    def ask(self, history: list[dict], question: str, player: PlayerContext,
            on_tool: Optional[Callable[[str, dict], None]] = None) -> tuple[str, list[dict]]:
        """history — прежние сообщения (без системного); возвращает (ответ, новая история)."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if not history:
            history = [{"role": "user", "content": "Данные анализа:\n" + build_context(self.metrics, self.job, player)}]
        messages += history + [{"role": "user", "content": question}]
        new = history + [{"role": "user", "content": question}]
        for _ in range(self.max_tool_rounds + 1):
            msg = self.client.chat(messages, TOOLS)
            calls = msg.get("tool_calls") or []
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            new.append(assistant)
            if not calls:
                return assistant["content"], new
            for call in calls:
                fn = call.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if on_tool:
                    on_tool(fn.get("name", ""), args)
                result = {"role": "tool", "tool_call_id": call.get("id"), "content": self._tool(fn.get("name", ""), args)}
                messages.append(result)
                new.append(result)
        raise RuntimeError("модель не завершила ответ за отведённое число вызовов инструментов")
