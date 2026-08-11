#!/usr/bin/env python3
"""Оркестратор req-reverse v3.

Двухслойная архитектура: LLM порождает -> код проверяет -> LLM правит по
тексту конкретной ошибки. Гейт — всегда код; ретраи ограничены (по умолчанию
2), затем задача помечается как эскалация.

Что делает runner:
  * читает config.yaml (репозиторий, workspace, команда CLI-агента, пороги,
    соответствие этап -> уровень модели);
  * строит план прогона из инвентаризации этапа 0 (фан-аут этапов 1 и 4 по
    модулям, фильтр тестов/вендоринга/генерённого кода);
  * для каждой задачи подставляет переменные в промпт, вызывает агента,
    маскирует секреты в артефакте, валидирует схемой и checks, при ошибке
    формирует retry-промпт с текстом ошибки;
  * идемпотентен: повторный запуск пропускает этапы с валидным выходом
    (--force перегенерирует);
  * ведёт журнал workspace/run.log и итоговую сводку workspace/run_summary.md.

Запуск:
  python runner.py --config config.yaml            # этапы 0–7 и 90
  python runner.py --stages 0,1,4 --force
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import fnmatch
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checks import completeness, coverage, evidence_binding, id_allocator, schema_check, secrets_mask  # noqa: E402
from checks._common import CheckResult, Finding, render_for_retry  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_STAGES = ["0", "1", "2", "3", "4", "5", "6", "7", "90", "92"]

# Ярусная глубина реверса: L0 — оглавление (этапы 0–1), L1 — + интеграционные
# зоны (2–3), L2 — полный прогон (4+ и валидация).
TIER_RANK = {"L0": 0, "L1": 1, "L2": 2}
# минимальный ярус, с которого выполняется этап
STAGE_MIN_TIER = {"0": "L0", "1": "L0", "2": "L1", "3": "L1",
                  "4": "L2", "5": "L2", "6": "L2", "7": "L2",
                  "89": "L2", "90": "L2", "92": "L2"}


# ── конфигурация ────────────────────────────────────────────────────────────

@dataclass
class Config:
    base: Path
    repo_path: Path
    repo_name: str
    workspace: Path
    agent_command: str
    agent_by_stage: dict[str, str]
    agent_stdin: bool
    timeout_sec: int
    metrics_regex: str
    models: dict[str, str]
    exclude: list[str]
    min_lines: int
    stage90_pct: float
    retries: int
    tier_default: str
    tier_map: list[tuple[str, str]]
    inverse: bool

    @classmethod
    def load(cls, path: Path) -> "Config":
        base = path.resolve().parent
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        def as_path(p: str) -> Path:
            p = Path(p)
            return p if p.is_absolute() else (base / p).resolve()

        agent = raw.get("agent") or {}
        mf = raw.get("module_filter") or {}
        tiers = raw.get("tiers")
        if tiers is None:
            # секции tiers нет — поведение базовой постановки: всё L2
            tier_default, tier_map = "L2", []
        else:
            tier_default = str(tiers.get("default", "L0"))
            tier_map = [(str(e["path"]), str(e["tier"]))
                        for e in (tiers.get("map") or [])]
        for _p, t in tier_map + [("", tier_default)]:
            if t not in TIER_RANK:
                raise ValueError(f"неизвестный ярус {t!r} (допустимы {sorted(TIER_RANK)})")
        return cls(
            base=base,
            repo_path=as_path(raw["repo"]["path"]),
            repo_name=str(raw["repo"]["name"]),
            workspace=as_path(raw.get("workspace", "workspace")),
            agent_command=agent.get("command", ""),
            agent_by_stage={str(k): v for k, v in (agent.get("by_stage") or {}).items()},
            agent_stdin=bool(agent.get("stdin", False)),
            timeout_sec=int(agent.get("timeout_sec", 1800)),
            metrics_regex=agent.get("metrics_regex") or "",
            models={str(k): str(v) for k, v in (raw.get("models") or {}).items()},
            exclude=list(mf.get("exclude") or []),
            min_lines=int(mf.get("min_lines", 0)),
            stage90_pct=float((raw.get("thresholds") or {}).get("stage90_confirmed_pct", 97)),
            retries=int(raw.get("retries", 2)),
            tier_default=tier_default,
            tier_map=tier_map,
            inverse=bool((raw.get("validation") or {}).get("inverse", False)),
        )

    def command_for(self, stage: str) -> str:
        # пустое переопределение = «не задано», падаем на общую команду
        return self.agent_by_stage.get(stage) or self.agent_command


# ── журнал ──────────────────────────────────────────────────────────────────

class RunLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, msg: str) -> None:
        stamp = _dt.datetime.now().isoformat(timespec="seconds")
        line = f"{stamp} {msg}"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line)


@dataclass
class TaskOutcome:
    tag: str
    status: str          # ок | пропущен | эскалация
    retries: int = 0
    tokens: int | None = None
    errors: str = ""


# ── оркестратор ─────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, cfg: Config, force: bool = False, dry_run: bool = False):
        self.cfg = cfg
        self.force = force
        self.dry_run = dry_run
        self.ws = cfg.workspace
        self.log = RunLog(self.ws / "run.log")
        self.outcomes: list[TaskOutcome] = []
        self.mask_events: list[str] = []
        self._prepare_dirs()

    # каталоги артефактов — фиксированная структура v2
    def _prepare_dirs(self) -> None:
        for d in ["inventory", f"extracts/{self.cfg.repo_name}/modules",
                  f"extracts/{self.cfg.repo_name}/rules", "integration",
                  "final", "validation", "reports", "_scratch/prompts",
                  "_scratch/logs"]:
            (self.ws / d).mkdir(parents=True, exist_ok=True)

    # ── пути артефактов ────────────────────────────────────────────────────

    def inventory_file(self) -> Path:
        return self.ws / "inventory" / f"{self.cfg.repo_name}.yaml"

    def extracts(self) -> Path:
        return self.ws / "extracts" / self.cfg.repo_name

    def rules_dir(self) -> Path:
        return self.extracts() / "rules"

    @staticmethod
    def module_basename(module_path: str) -> str:
        # правило v2: tenacity/nap.py -> nap; tenacity/__init__.py -> __init__
        return Path(module_path).stem if Path(module_path).suffix else Path(module_path).name

    # ── ярусная глубина реверса ────────────────────────────────────────────

    def tiers_override_file(self) -> Path:
        return self.ws / "tiers.yaml"

    def tier_overrides(self) -> dict[str, str]:
        """Поднятые ярусы из workspace (команда --raise-tier); поверх config."""
        p = self.tiers_override_file()
        if not p.is_file():
            return {}
        return {str(k): str(v)
                for k, v in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).items()}

    def raise_tier(self, path: str, tier: str) -> None:
        if tier not in TIER_RANK:
            raise ValueError(f"неизвестный ярус {tier!r}")
        overrides = self.tier_overrides()
        current = self.tier_of(path)
        if TIER_RANK[tier] <= TIER_RANK[current]:
            self.log.write(f"[ярусы] {path} уже {current} — поднятие до {tier} не требуется")
            return
        overrides[path] = tier
        self.tiers_override_file().write_text(
            yaml.safe_dump(overrides, allow_unicode=True, sort_keys=True),
            encoding="utf-8")
        self.log.write(f"[ярусы] {path} поднят до {tier} (сохранено в {self.tiers_override_file()})")

    def tier_of(self, module_path: str) -> str:
        """Ярус пути: самое длинное совпадение по карте (override поверх config).

        Запись карты — файл либо префикс-каталог (`src/payments/`).
        Пути вне карты получают ярус по умолчанию.
        """
        entries = list(self.cfg.tier_map) + list(self.tier_overrides().items())

        def matches(entry_path: str) -> bool:
            e = entry_path.rstrip("/")
            return module_path == e or module_path.startswith(e + "/")

        best: tuple[int, str] | None = None
        for entry_path, tier in entries:
            if matches(entry_path):
                key = (len(entry_path.rstrip("/")), tier)
                # override идёт после config и при равной длине побеждает
                if best is None or key[0] >= best[0]:
                    best = key
        return best[1] if best else self.cfg.tier_default

    def stage_allowed(self, stage: str, max_tier: str) -> bool:
        need = STAGE_MIN_TIER.get(stage, "L2")
        return TIER_RANK[max_tier] >= TIER_RANK[need]

    # ── план прогона ───────────────────────────────────────────────────────

    def modules_from_inventory(self) -> list[str]:
        inv_path = self.inventory_file()
        if not inv_path.is_file():
            return []
        data = yaml.safe_load(inv_path.read_text(encoding="utf-8")) or {}
        modules: list[str] = []
        for m in data.get("modules") or []:
            path = m.get("path") if isinstance(m, dict) else str(m)
            if not path:
                continue
            if any(fnmatch.fnmatch(path, pat) for pat in self.cfg.exclude):
                self.log.write(f"[план] модуль {path} исключён фильтром")
                continue
            if self.cfg.min_lines:
                size = m.get("approx_size") if isinstance(m, dict) else None
                try:
                    if size is not None and int(size) < self.cfg.min_lines:
                        self.log.write(f"[план] модуль {path} короче {self.cfg.min_lines} строк — пропуск")
                        continue
                except (TypeError, ValueError):
                    pass
            modules.append(path)
        return modules

    # ── промпты и агент ────────────────────────────────────────────────────

    def render_prompt(self, stage: str, variables: dict[str, str]) -> str:
        prompt_file = sorted((HERE / "prompts").glob(f"{int(stage):02d}_*.md"))
        if not prompt_file:
            raise FileNotFoundError(f"нет промпта этапа {stage} в prompts/")
        text = prompt_file[0].read_text(encoding="utf-8")
        for key, val in variables.items():
            text = text.replace("{" + key + "}", str(val))
        return text

    def base_variables(self) -> dict[str, str]:
        return {
            "REPO_PATH": str(self.cfg.repo_path),
            "REPO_NAME": self.cfg.repo_name,
            "WS": str(self.ws),
        }

    def call_agent(self, stage: str, tag: str, prompt_text: str,
                   out_file: Path) -> int | None:
        """Запуск CLI-агента; возвращает токены (если агент отдаёт метрики).

        Промпт передаётся одним из трёх способов — по выбору команды в config:
          {prompt_file} — путь к файлу с готовым промптом;
          {prompt_text} — текст промпта одним аргументом argv (для CLI вида
                          `qwen -p <текст>`); подстановка по-токенно, поэтому
                          кавычки и переносы внутри текста ничего не ломают;
          agent.stdin: true — текст промпта подаётся агенту на stdin.
        """
        prompt_path = self.ws / "_scratch" / "prompts" / f"{tag}.md"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        subs = {
            "prompt_file": str(prompt_path),
            "prompt_text": prompt_text,
            "out_file": str(out_file),
            "model": self.cfg.models.get(stage, ""),
            "workdir": str(self.cfg.base),
        }

        def substitute(token: str) -> str:
            for key, val in subs.items():
                token = token.replace("{" + key + "}", val)
            return token

        # сначала разбор шаблона, потом подстановка: текст промпта попадает
        # в argv целиком одним аргументом, а не проходит через shlex
        argv = [substitute(tok) for tok in shlex.split(self.cfg.command_for(stage))]
        log_path = self.ws / "_scratch" / "logs" / f"{tag}.log"
        shown = " ".join(a if len(a) <= 120 else a[:117] + "…" for a in argv)
        self.log.write(f"[{tag}] агент: {shown}")
        if not argv:
            self.log.write(f"[{tag}] ОСТАНОВ: команда агента пуста (agent.command в config.yaml)")
            return None
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True,
                input=prompt_text if self.cfg.agent_stdin else None,
                timeout=self.cfg.timeout_sec, cwd=str(self.cfg.base),
            )
        except subprocess.TimeoutExpired:
            self.log.write(f"[{tag}] агент превысил тайм-аут {self.cfg.timeout_sec} с")
            return None
        except FileNotFoundError as e:
            self.log.write(f"[{tag}] команда агента не найдена: {e}")
            return None
        log_path.write_text(
            (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""),
            encoding="utf-8")
        if proc.returncode != 0:
            self.log.write(f"[{tag}] агент завершился с кодом {proc.returncode}, лог: {log_path}")
        if self.cfg.metrics_regex:
            m = re.search(self.cfg.metrics_regex, proc.stdout or "")
            if m:
                return int(m.group(1))
        return None

    # ── валидация выходов ──────────────────────────────────────────────────

    def _mask(self, path: Path) -> None:
        if not path.is_file():
            return
        events = secrets_mask.mask_file(path)
        for e in events:
            note = f"[маскирование] {path}:{e['line']} — {e['kind']}"
            self.log.write(note)
            self.mask_events.append(note)

    def validate_stage_output(self, stage: str, out_files: list[Path],
                              module: str | None = None) -> list[CheckResult]:
        """Механический гейт этапа. Пустой список ошибок = задача принята."""
        results: list[CheckResult] = []
        existing = [p for p in out_files if p.is_file()]
        if not existing:
            r = CheckResult(check="выход этапа")
            r.findings.append(Finding(
                artifact=" | ".join(str(p) for p in out_files),
                path="(файл)",
                expected="агент записывает артефакт по указанному пути",
                got="файл не создан",
                hint="ответь записью файла, а не сообщением в чат",
            ))
            return [r]
        for p in existing:
            self._mask(p)

        schema_kind = {"0": "inventory", "1": "module", "2": "domain",
                       "3": "interface", "4": "rules", "92": "predictions"}.get(stage)
        if schema_kind:
            for p in existing:
                results.append(schema_check.validate_file(p, schema_kind))

        if stage == "4":
            for p in existing:
                results.append(evidence_binding.check_rules_file(p, self.cfg.repo_path))
            results.append(id_allocator.check_collisions(self.rules_dir()))

        if stage == "7":
            results.append(completeness.check(
                self.rules_dir(),
                self.ws / "final" / "stories.md",
                self.ws / "final" / "traceability.csv",
                HERE / "templates" / "traceability_matrix.csv",
            ))

        if stage == "90":
            for p in existing:
                if p.suffix == ".csv":
                    results.append(schema_check.validate_file(p, "validation"))
        return results

    # ── одна задача: промпт -> агент -> проверки -> ретраи ─────────────────

    def run_task(self, stage: str, tag: str, variables: dict[str, str],
                 out_files: list[Path], module: str | None = None) -> TaskOutcome:
        # идемпотентность: валидный выход без --force не перегенерируется
        if not self.force and any(p.is_file() for p in out_files):
            pre = self.validate_stage_output(stage, out_files, module)
            if not any(r.errors for r in pre):
                self.log.write(f"[{tag}] выход валиден — пропуск (--force для перегенерации)")
                out = TaskOutcome(tag=tag, status="пропущен")
                self.outcomes.append(out)
                return out

        if self.dry_run:
            self.log.write(f"[{tag}] dry-run: задача запланирована")
            out = TaskOutcome(tag=tag, status="dry-run")
            self.outcomes.append(out)
            return out

        prompt_text = self.render_prompt(stage, variables)
        tokens_total = 0
        errors_text = ""
        for attempt in range(self.cfg.retries + 1):
            attempt_tag = tag if attempt == 0 else f"{tag}.retry{attempt}"
            text = prompt_text
            if attempt > 0:
                text = (
                    prompt_text
                    + "\n\n## ИСПРАВЛЕНИЕ ОШИБОК (попытка "
                    + f"{attempt} из {self.cfg.retries})\n\n"
                    + "Предыдущий выход не прошёл механические проверки "
                    + "детерминированного слоя. Исправь артефакт и перезапиши "
                    + "его целиком по тому же пути. Ошибки:\n\n"
                    + errors_text + "\n"
                )
            tokens = self.call_agent(stage, attempt_tag, text, out_files[0])
            if tokens:
                tokens_total += tokens
            results = self.validate_stage_output(stage, out_files, module)
            errors_text = render_for_retry(results)
            warn_count = sum(len(r.warnings) for r in results)
            if warn_count:
                self.log.write(f"[{attempt_tag}] предупреждений: {warn_count} (не блокируют)")
            if not errors_text:
                self.log.write(f"[{attempt_tag}] проверки пройдены")
                out = TaskOutcome(tag=tag, status="ок", retries=attempt,
                                  tokens=tokens_total or None)
                self.outcomes.append(out)
                return out
            self.log.write(f"[{attempt_tag}] проверки НЕ пройдены:\n{errors_text}")

        self.log.write(f"[{tag}] ретраи исчерпаны — ЭСКАЛАЦИЯ (нужен человек)")
        out = TaskOutcome(tag=tag, status="эскалация", retries=self.cfg.retries,
                          tokens=tokens_total or None, errors=errors_text)
        self.outcomes.append(out)
        return out

    # ── этапы ──────────────────────────────────────────────────────────────

    def stage_0(self) -> None:
        v = self.base_variables()
        v["OUT_FILE"] = str(self.inventory_file())
        self.run_task("0", "00_inventory", v, [self.inventory_file()])

    def stage_1(self, modules: list[str]) -> None:
        for m in modules:
            out = self.extracts() / "modules" / f"{self.module_basename(m)}.yaml"
            v = self.base_variables()
            v["MODULE_PATH"] = m
            v["OUT_FILE"] = str(out)
            self.run_task("1", f"01_module_{self.module_basename(m)}", v, [out], m)

    def stage_2(self) -> None:
        out = self.extracts() / "domain.yaml"
        v = self.base_variables()
        v["OUT_FILE"] = str(out)
        self.run_task("2", "02_domain", v, [out])

    def stage_3(self) -> None:
        out_openapi = self.extracts() / "api.openapi.yaml"
        out_cli = self.extracts() / "api.cli.yaml"
        v = self.base_variables()
        v["OUT_FILE_OPENAPI"] = str(out_openapi)
        v["OUT_FILE_CLI"] = str(out_cli)
        self.run_task("3", "03_interface", v, [out_openapi, out_cli])

    def stage_4(self, modules: list[str]) -> None:
        # блоки ID выдаются кодом ДО запуска агентов: модули не читают чужую
        # нумерацию (известный провал: два модуля взяли R-101…R-104)
        assigned: dict[str, int] = {}
        next_start = id_allocator.next_block_start(self.rules_dir())
        for m in modules:
            out = self.rules_dir() / f"{self.module_basename(m)}.yaml"
            existing = id_allocator.used_ids(out.parent) if out.is_file() else {}
            own = sorted(n for n, files in existing.items() if out.name in files)
            if own:
                # у модуля уже есть блок — при ретрае он нумерует в нём же
                assigned[m] = own[0]
            else:
                assigned[m] = next_start
                next_start += id_allocator.BLOCK
        for m in modules:
            out = self.rules_dir() / f"{self.module_basename(m)}.yaml"
            v = self.base_variables()
            v["MODULE_PATH"] = m
            v["OUT_FILE"] = str(out)
            v["ID_START"] = f"{assigned.get(m, 1):03d}"
            self.run_task("4", f"04_rules_{self.module_basename(m)}", v, [out], m)

    def stage_5(self) -> None:
        out = self.ws / "integration" / "flows.md"
        v = self.base_variables()
        v["OUT_FILE"] = str(out)
        self.run_task("5", "05_flows", v, [out])

    def stage_6(self) -> None:
        out = self.ws / "final" / "stories.md"
        v = self.base_variables()
        v["OUT_FILE"] = str(out)
        self.run_task("6", "06_stories", v, [out])

    def stage_7(self) -> None:
        out_srs = self.ws / "final" / "SRS.md"
        out_trace = self.ws / "final" / "traceability.csv"
        v = self.base_variables()
        v["OUT_FILE"] = str(out_srs)
        v["OUT_FILE_TRACE"] = str(out_trace)
        self.run_task("7", "07_srs", v, [out_srs, out_trace])

    def stage_92(self) -> None:
        """Инверсная верификация: по statement предсказывается код, предсказание
        механически сверяется с фактом (checks/inverse_match.py).

        Метрика качества формулировок, дополняющая этап 90; пороги приёмки в v3
        не калибруются — этап только собирает статистику.
        """
        if not self.cfg.inverse:
            self.log.write("[92] инверсная верификация выключена (validation.inverse: false) — пропуск")
            return
        if self.cfg.command_for("92") == self.cfg.command_for("4") and not self.dry_run:
            self.log.write(
                "[92] ОСТАНОВ: команда агента этапа 92 совпадает с этапом 4. "
                "Предсказывает НЕ та модель, что извлекала правила — задайте "
                "agent.by_stage['92'] в config.yaml.")
            self.outcomes.append(TaskOutcome(
                tag="92_inverse", status="эскалация",
                errors="кросс-вендорное правило: агент этапа 92 совпадает с этапом 4"))
            return

        inverse_dir = self.ws / "validation" / "inverse"
        inverse_dir.mkdir(parents=True, exist_ok=True)
        for rules_file in sorted(self.rules_dir().glob("*.yaml")):
            base = rules_file.stem
            data = yaml.safe_load(rules_file.read_text(encoding="utf-8")) or {}
            statements = {
                "capability": data.get("module", base),
                # модели отдаются ТОЛЬКО формулировки: без evidence и source
                # предсказание не вырождается в пересказ цитат
                "rules": [{"id": r.get("id"), "statement": r.get("statement")}
                          for r in (data.get("rules") or []) if isinstance(r, dict)],
            }
            if not statements["rules"]:
                continue
            st_file = inverse_dir / f"{base}.statements.yaml"
            st_file.write_text(
                yaml.safe_dump(statements, allow_unicode=True, sort_keys=False),
                encoding="utf-8")
            out = inverse_dir / f"{base}.predictions.yaml"
            v = self.base_variables()
            v["CAPABILITY"] = statements["capability"]
            v["STATEMENTS_FILE"] = str(st_file)
            v["OUT_FILE"] = str(out)
            self.run_task("92", f"92_inverse_{base}", v, [out])

        if not self.dry_run and any(inverse_dir.glob("*.predictions.yaml")):
            from checks import inverse_match
            report = inverse_match.analyze(self.cfg.repo_path, inverse_dir)
            out_md = self.ws / "reports" / "inverse_match.md"
            out_md.write_text(inverse_match.render_md(report), encoding="utf-8")
            self.log.write(
                f"[92] подтверждено предсказаний {report['confirmed']}/{report['total']}"
                f" = {report['share_pct']}% — {out_md} (пороги не калибруются, v3.1)")

    def stage_89(self) -> None:
        out = self.ws / "validation" / "docs_instructions.md"
        v = self.base_variables()
        v["OUT_FILE"] = str(out)
        self.run_task("89", "89_docs", v, [out])

    def stage_90(self) -> None:
        # кросс-вендорная валидация: этап 90 обязан идти другой командой
        if self.cfg.command_for("90") == self.cfg.command_for("4") and not self.dry_run:
            self.log.write(
                "[90] ОСТАНОВ: команда агента этапа 90 совпадает с этапом 4. "
                "Валидацию гоняет НЕ та модель, что извлекала правила — задайте "
                "agent.by_stage['90'] в config.yaml.")
            self.outcomes.append(TaskOutcome(
                tag="90_validate", status="эскалация",
                errors="кросс-вендорное правило: агент этапа 90 совпадает с этапом 4"))
            return

        # слой 1 повторно, кодом, ПЕРЕД этапом 90
        layer1_results = evidence_binding.check_rules_dir(self.rules_dir(), self.cfg.repo_path)
        layer1_report = self.ws / "validation" / "layer1_report.md"
        failed = render_for_retry(layer1_results)
        total = sum(r.stats.get("rules_total", 0) for r in layer1_results)
        bad = sum(len(r.errors) for r in layer1_results)
        layer1_report.write_text(
            "# Слой 1 (привязка) — выполнен кодом\n\n"
            f"Правил: {total}, провалов привязки: {bad}.\n\n"
            + (("## Провалы слоя 1 (вердикт НЕ ОТНОСИТСЯ без слоя 2)\n\n" + failed)
               if failed else "Все правила прошли слой 1.\n"),
            encoding="utf-8")
        if bad:
            self.log.write(f"[90] слой 1: {bad} провалов — правила вернутся на этап 4 "
                           f"(см. {layer1_report})")

        out_csv = self.ws / "validation" / "validation.csv"
        out_md = self.ws / "validation" / "validation.md"
        v = self.base_variables()
        v["LAYER1_REPORT"] = str(layer1_report)
        v["OUT_FILE_CSV"] = str(out_csv)
        v["OUT_FILE_MD"] = str(out_md)
        self.run_task("90", "90_validate", v, [out_csv, out_md])
        self._check_threshold(out_csv)

    def _check_threshold(self, csv_path: Path) -> None:
        if not csv_path.is_file():
            return
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return
        confirmed = sum(1 for r in rows if (r.get("verdict") or "").strip() == "ПОДТВЕРЖДЕНО")
        pct = round(100 * confirmed / len(rows), 1)
        status = "ДОСТИГНУТ" if pct >= self.cfg.stage90_pct else "НЕ достигнут"
        self.log.write(f"[90] подтверждено {confirmed}/{len(rows)} = {pct}% "
                       f"(порог {self.cfg.stage90_pct}%: {status})")
        if pct < self.cfg.stage90_pct:
            self.log.write("[90] порог не достигнут: неподтверждённые правила "
                           "возвращаются на этап 4 (исправлять правила по коду, "
                           "а не вердикты)")

    # ── полнота и сводка ───────────────────────────────────────────────────

    def coverage_report(self, modules: list[str]) -> dict | None:
        if not self.rules_dir().is_dir() or not any(self.rules_dir().glob("*.yaml")):
            return None
        report = coverage.analyze(self.cfg.repo_path, self.rules_dir(), modules or None)
        out = self.ws / "reports" / "coverage.md"
        out.write_text(coverage.render_md(report), encoding="utf-8")
        self.log.write(f"[полнота] покрытие ветвлений {report['coverage_pct']}% — {out}")
        return report

    def write_summary(self, cov: dict | None) -> None:
        tmpl = (HERE / "reports" / "run_summary.md.tmpl").read_text(encoding="utf-8")
        rows = "\n".join(
            f"| {o.tag} | {o.status} | {o.retries} | {o.tokens if o.tokens is not None else '—'} |"
            for o in self.outcomes)
        escalations = [o for o in self.outcomes if o.status == "эскалация"]
        esc_text = "\n".join(
            f"### {o.tag}\n\n{o.errors or '(см. run.log)'}\n" for o in escalations) or "Нет."
        tokens_known = [o.tokens for o in self.outcomes if o.tokens]
        acc = self._accuracy_line()
        text = (tmpl
                .replace("{DATE}", _dt.datetime.now().isoformat(timespec="seconds"))
                .replace("{REPO}", self.cfg.repo_name)
                .replace("{TASK_ROWS}", rows)
                .replace("{ESCALATIONS}", esc_text)
                .replace("{TOKENS}", str(sum(tokens_known)) if tokens_known else
                         "агент не отдал метрик")
                .replace("{ACCURACY}", acc)
                .replace("{COVERAGE}", f"{cov['coverage_pct']}% "
                         f"({cov['branches_covered']}/{cov['branches_total']} веток, "
                         "подробно — reports/coverage.md)" if cov else "не считалась")
                .replace("{MASKED}", "\n".join(self.mask_events) or "Секретов не найдено."))
        out = self.ws / "run_summary.md"
        out.write_text(text, encoding="utf-8")
        self.log.write(f"[сводка] {out}")

    def _accuracy_line(self) -> str:
        csv_path = self.ws / "validation" / "validation.csv"
        if not csv_path.is_file():
            return "этап 90 не прогнан"
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return "validation.csv пуст"
        confirmed = sum(1 for r in rows if (r.get("verdict") or "").strip() == "ПОДТВЕРЖДЕНО")
        pct = round(100 * confirmed / len(rows), 1)
        status = "ДОСТИГНУТ" if pct >= self.cfg.stage90_pct else "НЕ достигнут"
        return f"{confirmed}/{len(rows)} = {pct}% (порог {self.cfg.stage90_pct}%: {status})"

    # ── главный цикл ───────────────────────────────────────────────────────

    def run(self, stages: list[str]) -> int:
        self.log.write(f"=== прогон req-reverse: {self.cfg.repo_name}, этапы {','.join(stages)} ===")
        # комплект не привязан к конкретному LLM/CLI: команду агента задаёт
        # пользователь; без неё останавливаемся сразу, а не ретраями
        if not self.dry_run and not self.cfg.agent_command.strip():
            self.log.write(
                "ОСТАНОВ: agent.command в config.yaml не заполнен. Впишите "
                "команду вашего CLI-агента (промпт — через {prompt_text}, "
                "{prompt_file} или stdin: true); примеры под конкретные "
                "инструменты — prompts/RUNBOOK.md, раздел «Настройка CLI-агента».")
            return 2
        if "0" in stages:
            self.stage_0()
        modules = self.modules_from_inventory()
        if modules:
            self.log.write(f"[план] модулей для этапов 1/4: {len(modules)}: "
                           + ", ".join(modules))
        elif {"1", "4"} & set(stages):
            self.log.write("[план] инвентаризация не дала модулей — этапы 1/4 пропущены")

        # ярусная глубина: план фильтруется картой tiers (умолчание вне карты —
        # tiers.default; без секции tiers — полный прогон, как в базовой версии)
        module_tiers = {m: self.tier_of(m) for m in modules}
        l2_modules = [m for m in modules if module_tiers[m] == "L2"]
        max_tier = max(
            [self.cfg.tier_default] + list(module_tiers.values()),
            key=lambda t: TIER_RANK[t])
        if self.cfg.tier_map or self.tier_overrides() or self.cfg.tier_default != "L2":
            self.log.write(
                "[ярусы] " + ", ".join(f"{m}: {t}" for m, t in module_tiers.items())
                + f"; максимальный ярус прогона: {max_tier}")

        dispatch = {
            "1": lambda: self.stage_1(modules),
            "2": self.stage_2,
            "3": self.stage_3,
            "4": lambda: self.stage_4(l2_modules),
            "5": self.stage_5,
            "6": self.stage_6,
            "7": self.stage_7,
            "89": self.stage_89,
            "90": self.stage_90,
            "92": self.stage_92,
        }
        for s in stages:
            if s == "0":
                continue
            fn = dispatch.get(s)
            if fn is None:
                self.log.write(f"[план] этап {s} runner'ом не оркестрируется — пропуск")
                continue
            if not self.stage_allowed(s, max_tier):
                self.log.write(f"[план] этап {s} пропущен: требует яруса "
                               f"{STAGE_MIN_TIER[s]}, максимум прогона — {max_tier}")
                continue
            fn()

        cov = self.coverage_report(l2_modules or modules) if not self.dry_run else None
        if not self.dry_run:
            self.write_summary(cov)
        escal = [o for o in self.outcomes if o.status == "эскалация"]
        self.log.write(f"=== итог: задач {len(self.outcomes)}, "
                       f"эскалаций {len(escal)} ===")
        return 1 if escal else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Оркестратор req-reverse v3")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help="список этапов через запятую (по умолчанию 0–7,90)")
    ap.add_argument("--force", action="store_true",
                    help="перегенерировать даже валидные выходы")
    ap.add_argument("--dry-run", action="store_true",
                    help="построить план без вызова агента")
    ap.add_argument("--raise-tier", nargs=2, metavar=("ПУТЬ", "ЯРУС"),
                    help="поднять ярус зоны (L0|L1|L2) и догнать недостающие "
                         "этапы; готовые артефакты нижних ярусов не перегенерируются")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    runner = Runner(cfg, force=args.force, dry_run=args.dry_run)
    if args.raise_tier:
        path, tier = args.raise_tier
        runner.raise_tier(path, tier)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    return runner.run(stages)


if __name__ == "__main__":
    sys.exit(main())
