#!/usr/bin/env python3
"""CLI слежения за игроком.

Примеры:
  # цель задана рамкой на кадре 0 (координаты исходного кадра), номер 10
  python track.py match.MOV --init-box 1210,540,1290,720 --number 10 --out out/

  # цель задана точкой (клик) на 3-й секунде
  python track.py match.MOV --init-point 1250,640 --start-sec 3 --out out/

  # интерактивный выбор: окно с первым кадром, клик по игроку, Enter
  python track.py match.MOV --select --out out/

Выход в --out: track.json (по кадрам), track.csv, events.log, summary.json,
annotated.mp4 (если не --no-video).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracker import export  # noqa: E402
from tracker.appearance import CompositeEncoder, PartColorEncoder  # noqa: E402
from tracker.pipeline import InitSpec, Pipeline, PipelineConfig  # noqa: E402
from tracker.render import draw  # noqa: E402
from tracker.video import FrameSource, probe  # noqa: E402


def parse_floats(s: str, n: int) -> tuple[float, ...]:
    vals = tuple(float(v) for v in s.split(","))
    if len(vals) != n:
        raise argparse.ArgumentTypeError(f"ожидалось {n} чисел через запятую, получено {len(vals)}")
    return vals


def build_encoder(kind: str):
    if kind == "color":
        return PartColorEncoder()
    if kind == "osnet":
        from tracker.appearance import TorchReidEncoder

        return CompositeEncoder([(TorchReidEncoder(), 1.0), (PartColorEncoder(), 0.6)])
    raise SystemExit(f"неизвестный кодировщик: {kind}")


def build_reader(kind: str):
    if kind == "none":
        return None
    if kind == "easyocr":
        from tracker.jersey import EasyOcrReader

        return EasyOcrReader()
    raise SystemExit(f"неизвестный OCR: {kind}")


def interactive_select(frame) -> tuple[float, float]:
    import cv2

    picked: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked.clear()
            picked.append((float(x), float(y)))

    win = "Кликните по игроку, затем Enter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        if picked:
            cv2.circle(view, (int(picked[0][0]), int(picked[0][1])), 8, (0, 255, 0), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and picked:
            break
        if key == 27:
            raise SystemExit("выбор отменён")
    cv2.destroyWindow(win)
    return picked[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Слежение за одним игроком на видео (iPhone 15 Pro Max).")
    ap.add_argument("video")
    ap.add_argument("--out", default="out", help="каталог результатов")
    ap.add_argument("--init-box", type=lambda s: parse_floats(s, 4), help="x1,y1,x2,y2 в координатах исходного кадра")
    ap.add_argument("--init-point", type=lambda s: parse_floats(s, 2), help="x,y в координатах исходного кадра")
    ap.add_argument("--select", action="store_true", help="интерактивный выбор кликом (нужен GUI)")
    ap.add_argument("--init-frame", type=int, default=0, help="кадр (в обработанном фрагменте), на котором задана цель")
    ap.add_argument("--number", help="номер целевого игрока, если известен")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--width", type=int, default=1280, help="ширина кадра обработки (0 — исходная)")
    ap.add_argument("--weights", default="yolov8n.pt", help="веса YOLO (ultralytics)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda:0")
    ap.add_argument("--encoder", choices=["color", "osnet"], default="color")
    ap.add_argument("--ocr", choices=["none", "easyocr"], default="none")
    ap.add_argument("--no-team", action="store_true", help="не разделять по цвету формы")
    ap.add_argument("--no-video", action="store_true", help="не писать annotated.mp4")
    ap.add_argument("--tonemap", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--detector", choices=["yolo"], default="yolo")
    args = ap.parse_args(argv)

    info = probe(args.video)
    print(f"Видео: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.codec} {info.pix_fmt} "
          f"{'HDR ' + info.color_transfer if info.is_hdr else 'SDR'}, поворот {info.rotation}°, "
          f"{'VFR' if info.is_vfr else 'CFR'}, ~{info.nb_frames} кадров")
    tonemap = None if args.tonemap == "auto" else args.tonemap == "on"
    source = FrameSource(args.video, max_width=args.width or None, tonemap=tonemap,
                         start_sec=args.start_sec, max_frames=args.max_frames)
    scale = source.scale
    print(f"Обработка: {source.width}x{source.height} (масштаб {scale:.3f}), тонмаппинг {'вкл' if source.tonemap else 'выкл'}")

    init = InitSpec(frame=args.init_frame, number=args.number)
    if args.init_box:
        init.box = tuple(v * scale for v in args.init_box)
    elif args.init_point:
        init.point = (args.init_point[0] * scale, args.init_point[1] * scale)
    elif args.select:
        first = next(iter(FrameSource(args.video, max_width=args.width or None, tonemap=tonemap,
                                      start_sec=args.start_sec, max_frames=1)))
        init.point = interactive_select(first)
        init.frame = 0
    else:
        ap.error("задайте цель: --init-box, --init-point или --select")

    from tracker.detection import YoloDetector

    detector = YoloDetector(args.weights, imgsz=args.imgsz, device=args.device)
    cfg = PipelineConfig(init=init, use_team=not args.no_team)
    pipe = Pipeline((source.width, source.height), detector, build_encoder(args.encoder), build_reader(args.ocr), cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        import cv2

        writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), source.fps,
                                 (source.width, source.height))
    records = []
    t_start = time.perf_counter()
    last_print = t_start

    def on_frame(frame, res):
        nonlocal last_print
        records.append(export.observation_to_dict(res.observation, scale, source.fps))
        if writer is not None:
            writer.write(draw(frame, res.observation, res.tracks))
        if res.observation.event:
            print(f"[{res.frame_idx:6d} {res.frame_idx / source.fps:7.2f}s] {res.observation.event} track={res.observation.track_id}")
        now = time.perf_counter()
        if now - last_print > 5:
            fps_proc = (res.frame_idx + 1) / (now - t_start)
            print(f"  кадр {res.frame_idx}, {fps_proc:.1f} кадр/с, состояние {res.observation.state.value}")
            last_print = now

    try:
        pipe.run(source, on_frame)
    finally:
        if writer is not None:
            writer.release()

    meta = {"video": str(args.video), "fps": source.fps, "frame_size": [info.width, info.height],
            "process_size": [source.width, source.height], "scale": scale, "init": vars(init),
            "encoder": args.encoder, "ocr": args.ocr, "weights": args.weights}
    export.write_json(out / "track.json", records, meta)
    export.write_csv(out / "track.csv", records)
    export.write_events(out / "events.log", records, source.fps)
    summary = export.summarize(records)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Итог:", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
