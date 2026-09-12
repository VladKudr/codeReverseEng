#!/usr/bin/env python3
"""Запуск веб-приложения: python serve.py [--host 0.0.0.0] [--port 8000] [--data data]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"), help="каталог роликов и задач")
    args = ap.parse_args()
    import uvicorn

    from webapp.server import create_app

    app = create_app(data_dir=args.data)
    print(f"player_tracker: http://{args.host}:{args.port}/  (данные в {args.data})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
