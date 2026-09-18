"""python -m svl_solver.api [--host 127.0.0.1] [--port 8001]"""

from __future__ import annotations

import argparse

import uvicorn


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="svl-api", description="ShadowVale solver sidecar")
    p.add_argument("--host", default="127.0.0.1", help="bind address (keep it local)")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--log-level", default="info")
    a = p.parse_args(argv)
    uvicorn.run("svl_solver.api.app:app", host=a.host, port=a.port, log_level=a.log_level)


if __name__ == "__main__":
    main()
