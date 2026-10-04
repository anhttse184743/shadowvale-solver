"""python -m svl_solver.api [--host 127.0.0.1] [--port 8001] [--parent-pid PID]"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import uvicorn


def exit_with(parent_pid: int) -> threading.Thread:
    """End this process (and so its solver worker) when ``parent_pid`` ends — the game passes its own
    id, so quitting or crashing never leaves a sidecar holding the port. Killing the game's handle
    alone is not enough: a packaged sidecar is a launcher with the server as its child."""
    def wait() -> None:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x00100000, False, parent_pid)       # SYNCHRONIZE
            if handle:
                kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)               # INFINITE
                kernel32.CloseHandle(handle)
        else:
            while True:
                try:
                    os.kill(parent_pid, 0)
                except OSError:
                    break
                time.sleep(1.0)
        os._exit(0)

    t = threading.Thread(target=wait, name="svl-parent-watch", daemon=True)
    t.start()
    return t


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="svl-api", description="ShadowVale solver sidecar")
    p.add_argument("--host", default="127.0.0.1", help="bind address (keep it local)")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--log-level", default="info")
    p.add_argument("--parent-pid", type=int, default=0, help="exit when this process ends (the game passes its own)")
    a = p.parse_args(argv)
    if a.parent_pid > 0:
        exit_with(a.parent_pid)
    uvicorn.run("svl_solver.api.app:app", host=a.host, port=a.port, log_level=a.log_level)


if __name__ == "__main__":
    main()
