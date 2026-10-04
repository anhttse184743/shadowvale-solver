"""The sidecar ends with the game: started with --parent-pid, it exits when that process does."""

import subprocess
import sys
import time
import urllib.request

PORT = 8019


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as r:
            return r.status == 200
    except OSError:
        return False


def test_the_sidecar_exits_when_its_parent_process_ends():
    game = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    api = subprocess.Popen([sys.executable, "-m", "svl_solver.api", "--port", str(PORT), "--log-level", "warning",
                            "--parent-pid", str(game.pid)])
    try:
        for _ in range(60):
            if _healthy():
                break
            time.sleep(.5)
        assert _healthy(), "the sidecar came up"
        game.kill()
        game.wait()
        api.wait(timeout=10)                  # raises if it is still running
        assert not _healthy(), "nothing answers on the port any more"
    finally:
        for p in (game, api):
            if p.poll() is None:
                p.kill()
