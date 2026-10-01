"""PyInstaller entry point of the packaged sidecar (svl-solver.exe): same as `svl-api`."""

from svl_solver.cli import api_main

if __name__ == "__main__":
    api_main()
