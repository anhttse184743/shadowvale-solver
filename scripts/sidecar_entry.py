"""PyInstaller entry point of the packaged sidecar (svl-solver.exe): same as `svl-api`."""

import multiprocessing

from svl_solver.cli import api_main

if __name__ == "__main__":
    multiprocessing.freeze_support()   # the solver worker is a child process of the frozen exe
    api_main()
