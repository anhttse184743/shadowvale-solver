"""Localhost sidecar the Unity client calls during play.

    GET  /health    liveness + which variants are implemented
    GET  /variants  registry listing
    POST /solve     QuboRequest -> CoordinationPlan within latency_budget_ms

Contract with the game (UC-12 exception flows):
  * the response is ALWAYS a feasible assignment;
  * if the requested variant is unknown/unavailable → greedy, fallback_reason set;
  * if the variant does not finish inside the budget → greedy result,
    hit_latency_budget = true (the game records a solver_timeout event);
  * the budget counts from when the request arrives. Research solves run one at a time in a
    worker *process* — in threads they shared the GIL with each other and with the parsing of the
    next request (about 230 kB of JSON for 6 agents × 23 nodes), so two at once took 130–225 ms
    for a 120 ms budget (measured 04/10/2026). A request that had to wait gets only what is left
    of its budget, and greedy when too little is left;
  * the server binds to 127.0.0.1 only; there is no auth by design.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import threading
import time

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..qubo.formulation import QuboProblem
from ..solvers.registry import REGISTRY, get_solver, list_variants
from ..solvers.greedy import GreedySolver
from .schemas import CoordinationPlan, Health, QuboRequest, VariantInfo

# One research solve at a time, in one worker process. The lane is released when the solve really
# ends — a variant that overran its budget keeps it until it stops, never slowing the next unseen.
_LANE = threading.Lock()
_POOL: cf.ProcessPoolExecutor | None = None
MIN_SOLVE_MS = 10             # less budget than this left after queueing → greedy instead


def _pool() -> cf.ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = cf.ProcessPoolExecutor(max_workers=1, initializer=_worker_init, initargs=(os.getpid(),))
    return _POOL


def _worker_init(server_pid: int) -> None:
    # The worker ends with the server even when the server is killed outright (it then has no
    # chance to shut the pool down) — otherwise a packaged sidecar leaves it, and the launcher
    # waiting to delete the files the worker still holds, running after the game is gone.
    from .__main__ import exit_with
    exit_with(server_pid)


def _run(variant: str, problem: QuboProblem, budget_ms: int, seed: int):
    return get_solver(variant).solve(problem, budget_ms, seed)


def warm_up() -> None:
    """Start the worker and import the solvers in it, so the game's first solve is not the slow one."""
    tiny = QuboProblem.from_flat([-1.0, 2.0, 0.0, -1.0], 1, 2, instance_id="warm-up")
    with _LANE:
        _pool().submit(_run, "qiea", tiny, 5, 0).result(timeout=60)


def create_app() -> FastAPI:
    app = FastAPI(title="ShadowVale Solver Sidecar", version=__version__)

    @app.on_event("startup")
    def _start_worker() -> None:
        warm_up()

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(version=__version__,
                      variants_available=[v["id"] for v in list_variants() if v["available"]])

    @app.get("/variants", response_model=list[VariantInfo])
    def variants() -> list[VariantInfo]:
        return [VariantInfo(**v) for v in list_variants()]

    @app.post("/solve", response_model=CoordinationPlan)
    def solve(req: QuboRequest) -> CoordinationPlan:
        t0 = time.perf_counter()     # the budget counts from here, queueing included
        n = req.agent_count * req.node_count
        if len(req.q_matrix) != n * n:
            raise HTTPException(status_code=422,
                                detail=f"q_matrix has {len(req.q_matrix)} entries, expected {n * n}")
        if req.agent_count > req.node_count:
            raise HTTPException(status_code=422, detail="agent_count exceeds node_count: infeasible")

        problem = QuboProblem.from_flat(req.q_matrix, req.agent_count, req.node_count,
                                        offset=req.offset, lam=req.penalty_conflict,
                                        instance_id=req.instance_id)
        fallback_reason: str | None = None
        variant = req.variant
        if variant not in REGISTRY or not REGISTRY[variant].available:
            fallback_reason = f"variant '{variant}' unavailable, used greedy"
            variant = "greedy"

        hit = False
        if variant == "greedy":
            sol = GreedySolver().solve(problem, req.latency_budget_ms, req.seed)
        else:
            sol, hit, reason = _solve_in_lane(variant, problem, req, t0)
            fallback_reason = reason or fallback_reason
        elapsed = (time.perf_counter() - t0) * 1000.0
        hit = hit or elapsed > req.latency_budget_ms

        return CoordinationPlan(
            agent_to_target=sol.assignment,
            objective_value=-sol.energy,
            solve_latency_ms=round(elapsed, 3),
            solver_variant_id=sol.variant,
            hit_latency_budget=hit,
            feasible_raw=sol.feasible,
            energy=sol.energy,
            fallback_reason=fallback_reason,
            instance_id=req.instance_id,
            samples_evaluated=int(sol.meta.get("samples_evaluated", 0)),
            raw_feasible_samples=int(sol.meta.get("raw_feasible_samples", 0)),
        )

    return app


def _solve_in_lane(variant: str, problem: QuboProblem, req: QuboRequest, t0: float):
    """(solution, hit_budget, fallback_reason) — the variant with what is left of the budget, or greedy."""
    budget = req.latency_budget_ms
    left = lambda: budget - (time.perf_counter() - t0) * 1000.0
    greedy = lambda: GreedySolver().solve(problem, budget, req.seed)
    if not _LANE.acquire(timeout=max(0.0, (left() - MIN_SOLVE_MS) / 1000.0)):
        return greedy(), True, f"variant '{variant}' queued behind other solves for its whole budget, used greedy"
    remaining = left()
    if remaining < MIN_SOLVE_MS:
        _LANE.release()
        return greedy(), True, f"variant '{variant}' queued behind other solves for its whole budget, used greedy"
    try:
        fut = _pool().submit(_run, variant, problem, max(1, int(remaining)), req.seed)
    except BaseException:
        _LANE.release()
        raise
    fut.add_done_callback(lambda _: _LANE.release())
    try:
        return fut.result(timeout=remaining / 1000.0), False, None
    except cf.TimeoutError:
        return greedy(), True, f"variant '{variant}' exceeded {budget} ms, used greedy"
    except ValueError as e:      # e.g. brute force too large
        return greedy(), False, f"variant '{variant}' failed ({e}), used greedy"
    except cf.process.BrokenProcessPool:
        _reset_pool()
        return greedy(), True, f"variant '{variant}' worker crashed, used greedy"


def _reset_pool() -> None:
    global _POOL
    _POOL = None


app = create_app()
