"""Localhost sidecar the Unity client calls during play.

    GET  /health    liveness + which variants are implemented
    GET  /variants  registry listing
    POST /solve     QuboRequest -> CoordinationPlan within latency_budget_ms

Contract with the game (UC-12 exception flows):
  * the response is ALWAYS a feasible assignment;
  * if the requested variant is unknown/unavailable → greedy, fallback_reason set;
  * if the variant does not finish inside the budget → greedy result,
    hit_latency_budget = true (the game records a solver_timeout event);
  * the server binds to 127.0.0.1 only; there is no auth by design.
"""

from __future__ import annotations

import concurrent.futures as cf
import time

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..qubo.formulation import QuboProblem
from ..solvers.registry import REGISTRY, get_solver, list_variants
from ..solvers.greedy import GreedySolver
from .schemas import CoordinationPlan, Health, QuboRequest, VariantInfo

_POOL = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="svl-solve")


def create_app() -> FastAPI:
    app = FastAPI(title="ShadowVale Solver Sidecar", version=__version__)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(version=__version__,
                      variants_available=[v["id"] for v in list_variants() if v["available"]])

    @app.get("/variants", response_model=list[VariantInfo])
    def variants() -> list[VariantInfo]:
        return [VariantInfo(**v) for v in list_variants()]

    @app.post("/solve", response_model=CoordinationPlan)
    def solve(req: QuboRequest) -> CoordinationPlan:
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

        t0 = time.perf_counter()
        hit = False
        try:
            fut = _POOL.submit(get_solver(variant).solve, problem, req.latency_budget_ms, req.seed)
            sol = fut.result(timeout=req.latency_budget_ms / 1000.0)
        except cf.TimeoutError:
            hit = True
            fallback_reason = f"variant '{variant}' exceeded {req.latency_budget_ms} ms, used greedy"
            sol = GreedySolver().solve(problem, req.latency_budget_ms, req.seed)
        except ValueError as e:      # e.g. brute force too large
            fallback_reason = f"variant '{variant}' failed ({e}), used greedy"
            sol = GreedySolver().solve(problem, req.latency_budget_ms, req.seed)
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


app = create_app()
