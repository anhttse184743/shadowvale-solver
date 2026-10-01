"""Exports the sidecar's wire contract for the Unity contract test (M10).

The fixture lists every field of ``QuboRequest`` and ``CoordinationPlan`` (and which are
required), the solver ids the sidecar knows, and a real ``/solve`` response. The C# test
serialises its own ``QuboRequest`` and parses the response, and fails when the two sides drift
apart — a renamed field would otherwise only show up as a silent greedy fallback in game.

    python -m svl_solver.tools.export_contract <out.json>
"""

from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

from ..api.app import create_app
from ..api.schemas import CoordinationPlan, QuboRequest
from ..instances.generator import generate
from ..instances.io import to_qubo_request
from ..solvers.registry import REGISTRY


def _fields(model) -> dict:
    schema = model.model_json_schema()
    return {"fields": sorted(schema["properties"]), "required": sorted(schema.get("required", []))}


def contract() -> dict:
    inst = generate(seed=5, n_nodes=6, n_agents=2, chokepoint_density=0.3)
    request = to_qubo_request(inst, latency_budget_ms=120, variant="qiea")
    request["instance_id"] = "contract-sample"
    response = TestClient(create_app()).post("/solve", json=request)
    response.raise_for_status()
    return {
        "qubo_request": _fields(QuboRequest),
        "coordination_plan": _fields(CoordinationPlan),
        "variants": sorted(REGISTRY),
        "sample_request": request,
        "sample_plan": response.json(),
    }


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if argv else "solver_contract.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(contract(), f, indent=1)
        f.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
