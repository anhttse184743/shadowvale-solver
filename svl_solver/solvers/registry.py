"""String id → solver. The id is what Unity puts in ``ai_settings.default_solver_variant``
and what telemetry records as ``ai_variant``."""

from __future__ import annotations

from .base import Solver
from .brute_force import BruteForceSolver
from .ga import GeneticSolver
from .greedy import GreedySolver
from .planned import DWaveQpuSolver
from .qaoa import QaoaSolver
from .qiea import QuantumInspiredEvolutionarySolver
from .sa import ClassicalAnnealingSolver
from .sqa import SimulatedQuantumAnnealingSolver

REGISTRY: dict[str, type[Solver]] = {
    "greedy": GreedySolver,
    "brute": BruteForceSolver,
    "ga": GeneticSolver,
    "sa": ClassicalAnnealingSolver,
    "sqa": SimulatedQuantumAnnealingSolver,
    "qiea": QuantumInspiredEvolutionarySolver,
    "qaoa": QaoaSolver,
    "qpu": DWaveQpuSolver,
}

CLASSICAL = ("greedy", "ga", "sa", "brute")
QUANTUM_INSPIRED = ("sqa", "qiea", "qaoa", "qpu")


def get_solver(variant: str, **kwargs) -> Solver:
    try:
        cls = REGISTRY[variant]
    except KeyError:
        raise KeyError(f"unknown solver variant '{variant}'; known: {sorted(REGISTRY)}") from None
    return cls(**kwargs)


def list_variants() -> list[dict]:
    out = []
    for vid, cls in REGISTRY.items():
        out.append({
            "id": vid,
            "available": bool(cls.available),
            "family": "quantum_inspired" if vid in QUANTUM_INSPIRED else "classical",
            "due": getattr(cls, "due", None) if not cls.available else None,
        })
    return out
