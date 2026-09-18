"""Benchmark suites: the fixed grids of (nodes × agents × density × seeds)
that every solver is run on. Defined in YAML under experiments/configs/ and
mirrored here as built-ins so tests and the API never depend on file paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

from ..model.instance import Instance, Weights
from .generator import generate


@dataclass
class SuiteSpec:
    name: str
    nodes: list[int]
    agents: list[int]
    densities: list[float]
    seeds: list[int]
    budget_ms: int = 120
    brute_force_max_candidates: int = 200_000   # exact optimum only when N!/(N−A)! ≤ this
    vision_hops: int = 2
    weights: Weights = field(default_factory=Weights)
    extra: dict = field(default_factory=dict)

    @property
    def n_instances(self) -> int:
        return len(self.nodes) * len(self.agents) * len(self.densities) * len(self.seeds)

    @staticmethod
    def from_dict(d: dict) -> "SuiteSpec":
        d = dict(d)
        w = d.pop("weights", None)
        return SuiteSpec(
            name=str(d.pop("name")),
            nodes=[int(v) for v in d.pop("nodes")],
            agents=[int(v) for v in d.pop("agents")],
            densities=[float(v) for v in d.pop("densities")],
            seeds=[int(v) for v in _expand_seeds(d.pop("seeds"))],
            budget_ms=int(d.pop("budget_ms", 120)),
            brute_force_max_candidates=int(d.pop("brute_force_max_candidates", 200_000)),
            vision_hops=int(d.pop("vision_hops", 2)),
            weights=Weights.from_dict(w) if w else Weights(),
            extra=d,
        )


def _expand_seeds(v) -> list[int]:
    """Accept [1,2,3] or {"start": 1, "count": 30}."""
    if isinstance(v, dict):
        start = int(v.get("start", 1))
        count = int(v["count"])
        return list(range(start, start + count))
    return [int(s) for s in v]


def load_suite(path: str | Path) -> SuiteSpec:
    with open(path, "r", encoding="utf-8") as f:
        return SuiteSpec.from_dict(yaml.safe_load(f))


def iter_instances(spec: SuiteSpec) -> Iterator[Instance]:
    for n in spec.nodes:
        for a in spec.agents:
            if a >= n:
                continue
            for dens in spec.densities:
                for s in spec.seeds:
                    yield generate(seed=s, n_nodes=n, n_agents=a,
                                   chokepoint_density=dens, vision_hops=spec.vision_hops)


BUILTIN_SUITES: dict[str, SuiteSpec] = {
    "tiny": SuiteSpec(name="tiny", nodes=[6, 8], agents=[2], densities=[0.3], seeds=[1, 2, 3],
                      ),
    "small": SuiteSpec(name="small", nodes=[6, 8, 10], agents=[2, 3], densities=[0.0, 0.3, 0.6],
                       seeds=list(range(1, 11))),
    "medium": SuiteSpec(name="medium", nodes=[20, 40, 60], agents=[4, 6, 8], densities=[0.0, 0.3, 0.6],
                        seeds=list(range(1, 31))),
    "full": SuiteSpec(name="full", nodes=[20, 40, 60, 80, 100], agents=[4, 8, 12],
                      densities=[0.0, 0.3, 0.6], seeds=list(range(1, 31))),
}
