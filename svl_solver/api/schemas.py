"""Wire contracts with the Unity client. Field names are snake_case and must
stay in sync with ``ShadowVale.Data.Coordination.QuboRequest`` /
``CoordinationPlan`` (Unity deserialises with a snake_case naming strategy).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QuboRequest(BaseModel):
    instance_id: str = ""
    seed: int = 0
    agent_count: int = Field(gt=0)
    node_count: int = Field(gt=0)
    agent_nodes: list[int] = Field(default_factory=list)
    cover_nodes: list[int] = Field(default_factory=list)
    escape_routes: list[int] = Field(default_factory=list)
    q_matrix: list[float] = Field(description="row-major (agent_count*node_count)^2 upper-triangular or symmetric")
    offset: float = 0.0
    weight_coverage: float = 1.0
    weight_redundancy: float = 0.5
    weight_flanking: float = 1.0
    weight_cover: float = 0.5
    weight_distance: float = 0.1
    penalty_conflict: float = 0.0
    latency_budget_ms: int = Field(default=120, gt=0)
    variant: str = "greedy"

    @field_validator("q_matrix")
    @classmethod
    def _non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("q_matrix must not be empty")
        return v


class CoordinationPlan(BaseModel):
    agent_to_target: list[int]
    objective_value: float
    solve_latency_ms: float
    solver_variant_id: str
    hit_latency_budget: bool
    feasible_raw: bool = True
    energy: float = 0.0
    fallback_reason: str | None = None
    instance_id: str = ""
    # Sampler metrics (sqa / qiea / qaoa); 0 for constructive solvers like greedy.
    samples_evaluated: int = 0
    raw_feasible_samples: int = 0


class VariantInfo(BaseModel):
    id: str
    available: bool
    family: str
    due: str | None = None


class Health(BaseModel):
    status: str = "ok"
    version: str
    variants_available: list[str]
