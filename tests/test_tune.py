"""bench/tune.py: equal-effort tuning — same trials per solver, defaults included, reproducible."""

import numpy as np

from svl_solver.bench.tune import SPACES, is_tuning_instance, sample, tune
from svl_solver.instances.generator import generate
from svl_solver.qubo.formulation import build_qubo


def _problems(k=3):
    return [(f"g{s}", build_qubo(generate(seed=s, n_nodes=8, n_agents=3, chokepoint_density=0.3))) for s in range(k)]


def test_every_solver_gets_the_same_number_of_trials_and_its_defaults_are_one_of_them():
    best, trials, info = tune(_problems(), ["ga", "sa"], trials=4, budget_ms=30, finalists=2, repeats=1, progress=None)
    for s in ("ga", "sa"):
        search = [t for t in trials if t.solver == s and t.stage == "search"]
        assert len(search) == 4
        assert search[0].params == {}, "trial 0 is the current defaults"
        assert len([t for t in trials if t.solver == s and t.stage == "final"]) in (2, 3)
        assert 0 <= best[s]["mean_gap"] <= best[s]["default_mean_gap"] + 1e-12, "never worse than the defaults on the same final runs"
        assert any(t.trial == 0 for t in trials if t.solver == s and t.stage == "final"), "the defaults reach the final"
    assert all(info["proven"].values()), "8-node instances: the exact solver proves every reference"


def test_tuning_is_reproducible_from_its_seed():
    a, _, _ = tune(_problems(2), ["qiea"], trials=3, budget_ms=20, finalists=1, repeats=1, seed=5, progress=None)
    b, _, _ = tune(_problems(2), ["qiea"], trials=3, budget_ms=20, finalists=1, repeats=1, seed=5, progress=None)
    assert a["qiea"]["params"] == b["qiea"]["params"]


def test_search_spaces_sample_inside_their_bounds():
    rng = np.random.default_rng(0)
    for solver, space in SPACES.items():
        for _ in range(50):
            p = sample(space, rng)
            for name, spec in space.items():
                if spec[0] == "choice":
                    assert p[name] in spec[1]
                else:
                    assert spec[1] - 1e-9 <= p[name] <= spec[2] + 1e-9, (solver, name, p[name])


def test_the_tuning_split_is_a_stable_quarter_of_the_ids():
    ids = [f"Map 1-abc-{i}" for i in range(4000)]
    share = sum(is_tuning_instance(i) for i in ids) / len(ids)
    assert 0.2 < share < 0.3
    assert [is_tuning_instance(i) for i in ids[:50]] == [is_tuning_instance(i) for i in ids[:50]]
