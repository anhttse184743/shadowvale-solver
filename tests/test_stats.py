"""bench.stats: the asymptotic tests match their textbook values (as scipy's method='asymptotic' does)."""

import math

import pytest

from svl_solver.bench.stats import bootstrap_ci, compare, mann_whitney, wilcoxon


def test_mann_whitney_fully_separated_samples():
    u, p = mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert u == 0.0
    # z = (|0 − 12.5| − 0.5) / sqrt(25·11/12) = 2.5067 → p = 0.01219
    assert p == pytest.approx(0.01219, abs=5e-5)


def test_mann_whitney_identical_samples_is_not_significant():
    assert mann_whitney([1, 1, 2, 2], [1, 1, 2, 2])[1] == pytest.approx(1.0)


def test_wilcoxon_all_positive_differences():
    w, p = wilcoxon(list(range(1, 11)))
    assert w == 55.0
    # z = (55 − 27.5 − 0.5) / sqrt(96.25) = 2.7521 → p = 0.005922
    assert p == pytest.approx(0.005922, abs=5e-5)


def test_wilcoxon_drops_zero_differences_and_handles_none_left():
    assert wilcoxon([0.0, 0.0])[1] == 1.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    d = [0.1, 0.2, 0.3, 0.4, 0.5] * 4
    lo, hi = bootstrap_ci(d)
    assert lo < 0.3 < hi
    assert bootstrap_ci(d) == (lo, hi)


def _rows(solver, values):
    return [{"solver": solver, "instance_id": f"i{k}", "repeat": 0, "gap": v} for k, v in enumerate(values)]


def test_compare_pairs_by_instance_and_reports_direction():
    rows = _rows("sqa", [0.0, 0.01, 0.0, 0.02, 0.0, 0.0, 0.01, 0.0]) + _rows("sa", [0.05, 0.04, 0.03, 0.06, 0.05, 0.02, 0.04, 0.03])
    r = compare(rows, "sqa", "sa", metric="gap")
    assert r["test"] == "wilcoxon" and r["n_pairs"] == 8
    assert r["mean_diff"] < 0 and r["ci95_diff"][1] < 0, "sqa has the lower gap"
    assert r["p_value"] < 0.05


def test_compare_falls_back_to_mann_whitney_when_unpaired():
    rows = _rows("a", [1, 2, 3]) + [{"solver": "b", "instance_id": "other", "repeat": 0, "gap": 5}]
    r = compare(rows, "a", "b", metric="gap")
    assert r["test"] == "mann_whitney" and r["n_pairs"] == 0
    assert not math.isnan(r["p_value"])
