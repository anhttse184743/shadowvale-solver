"""bench.replay: the Unity replay harness's runs, summarised and compared per bot and variant.

The fixture is real output of the harness (Map 1, camp C2, greedy, 6 seeds × 3 bots).
"""

from pathlib import Path

import pytest

from svl_solver.bench.replay import (audit, compare_family, compare_variants, format_summary, load_replay, summarize,
                                     wilson)

FIXTURE = Path(__file__).parent / "fixtures" / "replay_sample.jsonl"


def test_real_harness_output_loads_and_summarises():
    rows = load_replay(FIXTURE)
    assert len(rows) == 18
    assert {r["bot"] for r in rows} == {"same_route", "same_bush", "stone_lure"}
    summary = {s["bot"]: s for s in summarize(rows)}
    assert all(s["runs"] == 6 and s["variant"] == "greedy" for s in summary.values())
    for s in summary.values():
        lo, hi = s["capture_ci95"]
        assert 0 <= lo <= s["capture_rate"] <= hi <= 1
    assert "same_bush" in format_summary(list(summary.values()))


def test_wilson_interval_is_sane_at_the_extremes():
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0.2 < hi < 0.35
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1) and 0.65 < lo < 0.8
    assert wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-3)


def _run(bot, variant, seed, captured, hp, valid=True, plans=3, lockstep=True):
    return {"bot": bot, "variant": variant, "seed": seed, "captured": 1.0 if captured else 0.0,
            "outcome": "captured" if captured else "escaped", "player_hp": hp, "escape_time_mean": 0 if captured else 8.0,
            "coordination_score": 0.6, "plan_entropy_bits": 2.0, "valid_for_comparison": valid,
            "encounters": 1, "plans": plans, "lockstep_latency": lockstep, "plans_discarded": 0, "first_plan_delay_s": 0.1}


def test_variants_are_compared_on_paired_seeds():
    rows = []
    for seed in range(12):
        rows.append(_run("same_route", "greedy", seed, captured=False, hp=60))
        rows.append(_run("same_route", "qiea", seed, captured=seed % 2 == 0, hp=20 if seed % 2 == 0 else 40))
    r = compare_variants(rows, "qiea", "greedy", bot="same_route", metric="player_hp")
    assert r["test"] == "wilcoxon" and r["n_pairs"] == 12
    assert r["mean_diff"] < 0 and r["p_value"] < 0.01, "qiea leaves the bot with less HP on every seed"
    assert r["better_for_guards"] == "lower"
    c = compare_variants(rows, "qiea", "greedy", bot="same_route", metric="captured")
    assert c["mean_a"] == pytest.approx(0.5) and c["mean_b"] == 0


def test_runs_not_valid_for_comparison_are_left_out_by_default():
    rows = [_run("same_bush", "sqa", 1, True, 0, valid=False), _run("same_bush", "sqa", 2, False, 50)]
    assert summarize(rows)[0]["runs"] == 1
    assert summarize(rows, include_invalid=True)[0]["runs"] == 2


def test_audit_flags_runs_without_lockstep_and_variants_that_never_got_a_plan_through():
    # exp-v1's first replay: remote plans arrived late and were thrown away, the guards fought plan-less.
    rows = [_run("same_route", "greedy", s, False, 60, lockstep=False) for s in range(10)]
    rows += [_run("same_route", "qiea", s, True, 0, plans=0 if s < 7 else 2, lockstep=False) for s in range(10)]
    rows += [_run("same_route", "none", s, True, 0, plans=0, lockstep=False) for s in range(10)]
    warnings = audit(rows)
    assert any("without lockstep" in w for w in warnings)
    assert any("qiea: 70%" in w for w in warnings)
    assert not any(" none:" in w for w in warnings), "the control never plans by design"
    assert {s["variant"]: s["no_plan"] for s in summarize(rows, True)} == {"greedy": 0, "qiea": .7, "none": 0}
    assert audit([_run("same_route", v, 1, False, 50) for v in ("greedy", "qiea")]) == []


def test_every_variant_is_compared_with_each_baseline_under_one_holm_family():
    rows = []
    for seed in range(12):
        rows.append(_run("same_bush", "none", seed, True, 0))
        rows.append(_run("same_bush", "greedy", seed, False, 50))
        rows.append(_run("same_bush", "qiea", seed, seed % 3 == 0, 30))
    res = compare_family(rows, ["greedy", "none"], metric="player_hp", bot="same_bush")
    assert [(r["a"], r["b"]) for r in res] == [("none", "greedy"), ("qiea", "greedy"), ("qiea", "none")]
    assert all(r["p_holm"] >= r["p_value"] for r in res)
