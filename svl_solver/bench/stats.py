"""Statistical comparison between solver variants — planned for P4/P7 (W9, W14).

Intended API (needs the [plots] extra for scipy):

    compare(rows, "sqa", "greedy", metric="approx_ratio")
        -> {"n": ..., "mean_a": ..., "mean_b": ..., "mann_whitney_p": ..., "ci95_diff": (lo, hi)}

Pairing is by (instance_id, repeat) so both solvers are compared on identical
instances and seeds (paired Wilcoxon), falling back to Mann–Whitney U when the
pairing is incomplete.
"""

from __future__ import annotations


def compare(rows: list[dict], variant_a: str, variant_b: str, metric: str = "approx_ratio") -> dict:
    raise NotImplementedError("bench.stats.compare is scheduled for W9 (needs scipy via the [plots] extra)")
