"""QAPLIB — the standard benchmark library of the Quadratic Assignment Problem.

Burkard, Karisch & Rendl (1997), "QAPLIB – A Quadratic Assignment Problem Library",
Journal of Global Optimization 10:391–403; files from the official mirror at
coral.ise.lehigh.edu/data-sets/qaplib (see scripts/fetch_datasets.py).

Why it fits the thesis: a QAP places n facilities on n locations, one each, with a cost for
every *pair* of placements — exactly the structure of our coordination QUBO (one-hot
"one node per agent, one agent per node" plus pairwise terms). It is the hardest-studied
problem of that shape, with published optimal / best-known values, so it measures how the
same solvers (same one-hot encoding, same penalty construction) cope with assignment QUBOs
far from our own generator.

Encoding: x[i,k] = 1 when facility i is on location k (agent i, node k), so every solver runs
unchanged. For a feasible x the energy is exactly the QAP cost Σ_ij a_ij·b_p(i)p(j) (offset
included), so the gap to the published value is (E − best)/best.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..qubo.formulation import QuboProblem


@dataclass
class QapInstance:
    name: str
    n: int
    a: np.ndarray          # first matrix of the .dat file (flow, in most instances)
    b: np.ndarray          # second matrix (distance)
    best: float | None     # optimal / best-known cost from the .sln file
    best_perm: list[int] | None   # 0-based location of each facility

    def cost(self, perm: list[int]) -> float:
        p = np.asarray(perm)
        return float((self.a * self.b[np.ix_(p, p)]).sum())


def _numbers(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").split()


def read_dat(path: str | Path) -> tuple[int, np.ndarray, np.ndarray]:
    t = _numbers(Path(path))
    n = int(t[0])
    vals = np.asarray([float(v) for v in t[1:1 + 2 * n * n]])
    if vals.size != 2 * n * n:
        raise ValueError(f"{path}: expected {2 * n * n} matrix entries, found {vals.size}")
    return n, vals[:n * n].reshape(n, n), vals[n * n:].reshape(n, n)


def read_sln(path: str | Path) -> tuple[int, float, list[int] | None]:
    t = _numbers(Path(path))
    n, value = int(t[0]), float(t[1])
    perm = [int(v) - 1 for v in t[2:2 + n]] if len(t) >= 2 + n else None
    return n, value, perm


def load(data_dir: str | Path, soln_dir: str | Path | None = None, max_n: int | None = None) -> list[QapInstance]:
    """Every instance with a published solution (and n ≤ max_n), sorted by size then name."""
    data_dir = Path(data_dir)
    soln_dir = Path(soln_dir) if soln_dir else data_dir
    out = []
    for dat in sorted(data_dir.glob("*.dat")):
        n, a, b = read_dat(dat)
        if max_n is not None and n > max_n:
            continue
        sln = soln_dir / (dat.stem + ".sln")
        if not sln.exists():
            continue
        sn, best, perm = read_sln(sln)
        if sn != n:
            continue
        inst = QapInstance(dat.stem, n, a, b, best, perm)
        # A few .sln files (kra30a/b, tho30) list the inverse permutation (location → facility).
        if perm is not None and abs(inst.cost(perm) - best) > 1e-6 * max(1.0, abs(best)):
            inverse = [int(v) for v in np.argsort(perm)]
            inst.best_perm = inverse if abs(inst.cost(inverse) - best) <= 1e-6 * max(1.0, abs(best)) else None
        out.append(inst)
    return sorted(out, key=lambda q: (q.n, q.name))


def to_qubo(q: QapInstance) -> QuboProblem:
    """Assignment QUBO with the same penalty structure as the coordination formulation:
    λ·(Σ_k x_ik − 1)² per facility and λ·x_ik·x_jk per shared location. λ exceeds the largest
    total interaction of any single variable, so leaving or sharing a slot never pays."""
    n = q.n
    m = np.einsum("ij,kl->ikjl", q.a, q.b).reshape(n * n, n * n)   # M[(i,k),(j,l)] = a_ij·b_kl
    s = m + m.T
    Q = np.triu(s, 1)
    Q[np.diag_indices_from(Q)] = np.diag(m)                         # a_ii·b_kk
    lam = 1.0 + float(np.abs(np.triu(s, 1) + np.triu(s, 1).T).sum(axis=1).max() + np.abs(np.diag(m)).max())
    idx = lambda i, k: i * n + k
    for i in range(n):
        for k in range(n):
            Q[idx(i, k), idx(i, k)] -= lam                          # row one-hot, linear part
            for l in range(k + 1, n):
                Q[idx(i, k), idx(i, l)] += 2 * lam                  # two locations for one facility
    for k in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                Q[idx(i, k), idx(j, k)] += lam                      # two facilities on one location
    return QuboProblem(Q=Q, offset=lam * n, n_agents=n, n_nodes=n, lam=lam, instance_id=f"qaplib-{q.name}")
