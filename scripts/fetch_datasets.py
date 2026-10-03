"""Downloads the standard external datasets, checks them and unpacks them into datasets/external.

    .venv\Scripts\python scripts\fetch_datasets.py

The files are not in the repository: they belong to their authors. This script gets the exact
same bytes (SHA-256 checked) from the official sources, so the supplementary benchmarks can be
reproduced anywhere.

* QAPLIB — Burkard, Karisch & Rendl (1997), J. Global Optimization 10:391-403.
  Official mirror: https://coral.ise.lehigh.edu/data-sets/qaplib/
* Moving AI grid benchmarks — Sturtevant (2012), IEEE Trans. Computational Intelligence and AI in
  Games 4(2):144-148. https://movingai.com/benchmarks/grids.html — maps from Dragon Age: Origins,
  StarCraft, Warcraft III and Baldur's Gate II.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "datasets" / "external"

FILES = [
    ("qaplib/qapdata.tar.gz", "https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/qapdata.tar.gz",
     "c6bf0e277156d097df21124f53ca5a5d0fa5344481ee401e13acb67694e190b3", "qaplib/data"),
    ("qaplib/qapsoln.tar.gz", "https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/qapsoln.tar.gz",
     "002243dea9452000ec7ee5e43bd0aa651a639eb61039c4f35679531e67d71744", "qaplib/soln"),
    ("movingai/dao-map.zip", "https://movingai.com/benchmarks/dao/dao-map.zip",
     "229393af31caaec732e5f4fd923940845ab056e94dadfb96b8afc24da472d3af", "movingai/dao"),
    ("movingai/sc1-map.zip", "https://movingai.com/benchmarks/sc1/sc1-map.zip",
     "81285dee93dbd7594773e7292a4984a89da6531eb45424a90c99aa2f4144e091", "movingai/sc1"),
    ("movingai/wc3maps512-map.zip", "https://movingai.com/benchmarks/wc3maps512/wc3maps512-map.zip",
     "12ef51dea736c0dedfd4fc0a04234751af27c7247cb4ae53e9d0ff467976c17a", "movingai/wc3maps512"),
    ("movingai/bg512-map.zip", "https://movingai.com/benchmarks/bg512/bg512-map.zip",
     "babe4274380df5a757913ed520b988375a80879aefb43acc875caa2b32d1ed71", "movingai/bg512"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for rel, url, digest, unpack in FILES:
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or sha256(path) != digest:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        if sha256(path) != digest:
            raise SystemExit(f"{rel}: checksum mismatch — the source changed; check before using it")
        target = ROOT / unpack
        target.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                z.extractall(target)
        else:
            with tarfile.open(path) as t:
                t.extractall(target)
        print(f"ok  {rel}  -> {target}")


if __name__ == "__main__":
    main()
