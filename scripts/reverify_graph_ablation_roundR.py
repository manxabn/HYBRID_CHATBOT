"""
Re-verification of the n=31 prerequisite-graph ablation (scripts/ablate_
graph_augmentation_expanded.py), prompted by a direct question about
whether the paper's cited result (measured 2026-08-01 17:51) was still
current. Checked directly rather than assumed: the 2026-08-02 commit
"Rebuild Chroma index, re-measure every vector-dependent table" fixed a
real bug (the live Chroma index still serving old, uncorrected chunk
text) and explicitly re-measured several other tables, but its own
commit message does not list the graph ablation among them -- the only
change to a graph-ablation script in that commit was a safety fix
(--out flag) to the OLDER, superseded n=12 script, not a re-run of the
n=31 expansion this paper actually cites.

Same methodology as ablate_graph_augmentation_expanded.py (imported, not
reimplemented), only the output path overridden so the original,
possibly-stale result is preserved for direct before/after comparison
rather than silently overwritten.

Usage: python scripts/reverify_graph_ablation_roundR.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ablate_graph_augmentation_expanded as base

base.OUT_PATH = ROOT / "results" / "graph_ablation_expanded_raw_roundR.csv"

if __name__ == "__main__":
    base.main()
