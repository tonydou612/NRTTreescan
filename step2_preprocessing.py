#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Preprocessing & counts for aggregated data (icd10, k, count).

This module operates on long-format CSVs:
  - events:   icd10, k, count        (vaccinated cohort events per time window k)
  - baseline: icd10, k, count        (all-pop baseline per time window k) [optional]

What it computes (aligned to your method):
  - leaf list L (ICD-10 4th level) & group list G (ICD-10 3rd level)
  - per-look totals z^(k) and cumulative U^(k)
  - leaf counts c_l^(k)  → array shape (K, L)
  - cumulative leaf counts m_l^(k) = sum_{j=1..k} c_l^(j)
  - EXTERNAL q_l^(k) from baseline per-look (if provided) and time-invariant weights w_l from baseline (sum over k)
  - INTERNAL q_l^(k) = m_l^(k)/U^(k)

It also provides helpers to aggregate from leaves to groups (3rd level) using
a sparse membership matrix GxL.

CLI example:
  python step2_preprocessing.py \
      --events /path/events.csv \
      --baseline /path/baseline.csv \
      --out-npz /path/prepared.npz \
      --summary-csv /path/summary.csv
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

ICD3_PATTERN = re.compile(r'^([A-Z][0-9]{2})')  # captures 3-char block like 'A01'


# ------------------------- ICD-10 utilities -------------------------
def icd10_level3(code: str) -> str:
    """
    Extract ICD-10 level-3 block (e.g., 'A01' from 'A01.0', 'A01.01', 'A01').
    Fallback: first 3 valid chars if regex fails.
    """
    if not isinstance(code, str):
        code = str(code)
    m = ICD3_PATTERN.match(code)
    if m:
        return m.group(1)
    import re as _re
    clean = _re.sub(r'[^A-Za-z0-9]', '', code.upper())
    return clean[:3] if len(clean) >= 3 else clean


def get_leaves(df: pd.DataFrame) -> List[str]:
    """Sorted unique leaf codes (assumes df['icd10'] are level-4)."""
    return sorted(df['icd10'].astype(str).unique().tolist())


def build_group_maps(leaves: List[str]) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    """
    Returns:
      leaf_to_group_idx: mapping leaf_code -> group_index (0..G-1)
      group_to_leaf_idx: mapping group_key  -> list of leaf indices (in leaves array order)
    """
    group_keys = [icd10_level3(code) for code in leaves]
    uniq_groups = sorted(set(group_keys))
    group_index = {g: i for i, g in enumerate(uniq_groups)}
    leaf_to_group_idx = {code: group_index[icd10_level3(code)] for code in leaves}
    group_to_leaf_idx: Dict[str, List[int]] = {g: [] for g in uniq_groups}
    for i, code in enumerate(leaves):
        g = icd10_level3(code)
        group_to_leaf_idx[g].append(i)
    return leaf_to_group_idx, group_to_leaf_idx


# ------------------------- pivot & counts -------------------------
def pivot_counts_long(df: pd.DataFrame, leaves: List[str]) -> Tuple[np.ndarray, List[int]]:
    """
    Convert long table (icd10, k, count) to array of shape (K, L) aligned to sorted leaves,
    with rows ordered by sorted unique k (ascending). Missing entries -> 0.
    Returns (counts, k_list).
    """
    df = df.copy()
    df['icd10'] = df['icd10'].astype(str)
    df['k'] = df['k'].astype(int)
    df['count'] = df['count'].astype(int)

    k_list = sorted(df['k'].unique().tolist())
    idx = pd.MultiIndex.from_product([k_list, leaves], names=['k', 'icd10'])
    df_wide = (df.set_index(['k', 'icd10'])['count']
                 .reindex(idx, fill_value=0)
                 .unstack('icd10')
                 .loc[k_list, leaves])
    counts = df_wide.to_numpy(dtype=int)
    return counts, k_list


def cumulative_along_k(counts_kl: np.ndarray) -> np.ndarray:
    """Cumulative sum along k axis (axis=0)."""
    return counts_kl.cumsum(axis=0)


def totals_per_k(counts_kl: np.ndarray) -> np.ndarray:
    """z^(k): sum over leaves for each k (axis=1 over L)."""
    return counts_kl.sum(axis=1)


def cumulative_totals(z_k: np.ndarray) -> np.ndarray:
    """U^(k) = cumulative sum of z^(k) over k."""
    return z_k.cumsum()


# ------------------------- q from baseline/internal -------------------------
def q_from_external_baseline_per_k(baseline_long: pd.DataFrame, leaves: List[str]) -> Tuple[np.ndarray, List[int]]:
    """
    For each k, normalize baseline counts over leaves to get q_l^(k) (shape K x L).
    Missing rows → uniform over leaves.
    """
    base_counts, k_list = pivot_counts_long(baseline_long, leaves)
    row_sums = base_counts.sum(axis=1, keepdims=True).astype(float)
    q = np.zeros_like(base_counts, dtype=float)
    nonzero = (row_sums[:, 0] > 0)
    q[nonzero, :] = base_counts[nonzero, :] / row_sums[nonzero]
    if np.any(~nonzero):
        L = base_counts.shape[1]
        q[~nonzero, :] = 1.0 / L
    return q, k_list


def time_invariant_weights_from_baseline(baseline_long: pd.DataFrame, leaves: List[str]) -> np.ndarray:
    """
    Aggregate baseline across k to get time-invariant weights w_l (sum=1).
    """
    g = baseline_long.groupby('icd10', as_index=False)['count'].sum()
    g = g.set_index('icd10').reindex(leaves, fill_value=0.0)
    w = g['count'].to_numpy(dtype=float)
    total = w.sum()
    if total <= 0:
        w = np.full(len(leaves), 1.0 / len(leaves), dtype=float)
    else:
        w = w / total
    return w


def q_from_internal_cumulative(m_kl: np.ndarray, U_k: np.ndarray) -> np.ndarray:
    """
    q_l^(k) = m_l^(k) / U^(k); shape K x L. For rows with U_k==0, set uniform.
    """
    K, L = m_kl.shape
    q = np.zeros((K, L), dtype=float)
    for i in range(K):
        if U_k[i] > 0:
            q[i, :] = m_kl[i, :] / float(U_k[i])
        else:
            q[i, :] = 1.0 / L
    return q


# ------------------------- grouping (leaf → group) -------------------------
def build_group_membership(leaves: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Constructs a full hierarchy membership matrix M (G x L).
    Each leaf belongs to Chapter (A), Category (A01), and itself (A01.0).
    """
    node_to_leaves = {}
    for i, leaf in enumerate(leaves):
        # Extract ancestors: Chapter, Category, and Leaf itself
        # leaf.split('.')[0] handles cases like 'A01.0' -> 'A01'
        ancestors = [leaf[0], leaf.split('.')[0], leaf]
        for p in set(ancestors):
            if p not in node_to_leaves:
                node_to_leaves[p] = []
            node_to_leaves[p].append(i)
            
    all_nodes = sorted(node_to_leaves.keys())
    G, L = len(all_nodes), len(leaves)
    M = np.zeros((G, L), dtype=int)
    for gi, node in enumerate(all_nodes):
        for li in node_to_leaves[node]:
            M[gi, li] = 1
    return all_nodes, M


def aggregate_to_groups(counts_kl: np.ndarray, M_gl: np.ndarray) -> np.ndarray:
    """
    c_G^(k) = sum_{l in G} c_l^(k)  →  counts_kl (K x L), M (G x L)
    Returns K x G matrix.
    """
    return counts_kl @ M_gl.T


# ------------------------- container for results -------------------------
@dataclass
class Prepared:
    leaves: List[str]
    groups: List[str]
    k_list: List[int]
    counts_leaf: np.ndarray      # K x L
    counts_group: np.ndarray     # K x G
    cumulative_leaf: np.ndarray  # K x L (m_l^(k))
    z_k: np.ndarray              # K
    U_k: np.ndarray              # K
    q_external: Optional[np.ndarray]   # K x L or None
    w_external: Optional[np.ndarray]   # L or None
    q_internal: np.ndarray       # K x L


def prepare_from_long(
    events_long: pd.DataFrame,
    baseline_long: Optional[pd.DataFrame] = None,
) -> Prepared:
    """
    Main entry: given long tables, compute arrays and mappings.
    """
    leaves = get_leaves(events_long)
    counts_leaf, k_list = pivot_counts_long(events_long, leaves)

    # totals
    z_k = totals_per_k(counts_kl=counts_leaf)
    U_k = cumulative_totals(z_k)

    # cumulative leaf counts
    m_kl = cumulative_along_k(counts_leaf)

    # group membership & aggregation
    groups, M_gl = build_group_membership(leaves)
    counts_group = aggregate_to_groups(counts_leaf, M_gl)

    # q (external, internal)
    if baseline_long is not None:
        q_ext, k_list_base = q_from_external_baseline_per_k(baseline_long, leaves)
        if k_list_base != k_list:
            # reindex baseline to event's k_list
            idx = pd.MultiIndex.from_product([k_list, leaves], names=['k','icd10'])
            base_wide = (baseline_long.set_index(['k','icd10'])['count']
                           .reindex(idx, fill_value=0)
                           .unstack('icd10')
                           .loc[k_list, leaves])
            bc = base_wide.to_numpy(dtype=int)
            row_sums = bc.sum(axis=1, keepdims=True).astype(float)
            q_ext = np.divide(bc, row_sums, out=np.zeros_like(bc, dtype=float), where=row_sums>0)
        w_ext = time_invariant_weights_from_baseline(baseline_long, leaves)
    else:
        q_ext = None
        w_ext = None

    # internal q from cumulative
    q_int = q_from_internal_cumulative(m_kl, U_k)

    return Prepared(
        leaves=leaves, groups=groups, k_list=k_list,
        counts_leaf=counts_leaf, counts_group=counts_group,
        cumulative_leaf=m_kl, z_k=z_k, U_k=U_k,
        q_external=q_ext, w_external=w_ext, q_internal=q_int
    )


# ------------------------- CLI -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 2 preprocessing for TreeScan (aggregated data).")
    ap.add_argument("--events", type=str, required=True, help="Path to events CSV (icd10,k,count).")
    ap.add_argument("--baseline", type=str, default=None, help="Optional path to baseline CSV (icd10,k,count).")
    ap.add_argument("--out-npz", type=str, default=None, help="Optional NPZ to save prepared arrays.")
    ap.add_argument("--summary-csv", type=str, default=None, help="Optional CSV summary of per-look totals (z_k, U_k).")
    args = ap.parse_args()

    events = pd.read_csv(args.events)
    baseline = pd.read_csv(args.baseline) if args.baseline else None
    prep = prepare_from_long(events, baseline)

    # Save NPZ if requested
    if args.out_npz:
        np.savez_compressed(
            args.out_npz,
            leaves=np.array(prep.leaves, dtype=object),
            groups=np.array(prep.groups, dtype=object),
            k_list=np.array(prep.k_list, dtype=int),
            counts_leaf=prep.counts_leaf,
            counts_group=prep.counts_group,
            cumulative_leaf=prep.cumulative_leaf,
            z_k=prep.z_k,
            U_k=prep.U_k,
            q_external=prep.q_external if prep.q_external is not None else np.array([]),
            w_external=prep.w_external if prep.w_external is not None else np.array([]),
            q_internal=prep.q_internal,
        )

    # Summary CSV if requested
    if args.summary_csv:
        df_sum = pd.DataFrame({
            'k': prep.k_list,
            'z_k': prep.z_k,
            'U_k': prep.U_k,
        })
        df_sum.to_csv(args.summary_csv, index=False)

    # Console summary
    print(f"L={len(prep.leaves)}, G={len(prep.groups)}, K={len(prep.k_list)}")
    print(f"Totals  sum(z_k)={int(prep.z_k.sum())},  U_K(last)={int(prep.U_k[-1])}")
    if prep.w_external is not None:
        print("External baseline provided: w_external and q_external(k) are available.")
    else:
        print("No external baseline provided: only q_internal(k) is available.")

if __name__ == "__main__":
    main()