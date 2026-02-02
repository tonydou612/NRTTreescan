
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Expected counts & test statistics on the ICD-10 tree (group level by default).

Inputs (preferred): a prepared NPZ from step2_preprocessing.py containing:
  - leaves (L,), groups (G,), k_list (K,)
  - counts_leaf (K x L), counts_group (K x G)
  - cumulative_leaf (K x L), z_k (K,), U_k (K,)
  - q_external (K x L) [optional], w_external (L,) [optional]
  - q_internal (K x L)

This script computes, per time window k and per scanning unit G (ICD-10 3rd-level by default):
  - observed counts:          c_G^(k)
  - expected counts:          u_G^(k)   (from either external weights, external per-k q, or internal cumulative q)
  - Poisson log-likelihood ratio (LLR): T^(k)(G)  [Eq. (4) with one-sided I(c>u)]
  - window-wise maxima:       Λ_max^(k) and the top-J clusters with their stats

By default scanning units are 3rd-level groups (recommended by your method). You may switch to leaves.

Author: GPT-5 Thinking
"""
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Dict
import numpy as np
import pandas as pd


# ----------------------------- dataclass -----------------------------
@dataclass
class Step3Config:
    baseline_mode: Literal["ext_w", "ext_qk", "internal"] = "ext_w"
    scan_level:    Literal["groups", "leaves"] = "groups"
    top_j: int = 2


# ----------------------------- helpers -----------------------------
def _safe_llr_one_sided(c: np.ndarray, u: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Vectorized Poisson log-likelihood ratio (one-sided for increases):
      T = c*ln(c/u) + (C-c)*ln((C-c)/(C-u))  if c>u and 0<u<C
        = 0 otherwise
    Inputs can be broadcast arrays. Returns array of same shape as c/u.
    """
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    C = np.asarray(C, dtype=float)
    T = np.zeros_like(c, dtype=float)

    # Valid mask for one-sided increase
    valid = (u > 0) & (C > u) & (c > u) & (C >= c)
    if not np.any(valid):
        return T

    cv = c[valid]; uv = u[valid]; Cv = C[valid]
    # Avoid log(0); we already ensured uv>0, Cv>uv, Cv>=cv
    term1 = cv * np.log(cv / uv)
    # For cv == Cv, the second term becomes (C-c)*ln((C-c)/(C-u)) with (C-c)=0 → 0 * ln(0/...) → 0 by limit
    with np.errstate(divide='ignore', invalid='ignore'):
        numer = np.maximum(Cv - cv, 0.0)
        denom = np.maximum(Cv - uv, 1e-15)  # strictly positive
        ratio = np.divide(numer, denom, out=np.ones_like(numer), where=denom>0)
        term2 = numer * np.log(ratio, where=(numer>0))
        term2 = np.nan_to_num(term2, nan=0.0, posinf=0.0, neginf=0.0)
    T[valid] = term1 + term2
    return T


def _aggregate_q_to_groups(q_kl: np.ndarray, membership_GxL: np.ndarray) -> np.ndarray:
    """Aggregate per-leaf q_l^(k) (K x L) to group q_G^(k) (K x G)."""
    return q_kl @ membership_GxL.T   # (K x L) @ (L x G) -> (K x G)


# ----------------------------- core computations -----------------------------
def compute_expected(
    mode: str,
    z_k: np.ndarray,
    U_k: np.ndarray,
    # leaf-level inputs
    counts_leaf: np.ndarray,
    cumulative_leaf: np.ndarray,
    q_external_leaf: Optional[np.ndarray],
    w_external_leaf: Optional[np.ndarray],
    # group-level inputs
    counts_group: Optional[np.ndarray] = None,
    membership_GxL: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute expected counts u for either groups or leaves depending on provided arrays.
    Returns:
      - u (K x M) where M = number of scanning units (G for groups or L for leaves)
      - c (K x M) observed counts aligned with u
    """
    K = z_k.shape[0]

    if counts_group is not None:
        # ----- scanning on groups -----
        c = counts_group  # K x G

        if mode == "ext_w":
            if w_external_leaf is None or membership_GxL is None:
                raise ValueError("ext_w requires w_external (leaf) and membership_GxL.")
            # w_G = sum_{l in G} w_l ; u_G^(k) = w_G * z^(k)
            wG = membership_GxL @ w_external_leaf  # (G x L) @ (L,) -> (G,)
            u = np.outer(z_k, wG)                  # (K,) x (G,) -> (K x G)

        elif mode == "ext_qk":
            if q_external_leaf is None or membership_GxL is None:
                raise ValueError("ext_qk requires q_external (K x L) and membership_GxL.")
            qG = _aggregate_q_to_groups(q_external_leaf, membership_GxL)  # (K x G)
            u = qG * z_k[:, None]

        elif mode == "internal":
            # u_G^(k) = m_G^(k) * z^(k) / C^(k)
            if membership_GxL is None:
                raise ValueError("internal requires membership_GxL to build m_G from leaf cumulative.")
            mG = cumulative_leaf @ membership_GxL.T  # (K x L) @ (L x G) -> (K x G)
            u = np.zeros_like(mG, dtype=float)
            # For each k: u_k = mG_k * z_k / U_k
            for k in range(K):
                if U_k[k] > 0:
                    u[k, :] = mG[k, :] * (z_k[k] / float(U_k[k]))
                else:
                    u[k, :] = 0.0
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")

    else:
        # ----- scanning on leaves -----
        c = counts_leaf  # K x L

        if mode == "ext_w":
            if w_external_leaf is None:
                raise ValueError("ext_w requires w_external at leaf level.")
            u = z_k[:, None] * w_external_leaf[None, :]  # (K,1) * (1,L)

        elif mode == "ext_qk":
            if q_external_leaf is None:
                raise ValueError("ext_qk requires q_external (K x L).")
            u = q_external_leaf * z_k[:, None]

        elif mode == "internal":
            # u_l^(k) = m_l^(k) * z^(k) / C^(k)
            mL = cumulative_leaf  # (K x L)
            u = np.zeros_like(mL, dtype=float)
            for k in range(K):
                if U_k[k] > 0:
                    u[k, :] = mL[k, :] * (z_k[k] / float(U_k[k]))
                else:
                    u[k, :] = 0.0
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")

    return u, c


def compute_T_and_top(
    u: np.ndarray,
    c: np.ndarray,
    U_k: np.ndarray,
    unit_labels: List[str],
    top_j: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given expected u (K x M), observed c (K x M), and cumulative totals U_k (K,),
    compute T^(k)(unit) and extract per-k top-J.

    Returns:
      - stats_long: long table with columns [k, unit, c, u, T]
      - lambda_summary: per k summary with Λ_max and top-J rows:
            k, j_rank, unit, c, u, T
    """
    K, M = u.shape
    # Broadcast C^(k) to shape (K x M)
    C = np.repeat(U_k[:, None], M, axis=1)
    T = _safe_llr_one_sided(c, u, C)

    # Long stats
    rows = []
    for k in range(K):
        for j in range(M):
            rows.append((int(k+1), unit_labels[j], int(c[k, j]), float(u[k, j]), float(T[k, j])))
    stats_long = pd.DataFrame(rows, columns=["k", "unit", "c", "u", "T"])

    # Top-J per k
    lambda_rows = []
    for k in range(K):
        # sort descending by T
        idx = np.argsort(-T[k, :])
        for r in range(min(top_j, M)):
            j = idx[r]
            lambda_rows.append((int(k+1), int(r+1), unit_labels[j], int(c[k, j]), float(u[k, j]), float(T[k, j])))
    lambda_summary = pd.DataFrame(lambda_rows, columns=["k", "rank", "unit", "c", "u", "T"])

    return stats_long, lambda_summary


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 3: expected counts & LLR stats on ICD-10 tree.")
    ap.add_argument("--prepared", type=str, required=True, help="Path to prepared NPZ from Step 2.")
    ap.add_argument("--baseline-mode", type=str, choices=["ext_w", "ext_qk", "internal"],
                    default="ext_w", help="Use external weights (time-invariant), external per-k q, or internal cumulative q.")
    ap.add_argument("--scan-level", type=str, choices=["groups", "leaves"], default="groups",
                    help="Scan on ICD-10 3rd-level groups (default) or on leaves.")
    ap.add_argument("--top-j", type=int, default=2, help="How many top clusters per k to save.")
    ap.add_argument("--stats-out", type=str, required=True, help="Output CSV for long stats (k,unit,c,u,T).")
    ap.add_argument("--lambda-out", type=str, required=True, help="Output CSV for per-k top J (Λ_max etc.).")
    args = ap.parse_args()

    data = np.load(args.prepared, allow_pickle=True)

    leaves = list(data["leaves"])
    k_list = list(data["k_list"])
    counts_leaf = data["counts_leaf"]
    cumulative_leaf = data["cumulative_leaf"]
    z_k = data["z_k"]
    U_k = data["U_k"]

    # Optional arrays
    counts_group = data.get("counts_group", None)
    if counts_group is not None:
        counts_group = counts_group
    q_external = data.get("q_external", None)
    if q_external is not None and q_external.size == 0:
        q_external = None
    w_external = data.get("w_external", None)
    if w_external is not None and w_external.size == 0:
        w_external = None
    q_internal = data.get("q_internal", None)  # not used directly here, but kept if needed

    # Build membership_GxL if scanning on groups
    membership_GxL = None
    unit_labels: List[str]
    if args.scan_level == "groups":
        groups = list(data["groups"])
        unit_labels = groups
        
        # --- New Full Hierarchy Logic ---
        group_index = {g: i for i, g in enumerate(groups)}
        G, L = len(groups), len(leaves)
        membership_GxL = np.zeros((G, L), dtype=int)
        
        for li, leaf in enumerate(leaves):
            # Chapter (A), Category (A01), and Leaf (A01.0)
            ancestors = [leaf[0], leaf.split('.')[0], leaf]
            for p in set(ancestors):
                if p in group_index:
                    membership_GxL[group_index[p], li] = 1
        # --------------------------------
        
        c_units = counts_group if "counts_group" in data.files else None
    else:
        unit_labels = leaves
        c_units = None

    # Compute expected & observed
    u, c = compute_expected(
        mode=args.baseline_mode,
        z_k=z_k, U_k=U_k,
        counts_leaf=counts_leaf,
        cumulative_leaf=cumulative_leaf,
        q_external_leaf=q_external,
        w_external_leaf=w_external,
        counts_group=c_units,
        membership_GxL=membership_GxL,
    )

    # Compute T and top-J
    stats_long, lambda_summary = compute_T_and_top(u, c, U_k, unit_labels, top_j=args.top_j)

    # Save
    stats_long.to_csv(args.stats_out, index=False)
    lambda_summary.to_csv(args.lambda_out, index=False)

    # Small console summary
    print(f"Saved stats to {args.stats_out} (rows={len(stats_long)})")
    print(f"Saved per-k top-{args.top_j} to {args.lambda_out} (rows={len(lambda_summary)})")
    print(f"Baseline mode={args.baseline_mode}, scan level={args.scan_level}")

if __name__ == "__main__":
    main()
