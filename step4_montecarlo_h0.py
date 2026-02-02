#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 (integrated): Monte Carlo under H0 via conditional multinomial.
- Reads Step 2's prepared.npz
- Builds leaf-level baseline proportions q_l^(k) per chosen baseline-mode
- Optionally aggregates to ICD-10 level-3 groups for scanning
- Draws B multinomial replicates per look k to get ~Lambda_max^(k,b)
- (Optional) If given observed lambda_top.csv from Step 3, computes p^(k)

Outputs:
  1) --out-null         : CSV of null ~Lambda_max^(k,b) (and optional ~T_(2) if --top2)
  2) --out-null-npz     : NPZ with matrix form lambdamax[K,B] (optional)
  3) --out-pvals        : CSV with p-values per k if --lambda-obs is provided

Usage examples
--------------
# A) Generate null resamples only (group scan, external per-k baseline):
python step4_montecarlo_h0.py \
  --prepared prepared.npz \
  --baseline-mode ext_qk \
  --scan-level groups \
  --B 100000 \
  --seed 123 \
  --out-null null_lambdamax.csv

# B) 同时计算 p 值（读入 Step 3 的 lambda_top.csv，rank=1 为观测 Λ_max）：
python step4_montecarlo_h0.py \
  --prepared prepared.npz \
  --baseline-mode ext_qk \
  --scan-level groups \
  --B 100000 \
  --seed 123 \
  --out-null null_lambdamax.csv \
  --lambda-obs lambda_top.csv \
  --out-pvals pvalues.csv

# C) 仅使用前缀（序贯前缀评估，譬如只到第 6 个时间窗）：
python step4_montecarlo_h0.py \
  --prepared prepared.npz \
  --baseline-mode internal \
  --scan-level groups \
  --kmax 6 \
  --B 50000 \
  --out-null null_k6.csv
"""
from __future__ import annotations
import argparse
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import re


# -------------------- ICD-10 grouping --------------------
_ICD3_RE = re.compile(r'^([A-Z][0-9]{2})')
def icd10_level3(code: str) -> str:
    m = _ICD3_RE.match(str(code))
    return m.group(1) if m else str(code)[:3].upper()


def build_membership_GxL(leaves: List[str], groups: List[str]) -> np.ndarray:
    """
    Standardized builder for full hierarchy mapping.
    Maps each leaf to its Chapter, Category, and itself based on the provided groups list.
    """
    G, L = len(groups), len(leaves)
    M = np.zeros((G, L), dtype=int)
    group_to_idx = {g: i for i, g in enumerate(groups)}
    for li, leaf in enumerate(leaves):
        ancestors = [leaf[0], leaf.split('.')[0], leaf]
        for p in set(ancestors):
            if p in group_to_idx:
                gi = group_to_idx[p]
                M[gi, li] = 1
    return M


# -------------------- Poisson one-sided LLR --------------------
def llr_one_sided(c: np.ndarray, u: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    T = c*ln(c/u) + (C-c)*ln((C-c)/(C-u)) if c>u and 0<u<C, else 0.
    Broadcasting-safe: broadcast c, u, C to the same shape before masking.
    """
    c = np.asarray(c, dtype=float)
    u = np.asarray(u, dtype=float)
    C = np.asarray(C, dtype=float)

    # <<< fix: broadcast all to the same shape >>>
    c, u, C = np.broadcast_arrays(c, u, C)

    T = np.zeros_like(c, dtype=float)
    valid = (u > 0) & (C > u) & (c > u) & (C >= c)
    if not np.any(valid):
        return T

    cv = c[valid]; uv = u[valid]; Cv = C[valid]
    term1 = cv * np.log(cv / uv)
    with np.errstate(divide='ignore', invalid='ignore'):
        numer = np.maximum(Cv - cv, 0.0)
        denom = np.maximum(Cv - uv, 1e-15)
        ratio = np.divide(numer, denom, out=np.ones_like(numer), where=denom > 0)
        term2 = np.zeros_like(cv)
        pos = numer > 0
        term2[pos] = numer[pos] * np.log(ratio[pos])
    T[valid] = term1 + term2
    return T


# -------------------- p-value via rank --------------------
def p_value_from_rank(lambda_obs: np.ndarray, lambdas_null: np.ndarray) -> np.ndarray:
    """
    p_k = (1 + sum_b [Lambda_null(k,b) >= Lambda_obs(k)]) / (B+1)
    lambda_obs: (K,), lambdas_null: (K,B)
    """
    lambda_obs = np.asarray(lambda_obs, dtype=float).reshape(-1)
    lambdas_null = np.asarray(lambdas_null, dtype=float)
    if lambdas_null.ndim != 2:
        raise ValueError("lambdas_null must be K x B")
    K, B = lambdas_null.shape
    if lambda_obs.shape[0] != K:
        raise ValueError("lambda_obs length must match K in lambdas_null")
    ge = (lambdas_null >= lambda_obs[:, None]).sum(axis=1)
    return (1.0 + ge) / float(B + 1)


# -------------------- core: H0 multinomial resampling --------------------
def multinomial_H0(
    z_k: np.ndarray,
    q_l_k: np.ndarray,
    U_k: np.ndarray,
    membership_GxL: Optional[np.ndarray],
    B: int,
    top2: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Draw B H0 replicates per look k:
      counts_l^(k,b) ~ Multinomial(z_k, q_l_k[k,:]);
      aggregate to groups if membership is provided;
      compute LLR against expectations from q (same q drives both sampling & expectation).
    Returns:
      lambdamax[K,B], secondmax[K,B] (or None)
    """
    rng = np.random.default_rng(seed)
    z_k = np.asarray(z_k, dtype=int).reshape(-1)
    q_l_k = np.asarray(q_l_k, dtype=float)
    U_k = np.asarray(U_k, dtype=float).reshape(-1)

    K, L = q_l_k.shape
    if z_k.shape[0] != K or U_k.shape[0] != K:
        raise ValueError("Shapes mismatch among z_k / U_k / q_l_k.")

    scan_groups = membership_GxL is not None
    if scan_groups:
        M = np.asarray(membership_GxL, dtype=int)
        G = M.shape[0]
    else:
        G = L

    lambdamax = np.zeros((K, B), dtype=float)
    secondmax = np.zeros((K, B), dtype=float) if top2 else None

    for k in range(K):
        z = int(z_k[k]); C = float(U_k[k])
        q = q_l_k[k, :]
        # safety: ensure non-negative & normalized
        q = np.clip(q, 0.0, None)
        s = q.sum()
        if s <= 0:
            # fallback to uniform if baseline is degenerate
            q = np.full(L, 1.0 / L, dtype=float)
        else:
            q = q / s

        # draw all B at once: (B x L)
        counts = rng.multinomial(z, q, size=B)

        if scan_groups:
            cBG = counts @ M.T   # (B x G)
            qG = q @ M.T         # (G,)
            u = qG[None, :] * float(z)
        else:
            cBG = counts         # (B x L)
            u = q[None, :] * float(z)

        Cmat = np.full_like(cBG, C, dtype=float)
        T = llr_one_sided(cBG, u, Cmat)  # (B x M)
        # top-1 and optional top-2 per row
        # use partial sort? here full sort for clarity
        T_sorted = np.sort(T, axis=1)    # ascending
        lambdamax[k, :] = T_sorted[:, -1]
        if top2:
            secondmax[k, :] = T_sorted[:, -2] if T.shape[1] >= 2 else 0.0

    return lambdamax, secondmax


# -------------------- main (integrated CLI) --------------------
def main():
    ap = argparse.ArgumentParser(description="Step 4 Monte Carlo under H0 (integrated).")
    ap.add_argument("--prepared", type=str, required=True, help="Path to Step 2 prepared.npz")
    ap.add_argument("--baseline-mode", type=str, choices=["ext_w", "ext_qk", "internal"],
                    default="ext_qk", help="Choose q_l^(k): from w (repeated), external per-k q, or internal cumulative.")
    ap.add_argument("--scan-level", type=str, choices=["groups", "leaves"], default="groups",
                    help="Scan on level-3 groups (default) or leaves.")
    ap.add_argument("--B", type=int, default=100000, help="Number of H0 replicates per look.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    ap.add_argument("--top2", action="store_true", help="Also record 2nd largest per replicate.")
    ap.add_argument("--kmax", type=int, default=None, help="Use only first kmax looks (sequential prefix).")

    ap.add_argument("--out-null", type=str, required=True, help="CSV of null ~Lambda_max^(k,b) (and optional second).")
    ap.add_argument("--out-null-npz", type=str, default=None, help="Optional NPZ of lambdamax[K,B] (and second[K,B]).")

    ap.add_argument("--lambda-obs", type=str, default=None, help="Optional: Step 3 lambda_top.csv (rank=1 rows).")
    ap.add_argument("--out-pvals", type=str, default=None, help="Optional: CSV of p-values by look when --lambda-obs is set.")
    args = ap.parse_args()

    # Load prepared arrays
    data = np.load(args.prepared, allow_pickle=True)
    leaves = list(data["leaves"])
    z_k = np.asarray(data["z_k"], dtype=int)
    U_k = np.asarray(data["U_k"], dtype=float)

    # Optional prefix (sequential)
    if args.kmax is not None:
        kmax = int(args.kmax)
        z_k = z_k[:kmax]
        U_k = U_k[:kmax]

    # Build q_l_k by baseline-mode
    if args.baseline_mode == "ext_w":
        w_ext = np.asarray(data["w_external"])
        if w_ext.size == 0:
            raise RuntimeError("w_external missing in prepared.npz; provide baseline in Step 2.")
        q_l_k = np.repeat(w_ext[None, :], z_k.shape[0], axis=0)
    elif args.baseline_mode == "ext_qk":
        q_ext = np.asarray(data["q_external"])
        if q_ext.size == 0:
            raise RuntimeError("q_external missing in prepared.npz; provide baseline in Step 2.")
        q_l_k = q_ext
    else:
        q_int = np.asarray(data["q_internal"])
        q_l_k = q_int

    # Keep same prefix for q if kmax set
    if args.kmax is not None:
        q_l_k = q_l_k[:z_k.shape[0], :]

    # Membership matrix if scanning groups
    membership = None
    if args.scan_level == "groups":
        groups = list(data["groups"])
        membership = build_membership_GxL(leaves, groups)

    # Run H0 multinomial resampling
    lambdamax, second = multinomial_H0(
        z_k=z_k,
        q_l_k=q_l_k,
        U_k=U_k,
        membership_GxL=membership,
        B=args.B,
        top2=args.top2,
        seed=args.seed
    )

    # Save null results to CSV (long)
    K, B = lambdamax.shape
    rows = []
    if args.top2 and (second is not None):
        for k in range(K):
            for b in range(B):
                rows.append((k+1, b+1, float(lambdamax[k, b]), float(second[k, b])))
        cols = ["k", "b", "lambda_max", "second_max"]
    else:
        for k in range(K):
            for b in range(B):
                rows.append((k+1, b+1, float(lambdamax[k, b])))
        cols = ["k", "b", "lambda_max"]
    df_null = pd.DataFrame(rows, columns=cols)
    df_null.to_csv(args.out_null, index=False)
    print(f"[OK] Saved null resamples: {args.out_null} (rows={len(df_null)})  K={K}, B={B}")

    # Optional NPZ
    if args.out_null_npz:
        if args.top2 and (second is not None):
            np.savez_compressed(args.out_null_npz, lambdamax=lambdamax, secondmax=second)
        else:
            np.savez_compressed(args.out_null_npz, lambdamax=lambdamax)
        print(f"[OK] Saved matrix NPZ: {args.out_null_npz}")

    # Optional: compute p-values if given observed lambda_top.csv
    if args.lambda_obs is not None:
        if not args.out_pvals:
            raise RuntimeError("Provide --out-pvals when --lambda-obs is set.")
        df_obs = pd.read_csv(args.lambda_obs)
        obs = df_obs[df_obs["rank"] == 1].sort_values("k")
        # 如果设置了 kmax，只保留 <= kmax
        if args.kmax is not None:
            obs = obs[obs["k"] <= args.kmax]
        lambda_obs = obs["T"].to_numpy(dtype=float)
        k_obs = obs["k"].to_numpy(dtype=int)

        if lambda_obs.shape[0] != K:
            raise RuntimeError(f"Observed K={lambda_obs.shape[0]} does not match null K={K}. "
                               f"Use --kmax to align or check inputs.")

        p = p_value_from_rank(lambda_obs, lambdamax)
        out = pd.DataFrame({"k": k_obs, "lambda_obs": lambda_obs, "p": p})
        out.to_csv(args.out_pvals, index=False)
        print(f"[OK] Saved p-values: {args.out_pvals}")

if __name__ == "__main__":
    main()