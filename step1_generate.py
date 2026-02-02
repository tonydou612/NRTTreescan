#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic data generator (Plan B): baseline total >> events total.

Outputs:
  1) events CSV   : long format (icd10, k, count)         — vaccinated cohort events
  2) baseline CSV : long format (icd10, k, count)         — all-population baseline
  3) weights CSV  : optional    (icd10, w)                — time-invariant weights from baseline

Key features:
  - Control exact totals via --events-total and --baseline-total (e.g., 1,000 vs 100,000).
  - Otherwise use per-look means (--z0 / --baseline-z0) with seasonality.
  - Baseline leaf composition can be same as events or independent.
  - RR injection for events in specified looks (H1).
"""
import argparse
import math
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------
def make_icd10_codes(L: int) -> List[str]:
    """Create L synthetic ICD-10 leaf-level codes like A00.0, A00.1, ..."""
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    codes = []
    major = 0
    minor = 0
    li = 0
    while len(codes) < L:
        letter = letters[li % len(letters)]
        codes.append(f"{letter}{major:02d}.{minor}")
        minor += 1
        if minor > 9:
            minor = 0
            major += 1
        if major > 99:
            major = 0
            li += 1
    return codes[:L]


def seasonal_weights(K: int, seasonal: float) -> np.ndarray:
    """
    Positive weights across looks capturing seasonality.
    w_k ∝ 1 + seasonal * sin(2π k / K), clipped to >= 1e-6, then normalized.
    """
    ks = np.arange(1, K + 1)
    base = 1.0 + seasonal * np.sin(2 * np.pi * ks / K)
    base = np.clip(base, 1e-6, None)
    w = base / base.sum()
    return w


def split_total_across_K(total: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Allocate an integer total across K looks exactly via Multinomial(total, weights).
    """
    return rng.multinomial(total, weights)


def seasonal_counts(K: int, z0: int, seasonal: float, rng: np.random.Generator, jitter_rate: float = 0.03) -> np.ndarray:
    """
    Per-look totals when not fixing grand total:
      z_k ≈ round(z0 * (1 + seasonal * sin(...))) + small Poisson jitter
    """
    ks = np.arange(1, K + 1)
    base = z0 * (1.0 + seasonal * np.sin(2 * np.pi * ks / K))
    base = np.clip(base, 1.0, None)
    jitter = rng.poisson(lam=max(1e-9, jitter_rate * z0), size=K)
    z = np.maximum(1, np.round(base).astype(int) + jitter)
    return z


def dirichlet_q(L: int, concentration: float, rng: np.random.Generator) -> np.ndarray:
    """Baseline leaf proportions q_l ~ Dirichlet(concentration)."""
    alpha = np.full(L, concentration, dtype=float)
    q = rng.dirichlet(alpha)
    eps = 1e-12
    q = np.maximum(q, eps)
    q /= q.sum()
    return q


def apply_rr_to_q(q: np.ndarray, omega_idx: np.ndarray, rr: float) -> np.ndarray:
    """Apply RR to Ω and renormalize (Equation A6)."""
    p = q.copy()
    if len(omega_idx) > 0:
        p[omega_idx] *= rr
    s = p.sum()
    if s <= 0:
        raise ValueError("Sum of adjusted probabilities is non-positive.")
    p /= s
    return p


def to_long(icd10: List[str], counts_by_k: List[np.ndarray]) -> pd.DataFrame:
    """Convert per-look count vectors into long DataFrame: icd10, k, count."""
    records = []
    for k, vec in enumerate(counts_by_k, start=1):
        for code, c in zip(icd10, vec.tolist()):
            records.append((code, k, int(c)))
    return pd.DataFrame(records, columns=["icd10", "k", "count"])


def compute_time_invariant_weights(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate baseline across k to get time-invariant weights w_l.
    Returns DataFrame with columns ("icd10", "w"), sum(w)=1.
    """
    agg = (baseline_df
           .groupby("icd10", as_index=False)["count"].sum()
           .rename(columns={"count": "w_raw"}))
    total = float(agg["w_raw"].sum())
    if total <= 0:
        raise ValueError("Baseline total (summed over all k and icd10) is non-positive.")
    agg["w"] = agg["w_raw"] / total
    return agg[["icd10", "w"]]


# ----------------------------- generator core -----------------------------
def generate(
    L: int,
    K: int,
    # events totals
    events_total: Optional[int],
    z0: Optional[int],
    seasonal: float,
    # baseline totals
    baseline_total: Optional[int],
    baseline_scale: Optional[float],
    baseline_z0: Optional[int],
    baseline_seasonal: Optional[float],
    # leaf compositions
    q_conc: float,
    baseline_q_mode: str,          # "same" or "independent"
    baseline_q_conc: float,
    # signal injection for events (H1)
    mode: str,
    rr: float,
    inject_frac: float,
    inject_start: int,
    inject_end: int,
    # reproducibility
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray, np.ndarray]:
    """
    Returns: (events_df, baseline_df, icd10_codes, z_events, omega_idx)
    """
    rng = np.random.default_rng(seed)
    icd10_codes = make_icd10_codes(L)

    # ---- per-look totals for events ----
    if events_total is not None:
        w = seasonal_weights(K, seasonal)
        z_events = split_total_across_K(events_total, w, rng)
        z_events = np.maximum(1, z_events)  # avoid zero
        # if rounding made sum > events_total due to clipping, fix by borrowing from largest bins
        diff = int(z_events.sum() - events_total)
        if diff > 0:
            # reduce diff from largest entries
            idx = np.argsort(-z_events)
            i = 0
            while diff > 0 and i < len(idx):
                j = idx[i]
                if z_events[j] > 1:
                    take = min(diff, z_events[j] - 1)
                    z_events[j] -= take
                    diff -= take
                i += 1
    else:
        if z0 is None:
            raise ValueError("Provide either --events-total or --z0 for events.")
        z_events = seasonal_counts(K, z0, seasonal, rng)

    # ---- per-look totals for baseline (precedence: baseline_total > baseline_scale > baseline_z0 > fallback=events) ----
    b_seasonal = seasonal if baseline_seasonal is None else baseline_seasonal
    if baseline_total is not None:
        w_b = seasonal_weights(K, b_seasonal)
        z_base = split_total_across_K(baseline_total, w_b, rng)
        z_base = np.maximum(1, z_base)
        diff = int(z_base.sum() - baseline_total)
        if diff > 0:
            idx = np.argsort(-z_base)
            i = 0
            while diff > 0 and i < len(idx):
                j = idx[i]
                if z_base[j] > 1:
                    take = min(diff, z_base[j] - 1)
                    z_base[j] -= take
                    diff -= take
                i += 1
    elif baseline_scale is not None:
        z_base = np.maximum(1, np.round(baseline_scale * z_events).astype(int))
    elif baseline_z0 is not None:
        z_base = seasonal_counts(K, baseline_z0, b_seasonal, rng)
    else:
        z_base = z_events.copy()  # default: same scale

    # ---- leaf compositions ----
    q_events = dirichlet_q(L, q_conc, rng)
    if baseline_q_mode.lower() == "same":
        q_base = q_events
    else:
        q_base = dirichlet_q(L, baseline_q_conc, rng)

    # ---- injection set Ω for events (H1) ----
    if mode.lower() == "h1":
        n_inject = max(1, int(math.ceil(inject_frac * L)))
        omega_idx = rng.choice(L, size=n_inject, replace=False)
    else:
        omega_idx = np.array([], dtype=int)

    # ---- generate per-look leaf counts ----
    events_by_k = []
    baseline_by_k = []
    for k in range(1, K + 1):
        z_e = int(z_events[k - 1])
        z_b = int(z_base[k - 1])

        # baseline (all-pop): Multinomial(z_b, q_base)
        u_l = rng.multinomial(z_b, q_base)
        baseline_by_k.append(u_l)

        # events (vaccinated): Multinomial with optional RR
        if mode.lower() == "h1" and (inject_start <= k <= inject_end) and len(omega_idx) > 0:
            p = apply_rr_to_q(q_events, omega_idx, rr)
        else:
            p = q_events
        c_l = rng.multinomial(z_e, p)
        events_by_k.append(c_l)

    events_df = to_long(icd10_codes, events_by_k)
    baseline_df = to_long(icd10_codes, baseline_by_k)
    return events_df, baseline_df, icd10_codes, z_events, omega_idx


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="TreeScan synthetic data generator (Plan B).")

    # Scale/time
    ap.add_argument("--L", type=int, default=80, help="Number of ICD-10 leaf codes (level-4).")
    ap.add_argument("--K", type=int, default=24, help="Number of monitoring looks/time points.")
    ap.add_argument("--seasonal", type=float, default=0.25, help="Seasonality amplitude in [0,1).")

    # Events totals: either --events-total or --z0
    ap.add_argument("--events-total", type=int, default=None, help="Total events across all looks (vaccinated cohort).")
    ap.add_argument("--z0", type=int, default=250, help="Mean per-look events if --events-total not set.")

    # Baseline totals: precedence baseline-total > baseline-scale > baseline-z0 > fallback=events
    ap.add_argument("--baseline-total", type=int, default=None, help="Total baseline counts across all looks (all-pop).")
    ap.add_argument("--baseline-scale", type=float, default=None, help="Baseline per-look = round(scale * events per-look).")
    ap.add_argument("--baseline-z0", type=int, default=None, help="Mean per-look baseline if totals/scale not set.")
    ap.add_argument("--baseline-seasonal", type=float, default=None, help="Seasonality for baseline (default: same as events).")

    # Leaf composition (q)
    ap.add_argument("--q-conc", type=float, default=1.0, help="Dirichlet concentration for events baseline q.")
    ap.add_argument("--baseline-q-mode", type=str, choices=["same", "independent"], default="same",
                    help="Use same q as events or an independent one for baseline.")
    ap.add_argument("--baseline-q-conc", type=float, default=1.0, help="Dirichlet concentration if baseline-q-mode=independent.")

    # Signal (H1) for events
    ap.add_argument("--mode", type=str, choices=["h0", "h1"], default="h0", help="H0 or H1 for events.")
    ap.add_argument("--rr", type=float, default=2.0, help="RR for injected leaves (events) under H1.")
    ap.add_argument("--inject-frac", type=float, default=0.05, help="Fraction of leaves to inject under H1.")
    ap.add_argument("--inject-start", type=int, default=6, help="First look index (1-based) to inject under H1.")
    ap.add_argument("--inject-end", type=int, default=10, help="Last look index (1-based) to inject under H1 (inclusive).")

    # Repro & outputs
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--events-out", type=str, default="events.csv", help="Output CSV for events (icd10,k,count).")
    ap.add_argument("--baseline-out", type=str, default="baseline.csv", help="Output CSV for baseline (icd10,k,count).")
    ap.add_argument("--weights-out", type=str, default=None, help="OPTIONAL: Output CSV for time-invariant weights (icd10,w).")

    args = ap.parse_args()

    events_df, baseline_df, codes, z_events, omega_idx = generate(
        L=args.L, K=args.K,
        events_total=args.events_total, z0=args.z0, seasonal=args.seasonal,
        baseline_total=args.baseline_total, baseline_scale=args.baseline_scale,
        baseline_z0=args.baseline_z0, baseline_seasonal=args.baseline_seasonal,
        q_conc=args.q_conc, baseline_q_mode=args.baseline_q_mode, baseline_q_conc=args.baseline_q_conc,
        mode=args.mode, rr=args.rr, inject_frac=args.inject_frac,
        inject_start=args.inject_start, inject_end=args.inject_end,
        seed=args.seed
    )

    # Save
    events_df.to_csv(args.events_out, index=False)
    baseline_df.to_csv(args.baseline_out, index=False)

    # Optional weights from baseline
    if args.weights_out:
        weights_df = compute_time_invariant_weights(baseline_df)
        weights_df.to_csv(args.weights_out, index=False)

    # Summary
    print(f"Saved events   -> {args.events_out}  (rows={len(events_df)})  total={int(events_df['count'].sum())}")
    print(f"Saved baseline -> {args.baseline_out} (rows={len(baseline_df)}) total={int(baseline_df['count'].sum())}")
    if args.weights_out:
        print(f"Saved weights  -> {args.weights_out} (rows={weights_df.shape[0]})")
    print(f"L={args.L}, K={args.K}, seasonal={args.seasonal}, mode={args.mode}")
    if args.mode.lower() == 'h1':
        print(f"RR={args.rr}, inject_frac={args.inject_frac}, window=[{args.inject_start},{args.inject_end}]  | injected leaves={len(omega_idx)}")


if __name__ == "__main__":
    main()