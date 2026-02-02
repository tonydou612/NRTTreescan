#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 7: Visualization for sequential monitoring results (threshold method)

Inputs (at minimum --seq):
  --seq         : CSV from step6_sequential_decision.py (expects k, T1, c_k, cross, first_alarm, unit1)
  --boundaries  : (optional) boundaries.csv from step5 to fill U_k / t_k if missing
  --outdir      : Output directory for charts/report (default ./viz)
  --fmt         : Image format (png/pdf/svg; default png)
  --dpi         : Image resolution (default 150)
  --annot       : Annotate unit1 labels (default True)
  --topk-labels : Only annotate top-K T1 points (default 8; 0 = annotate all)

Outputs:
  viz/fig1_T_vs_ck.<fmt>         — T1 vs c_k across windows, with first crossing marked
  viz/fig2_info_time.<fmt>       — Information time trajectory (if U_k/t_k available)
  viz/fig3_cross_timeline.<fmt>  — Crossing (Yes/No) per window
  viz/report_step7.html          — Simple HTML report embedding the figures + first crossing summary
"""

from __future__ import annotations
import argparse, os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- utils -----------------
def pick_col(df: pd.DataFrame, candidates: List[str], required: bool=True, default=None) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Column not found. Candidates={candidates}, columns={list(df.columns)}")
    return default

def ensure_num(s: pd.Series, kind=float):
    return pd.to_numeric(s, errors="coerce").astype(kind)

def load_seq(seq_path: str) -> pd.DataFrame:
    df = pd.read_csv(seq_path)
    # Normalize key column names
    k   = pick_col(df, ["k", "K", "look"])
    t1  = pick_col(df, ["T1", "T", "Lambda", "lambda", "llr", "stat"])
    ck  = pick_col(df, ["c_k", "threshold", "ck", "c", "boundary"])
    u1  = pick_col(df, ["unit1", "unit", "code", "cluster", "unit_id"])
    cross = pick_col(df, ["cross"])
    fa    = pick_col(df, ["first_alarm"], required=False, default=None)

    out = pd.DataFrame({
        "k": ensure_num(df[k], int),
        "T1": ensure_num(df[t1], float),
        "c_k": ensure_num(df[ck], float),
        "unit1": df[u1].astype(str),
        "cross": df[cross].astype(bool),
    })
    if fa is not None:
        out["first_alarm"] = df[fa].astype(bool)
    else:
        out["first_alarm"] = False
        idx = out.index[out["cross"]].min() if out["cross"].any() else None
        if idx is not None:
            out.loc[idx, "first_alarm"] = True

    # Optional fields from seq (if present)
    for name, cands, kind in [
        ("U_k", ["U_k","Uk","cum_events"], float),
        ("t_k", ["t_k","tk","info_time"], float),
        ("alpha_k", ["alpha_k","alpha"], float),
        ("U_lim", ["U_lim","Ulimit","Ulim"], float),
    ]:
        try:
            col = pick_col(df, cands, required=False)
            out[name] = ensure_num(df[col], kind) if col is not None else np.nan
        except Exception:
            out[name] = np.nan

    return out.sort_values("k").reset_index(drop=True)

def maybe_merge_bounds(df: pd.DataFrame, bounds_path: str) -> pd.DataFrame:
    if not bounds_path or not os.path.exists(bounds_path):
        return df
    B = pd.read_csv(bounds_path)
    kB = pick_col(B, ["k","K","look"])
    take = {"k": kB}
    for want, cands in {
        "U_k": ["U_k","Uk","cum_events"],
        "t_k": ["t_k","tk","info_time"],
        "alpha_k": ["alpha_k","alpha"],
        "U_lim": ["U_lim","Ulimit","Ulim"],
    }.items():
        try:
            take[want] = pick_col(B, cands, required=False)
        except KeyError:
            pass
    B2 = B.rename(columns=take)[list(take.keys())]
    M = pd.merge(df, B2, on="k", how="left", suffixes=("","_bnd"))
    # Prefer seq values; fill missing from boundaries
    for col in ["U_k","t_k","alpha_k","U_lim"]:
        b = f"{col}_bnd"
        if b in M.columns:
            M[col] = np.where(M[col].notna(), M[col], M[b])
    drop_cols = [c for c in M.columns if c.endswith("_bnd")]
    return M.drop(columns=drop_cols)

# ----------------- plotting -----------------
def fig1_T_vs_ck(df: pd.DataFrame, outpath: str, fmt="png", dpi=150, annotate=True, topk_labels=8):
    k = df["k"].to_numpy(int)
    T1 = df["T1"].to_numpy(float)
    ck = df["c_k"].to_numpy(float)
    cross = df["cross"].to_numpy(bool)
    first_alarm = df["first_alarm"].to_numpy(bool)
    unit1 = df["unit1"].astype(str).to_numpy()

    plt.figure(figsize=(10, 5.5))
    # Threshold curve
    plt.plot(k, ck, marker="o", linewidth=1.5, label="Threshold c_k")
    # Observed T1
    plt.plot(k, T1, marker="o", linewidth=1.5, label="T1 (max LLR)")

    # Mark crossings
    if np.any(cross):
        kc = k[cross]; Tc = T1[cross]
        plt.scatter(kc, Tc, s=60, marker="s", label="Crossing", zorder=3)

    # First crossing vertical line and annotation
    if np.any(first_alarm):
        k_star = int(k[first_alarm][0])
        y0, y1 = np.nanmin(np.r_[T1, ck]), np.nanmax(np.r_[T1, ck])
        plt.axvline(k_star, linestyle="--", linewidth=1.2)
        plt.text(k_star, y1, f"  First crossing at k={k_star}", va="top", ha="left")

    # Optional annotations for unit1 (top-K by T1 to avoid clutter)
    if annotate:
        if topk_labels > 0:
            order = np.argsort(-T1)[:min(topk_labels, len(T1))]
            to_annot = set(order.tolist())
        else:
            to_annot = set(range(len(T1)))
        for i in range(len(k)):
            if i in to_annot:
                plt.annotate(unit1[i], (k[i], T1[i]), xytext=(4, 4), textcoords="offset points")

    plt.xlabel("Window k")
    plt.ylabel("Statistic")
    plt.title("T1 vs c_k over windows")
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, format=fmt, dpi=dpi)
    plt.close()

def fig2_info_time(df: pd.DataFrame, outpath: str, fmt="png", dpi=150):
    # Requires U_k or t_k
    if df["U_k"].isna().all() and df["t_k"].isna().all():
        return False
    k = df["k"].to_numpy(int)
    if df["t_k"].isna().all():
        if df["U_k"].isna().all() or df["U_lim"].isna().all():
            return False
        t = df["U_k"].to_numpy(float) / df["U_lim"].to_numpy(float)
    else:
        t = df["t_k"].to_numpy(float)

    plt.figure(figsize=(9.5, 4.2))
    plt.plot(k, t, marker="o")
    plt.xlabel("Window k")
    plt.ylabel("Information time t_k")
    plt.title("Information time trajectory")
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, format=fmt, dpi=dpi)
    plt.close()
    return True

def fig3_cross_timeline(df: pd.DataFrame, outpath: str, fmt="png", dpi=150):
    k = df["k"].to_numpy(int)
    cross = df["cross"].to_numpy(bool).astype(int)
    plt.figure(figsize=(10, 2.8))
    plt.bar(k, cross, width=0.8)
    plt.yticks([0,1], ["No", "Yes"])
    plt.xlabel("Window k")
    plt.title("Crossing timeline (threshold method)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, format=fmt, dpi=dpi)
    plt.close()

# ----------------- html report -----------------
def write_html_report(outdir: str, have_info: bool, summary_row: pd.Series|None):
    html_path = os.path.join(outdir, "report_step7.html")
    lines = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Step 7 Visualization Report</title>",
        "<h1>Step 7 Visualization Report</h1>",
        "<h2>Figure 1: T1 and threshold c_k</h2>",
        "<img src='fig1_T_vs_ck.png' style='max-width: 100%;'>",
    ]
    if have_info:
        lines += [
            "<h2>Figure 2: Information time trajectory</h2>",
            "<img src='fig2_info_time.png' style='max-width: 100%;'>",
        ]
    lines += [
        "<h2>Figure 3: Crossing timeline</h2>",
        "<img src='fig3_cross_timeline.png' style='max-width: 100%;'>",
    ]
    if summary_row is not None:
        lines += [
            "<h2>First crossing summary</h2>",
            "<table border='1' cellspacing='0' cellpadding='6'>",
            f"<tr><td>k</td><td>{int(summary_row['k'])}</td></tr>",
            f"<tr><td>T1</td><td>{float(summary_row['T1']):.4f}</td></tr>",
            f"<tr><td>c_k</td><td>{float(summary_row['c_k']):.4f}</td></tr>",
            f"<tr><td>unit1</td><td>{summary_row['unit1']}</td></tr>",
            "</table>"
        ]
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Step 7: Visualization (threshold method output)")
    ap.add_argument("--seq", required=True, help="Output CSV from step6 (seq_decision.csv or seq_alerts.csv)")
    ap.add_argument("--boundaries", default=None, help="(Optional) boundaries.csv to fill U_k / t_k if missing")
    ap.add_argument("--outdir", default="./viz")
    ap.add_argument("--fmt", choices=["png","pdf","svg"], default="png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--annot", action="store_true", default=True, help="Annotate unit1 labels (default True)")
    ap.add_argument("--no-annot", dest="annot", action="store_false", help="Disable annotations")
    ap.add_argument("--topk-labels", type=int, default=8, help="Annotate only top-K T1 points (default 8)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_seq(args.seq)
    df = maybe_merge_bounds(df, args.boundaries)

    # Figure 1: T1 vs c_k
    fig1 = os.path.join(args.outdir, f"fig1_T_vs_ck.{args.fmt}")
    fig1_T_vs_ck(df, fig1, fmt=args.fmt, dpi=args.dpi, annotate=args.annot, topk_labels=args.topk_labels)

    # Figure 2: Information time (if available)
    fig2 = os.path.join(args.outdir, f"fig2_info_time.{args.fmt}")
    have_info = fig2_info_time(df, fig2, fmt=args.fmt, dpi=args.dpi)

    # Figure 3: Crossing timeline
    fig3 = os.path.join(args.outdir, f"fig3_cross_timeline.{args.fmt}")
    fig3_cross_timeline(df, fig3, fmt=args.fmt, dpi=args.dpi)

    # First crossing summary
    summary = None
    if df["first_alarm"].any():
        summary = df.loc[df["first_alarm"]].iloc[0]

    # HTML report
    write_html_report(args.outdir, have_info=have_info, summary_row=summary)

    print(f"[OK] Charts & report generated at: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()