#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6 (unified): 序贯判定（两种路线二选一）
- method=threshold : 用 boundaries.csv 的 c_k 与 step3 的 T1 比较，首次越界报警
- method=pvalue    : 每窗 H0 重抽样，计算 Top-1/Top-2 的全树校正 p 值，Bonferroni/2 序贯报警

输入：
  --lambda-top     : step3_statistics.py 的 lambda_top.csv（每窗 Top1/Top2）
  --boundaries     : step5 的 boundaries.csv（含 c_k；pvalue 模式还需 alpha_k）
  --method         : threshold 或 pvalue
  --out            : 输出统一结果（默认 seq_decision.csv）

仅 pvalue 模式需要：
  --prepared       : step2 的 prepared.npz（含 leaves、可选 groups、z_k、q_*）
  --baseline-mode  : {ext_qk, ext_w, internal}
  --scan-level     : {groups, leaves}
  --B              : H0 重抽样次数（默认 20000）
  --seed           : 随机种子（默认 123）

输出：seq_decision.csv
  统一列：
    k, T1, unit1, T2, unit2, decision_basis, threshold, score, cross, first_alarm
  若 method=pvalue，还会附加：alpha_k, alpha_k_over2, p1, p2, min_p
  若 method=threshold，还会附加：c_k（与 threshold 同值便于核对）
"""
from __future__ import annotations
import argparse, os, sys, re
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# ---------- 小工具 ----------
def pick_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"找不到字段，候选 {candidates}，现有列 {list(df.columns)}")

def ensure_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="raise").astype(int)

def ensure_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="raise").astype(float)

# ---------- ICD-10 分组 ----------
_ICD3_RE = re.compile(r'^([A-Z][0-9]{2})')
def icd10_level3(code: str) -> str:
    m = _ICD3_RE.match(str(code))
    return m.group(1) if m else str(code)[:3].upper()

def build_groups_from_leaves(leaves: List[str]) -> List[str]:
    return sorted({icd10_level3(c) for c in leaves})

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

# ---------- LLR ----------
def llr_one_sided(c: np.ndarray, u: np.ndarray, C: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float); u = np.asarray(u, float); C = np.asarray(C, float)
    c, u, C = np.broadcast_arrays(c, u, C)
    T = np.zeros_like(c)
    valid = (u > 0) & (C > u) & (c > u) & (C >= c)
    if not np.any(valid): return T
    cv, uv, Cv = c[valid], u[valid], C[valid]
    term1 = cv * np.log(cv/uv)
    with np.errstate(divide='ignore', invalid='ignore'):
        numer = np.maximum(Cv - cv, 0.0)
        denom = np.maximum(Cv - uv, 1e-15)
        ratio = np.divide(numer, denom, out=np.ones_like(numer), where=denom>0)
        term2 = np.zeros_like(cv)
        pos = numer > 0
        term2[pos] = numer[pos] * np.log(ratio[pos])
    T[valid] = term1 + term2
    return T

# ---------- baseline q_l^(k) ----------
def get_q_l_k(data, mode: str, K: int) -> np.ndarray:
    if mode == "ext_qk":
        if "q_external" not in data: raise RuntimeError("prepared.npz 缺 q_external")
        q = np.asarray(data["q_external"])
        if q.shape[0] != K:
            last = q[-1:]
            q = np.concatenate([q, np.repeat(last, K-q.shape[0], axis=0)], 0) if K>q.shape[0] else q[:K]
        return q
    elif mode == "ext_w":
        if "w_external" not in data: raise RuntimeError("prepared.npz 缺 w_external")
        w = np.asarray(data["w_external"])
        if w.ndim != 1: raise RuntimeError("w_external 需为 1D")
        return np.repeat(w[None, :], K, axis=0)
    else:
        if "q_internal" not in data: raise RuntimeError("prepared.npz 缺 q_internal")
        q = np.asarray(data["q_internal"])
        if q.shape[0] != K:
            last = q[-1:]
            q = np.concatenate([q, np.repeat(last, K-q.shape[0], axis=0)], 0) if K>q.shape[0] else q[:K]
        return q

# ---------- H0：每窗 Top1/Top2 空分布 ----------
def null_top12_for_k(z: int, q: np.ndarray, C: float,
                     membership: Optional[np.ndarray], B: int,
                     rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回： (top1_null[B], top2_null[B]) for window k
    """
    L = q.size
    if z <= 0 or q.sum() <= 0:
        return np.zeros(B), np.zeros(B)
    p = np.clip(q, 0, None); p = p / p.sum()
    counts = rng.multinomial(z, p, size=B)  # (B x L)
    if membership is not None:
        M = membership                           # (G x L)
        cBG = counts @ M.T                       # (B x G)
        qG = p @ M.T                              # (G,)
        u = qG[None, :] * float(z)               # (B x G)
        T = llr_one_sided(cBG, u, np.full_like(cBG, C, dtype=float))  # (B x G)
    else:
        u = p[None, :] * float(z)                # (B x L)
        T = llr_one_sided(counts, u, np.full_like(counts, C, dtype=float))
    # 取 Top1 / Top2
    idx2 = np.argpartition(-T, kth=1, axis=1)[:, :2]  # (B x 2)
    vals2 = np.take_along_axis(T, idx2, axis=1)
    top1 = np.max(vals2, axis=1)
    which = (vals2[:, 0] >= vals2[:, 1]).astype(int)
    top2 = vals2[np.arange(B), 1 - which]
    return top1, top2

def p_from_rank(obs: float, null_vals: np.ndarray) -> float:
    B = null_vals.size
    return (1.0 + float(np.sum(null_vals >= obs))) / (B + 1.0)

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser(description="Step 6 (unified): threshold or p-value sequential alarm.")
    ap.add_argument("--lambda-top", required=True, help="step3 的 lambda_top.csv")
    ap.add_argument("--boundaries", required=True, help="step5 的 boundaries.csv")
    ap.add_argument("--method", choices=["threshold", "pvalue"], required=True)
    ap.add_argument("--out", default="seq_decision.csv")

    # pvalue 模式附加参数
    ap.add_argument("--prepared", type=str, default=None)
    ap.add_argument("--baseline-mode", choices=["ext_qk", "ext_w", "internal"], default="ext_qk")
    ap.add_argument("--scan-level", choices=["groups", "leaves"], default="groups")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # 读取 boundaries（c_k / alpha_k）
    dfB = pd.read_csv(args.boundaries)
    kB = pick_col(dfB, ["k", "K", "look"])
    dfB = dfB.rename(columns={kB: "k"}).sort_values("k")
    if "c_k" not in dfB.columns:
        # 尝试其他命名
        ck = dfB.columns.intersection(["ck", "c", "boundary"])
        if not len(ck):
            raise RuntimeError("boundaries.csv 缺少 c_k（或同义列 ck/c/boundary）")
        dfB = dfB.rename(columns={ck[0]: "c_k"})

    has_alpha = "alpha_k" in dfB.columns
    if not has_alpha and args.method == "pvalue":
        raise RuntimeError("pvalue 模式需要 boundaries.csv 含 alpha_k 列（Step5 输出）。")

    # 读取 lambda_top（Top1/Top2）
    dfL = pd.read_csv(args.lambda_top)
    kL = pick_col(dfL, ["k", "K", "look"])
    rank_col = pick_col(dfL, ["rank", "order", "j"])
    T_col = pick_col(dfL, ["T", "Lambda", "lambda", "stat", "llr"])
    unit_col = pick_col(dfL, ["unit", "unit_id", "code", "cluster"])

    dfL[kL] = ensure_int(dfL[kL])
    dfL[rank_col] = ensure_int(dfL[rank_col])
    dfL[T_col] = ensure_float(dfL[T_col])

    top1 = dfL[dfL[rank_col] == 1].rename(columns={kL: "k", T_col: "T1", unit_col: "unit1"})[["k", "T1", "unit1"]]
    top2 = dfL[dfL[rank_col] == 2].rename(columns={kL: "k", T_col: "T2", unit_col: "unit2"})[["k", "T2", "unit2"]]

    # 合并
    df = pd.merge(dfB, top1, on="k", how="left")
    df = pd.merge(df, top2, on="k", how="left")
    df = df.sort_values("k").reset_index(drop=True)

    # 准备输出目录
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---------- 路线 1：阈值法 ----------
    if args.method == "threshold":
        # 判定
        df["cross"] = (df["T1"] >= df["c_k"])
        # 首次越界
        first_idx = df.index[df["cross"]].min() if df["cross"].any() else None
        df["first_alarm"] = False
        if first_idx is not None:
            df.loc[first_idx, "first_alarm"] = True

        # 统一输出字段
        out = pd.DataFrame({
            "k": df["k"].astype(int),
            "T1": df["T1"],
            "unit1": df["unit1"],
            "T2": df.get("T2", np.nan),
            "unit2": df.get("unit2", np.nan),
            "decision_basis": "threshold",
            "threshold": df["c_k"],     # 用于核对
            "score": df["T1"],          # 和 threshold 比较
            "cross": df["cross"],
            "first_alarm": df["first_alarm"],
            "c_k": df["c_k"],           # 冗余保留
        })
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 阈值法完成 -> {out_path}")
        if first_idx is not None:
            r = out.iloc[first_idx]
            print(f"[ALARM] 首次越界 k={int(r['k'])} : T1={r['T1']:.4f} ≥ c_k={r['c_k']:.4f} ; 单元={r['unit1']}")
        else:
            print("[INFO] 未出现首次越界。")
        return

    # ---------- 路线 2：p 值法（Top-2 + Bonferroni/2） ----------
    # 需要 prepared.npz、q_l^(k)、z_k、可能的 groups
    if not args.prepared:
        raise RuntimeError("pvalue 模式需要 --prepared")
    data = np.load(args.prepared, allow_pickle=True)
    if "z_k" not in data:
        raise RuntimeError("prepared.npz 缺少 z_k（每窗总数）")
    z_k = np.asarray(data["z_k"], dtype=int).reshape(-1)
    K_data = z_k.size

    leaves = list(data["leaves"])
    if args.scan_level == "groups":
        if "groups" in data and data["groups"].size > 0:
            groups = list(data["groups"])
        else:
            groups = build_groups_from_leaves(leaves)
        membership = build_membership_GxL(leaves, groups)
    else:
        groups = build_groups_from_leaves(leaves)
        membership = None

    # 和 boundaries 的 k 对齐
    # 假设 k 从 1 开始顺序编号；按 df 的 k 最大值裁剪
    K = int(min(df["k"].max(), K_data))
    df = df[df["k"].between(1, K)].copy()
    df = df.sort_values("k").reset_index(drop=True)

    # 取 alpha_k（同样对齐）
    if "alpha_k" not in df.columns:
        raise RuntimeError("boundaries.csv 中找不到 alpha_k（pvalue 模式必须）")
    alpha_k = df["alpha_k"].to_numpy(float)

    # baseline q_l^(k)
    q_l_k = get_q_l_k(data, args.baseline_mode, K)

    # 构造观测向量与累计 C^(k)
    T1_obs = df["T1"].to_numpy(float)
    T2_obs = df["T2"].to_numpy(float) if "T2" in df.columns else np.full(K, np.nan)
    U_k = np.cumsum(z_k[:K]).astype(float)

    # H0 重抽样并计算 p1/p2
    rng = np.random.default_rng(args.seed)
    p1 = np.ones(K)
    p2 = np.ones(K)
    for i, k in enumerate(df["k"].to_numpy(int)):
        z = int(z_k[k-1]); C = float(U_k[k-1]); q = q_l_k[k-1, :]
        if z <= 0 or q.sum() <= 0:
            p1[i], p2[i] = 1.0, 1.0
            continue
        top1_null, top2_null = null_top12_for_k(z, q, C, membership, B=args.B, rng=rng)
        p1[i] = p_from_rank(T1_obs[i], top1_null)
        p2[i] = 1.0 if np.isnan(T2_obs[i]) else p_from_rank(T2_obs[i], top2_null)

    thresh = alpha_k / 2.0
    min_p = np.minimum(p1, p2)
    cross = min_p <= thresh
    first_idx = int(np.where(cross)[0][0]) if np.any(cross) else None

    # 统一输出字段
    out = pd.DataFrame({
        "k": df["k"].astype(int),
        "T1": T1_obs,
        "unit1": df.get("unit1", np.nan),
        "T2": T2_obs,
        "unit2": df.get("unit2", np.nan),
        "decision_basis": "pvalue",
        "threshold": thresh,         # alpha_k/2
        "score": min_p,              # 与 threshold 比较
        "cross": cross,
        "first_alarm": False,
        "alpha_k": alpha_k,
        "alpha_k_over2": thresh,
        "p1": p1,
        "p2": p2,
        "min_p": min_p,
    })
    if first_idx is not None:
        out.loc[first_idx, "first_alarm"] = True

    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] p 值法完成 -> {out_path}")
    if first_idx is not None:
        r = out.iloc[first_idx]
        print(f"[ALARM] 首次越界 k={int(r['k'])} : min_p={r['min_p']:.4g} ≤ alpha_k/2={r['alpha_k_over2']:.4g} ; 单元={r['unit1']} / {r['unit2']}")
    else:
        print("[INFO] 未出现首次越界（Top-2/Bonferroni）。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", str(e))
        raise
