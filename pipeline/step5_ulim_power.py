#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part3 (Scheme-1 aligned): power & U_lim with group-level injection by default.

- 默认方案一：与 groups 扫描层级对齐，采用 random3（随机挑 1 个第三级组，整组注入 RR）。
- 若用户设置 scan-level=groups 但 omega-mode=random_leaf，则自动覆盖为 random3 并提示。
- 逐个 U 评估时，立即将 (phase, U, power, timestamp) 追加写入 step5/power_curve.csv。
- 收敛后输出 step5/boundaries.csv（含 c_k, U_k, t_k, alpha_k 等）。

示例：
  python part3_power.py \
    --outdir RUN_DIR --prepared RUN_DIR/step2/prepared.npz \
    --baseline-mode ext_qk --scan-level groups \
    --time-shape from-prepared \
    --alpha 0.05 --spend obf \
    --B0 20000 --A 2000 --seed 123 \
    --rr 1.8 --omega-mode random3 \
    --target-power 0.90 --U-start 3000 --U-step 200
"""
from __future__ import annotations
import argparse, os, sys, re, math, time, csv
from typing import List, Optional, Iterable, Tuple, Dict
import numpy as np
import pandas as pd
from statistics import NormalDist

# ---------------- Normal helpers ----------------
_STD_N = NormalDist(mu=0.0, sigma=1.0)
def norm_cdf(x: float) -> float:
    return _STD_N.cdf(float(x))
def norm_ppf(p: float) -> float:
    if not (0.0 < p < 1.0): raise ValueError(f"norm_ppf: p in (0,1), got {p}")
    return _STD_N.inv_cdf(float(p))

# ---------------- ICD helpers ----------------
_ICD3_RE = re.compile(r'^([A-Z][0-9]{2})')
def icd3(code: str) -> str:
    m = _ICD3_RE.match(str(code))
    return m.group(1) if m else str(code)[:3].upper()

def build_groups_from_leaves(leaves: List[str]) -> List[str]:
    return sorted({icd3(c) for c in leaves})

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

def parse_omega_to_leaf_idx(omega_codes: List[str], leaves: List[str]) -> np.ndarray:
    idx = {c:i for i,c in enumerate(leaves)}
    out = []
    for code in omega_codes:
        s = code.strip().upper()
        if not s: continue
        if len(s) == 3:  # level-3 cluster
            for i, leaf in enumerate(leaves):
                if icd3(leaf) == s:
                    out.append(i)
        else:            # leaf code
            if s in idx: out.append(idx[s])
    return np.unique(out)

# ---------------- LLR (one-sided Poisson) ----------------
def llr_one_sided(c: np.ndarray, u: np.ndarray, C: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float); u = np.asarray(u, float); C = np.asarray(C, float)
    c, u, C = np.broadcast_arrays(c, u, C)
    T = np.zeros_like(c, float)
    valid = (u > 0) & (C > u) & (c > u) & (C >= c)
    if not np.any(valid): return T
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

# ---------------- baseline q_l^{(k)} ----------------
def get_q_l_k(data, mode: str, K: int) -> np.ndarray:
    if mode == "ext_w":
        if "w_external" not in data or data["w_external"].size == 0:
            raise RuntimeError("ext_w: 'w_external' missing in prepared.npz")
        w = np.asarray(data["w_external"], float).reshape(-1)
        w = np.where(w<0, 0, w)
        w = w / (w.sum() if w.sum()>0 else 1.0)
        return np.repeat(w[None,:], K, axis=0)
    elif mode == "ext_qk":
        if "q_external" not in data or data["q_external"].size == 0:
            raise RuntimeError("ext_qk: 'q_external' missing in prepared.npz")
        q = np.asarray(data["q_external"], float)
        if q.shape[0] != K:
            last = q[-1:,:]
            q = np.concatenate([q, np.repeat(last, K-q.shape[0], axis=0)], axis=0) if K>q.shape[0] else q[:K,:]
        q = np.where(q<0, 0, q)
        s = q.sum(axis=1, keepdims=True); s[s<=0]=1.0
        return q / s
    else:  # internal
        if "q_internal" not in data or data["q_internal"].size == 0:
            raise RuntimeError("internal: 'q_internal' missing in prepared.npz")
        q = np.asarray(data["q_internal"], float)
        if q.shape[0] != K:
            last = q[-1:,:]
            q = np.concatenate([q, np.repeat(last, K-q.shape[0], axis=0)], axis=0) if K>q.shape[0] else q[:K,:]
        q = np.where(q<0, 0, q)
        s = q.sum(axis=1, keepdims=True); s[s<=0]=1.0
        return q / s

# ---------------- time-shape z_k for total U ----------------
def build_zk_for_U(U: int, shape: np.ndarray) -> np.ndarray:
    shp = np.asarray(shape, float).clip(min=0.0)
    if shp.ndim != 1 or shp.size == 0:
        raise ValueError("time-shape must be 1-D and non-empty")
    if shp.sum() <= 0:
        shp = np.ones_like(shp)
    p = shp / shp.sum()
    z = np.floor(U * p).astype(int)
    rem = int(U - z.sum())
    if rem > 0:
        order = np.argsort(-p)
        z[order[:rem]] += 1
    return z

# ---------------- alpha-spending ----------------
def spend_obf(t: np.ndarray, alpha: float) -> np.ndarray:
    z = norm_ppf(1 - alpha/2.0)
    out = np.zeros_like(t, float)
    mask = (t > 0) & (t <= 1)
    x = z / np.sqrt(np.maximum(t[mask], 1e-12))
    out[mask] = 2.0 - 2.0 * norm_cdf(x)
    out[t >= 1] = alpha
    return np.clip(out, 0.0, alpha)

def spend_pocock(t: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * np.log(1.0 + (math.e - 1.0) * np.clip(t, 0, 1))
def spend_linear(t: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * np.clip(t, 0, 1)
def spend_power(t: np.ndarray, alpha: float, rho: float) -> np.ndarray:
    return alpha * np.power(np.clip(t, 0, 1), rho)

def alpha_splits(U_k: np.ndarray, U_lim: int, alpha: float, spend: str) -> np.ndarray:
    if U_lim <= 0: return np.zeros_like(U_k, float)
    t = U_k / float(U_lim)
    if spend == "obf":       A = spend_obf(t, alpha)
    elif spend == "pocock":  A = spend_pocock(t, alpha)
    elif spend == "linear":  A = spend_linear(t, alpha)
    elif spend == "power1p5":A = spend_power(t, alpha, 1.5)
    elif spend == "power2":  A = spend_power(t, alpha, 2.0)
    else: raise ValueError(f"Unknown spend '{spend}'")
    A_prev = np.concatenate([[0.0], A[:-1]])
    return np.maximum(A - A_prev, 0.0)

# ---------------- H0 null paths ----------------
def simulate_null_paths(z_k: np.ndarray, q_l_k: np.ndarray, U_k: np.ndarray,
                        membership: Optional[np.ndarray], B0: int, seed: int) -> np.ndarray:
    if B0 <= 0: raise ValueError("B0 must be positive")
    rng = np.random.default_rng(seed)
    z_k = np.asarray(z_k, int); U_k = np.asarray(U_k, float)
    K, L = q_l_k.shape
    if z_k.size != K or U_k.size != K:
        raise ValueError(f"Shape mismatch: K={K}, len(z_k)={z_k.size}, len(U_k)={U_k.size}")
    lmax = np.zeros((K, B0), float)
    scan_groups = membership is not None
    if scan_groups: M = np.asarray(membership, int)
    for k in range(K):
        z = int(z_k[k]); C = float(U_k[k]); q = q_l_k[k,:].astype(float)
        q = np.clip(q, 0, None); s = q.sum(); q = q/s if s>0 else np.full(L, 1.0/L)
        counts = rng.multinomial(z, q, size=B0)  # (B0 x L)
        if scan_groups:
            c = counts @ M.T      # (B0 x G)
            qG = q @ M.T          # (G,)
            u = qG[None,:] * float(z)
        else:
            c = counts            # (B0 x L)
            u = q[None,:] * float(z)
        Cmat = np.full_like(c, C, float)
        T = llr_one_sided(c, u, Cmat)       # (B0 x M)
        lmax[k,:] = T.max(axis=1)
    return lmax  # (K x B0)

# ---------------- 首次越界边界校准（条件） ----------------
def _q_nearest(vals: np.ndarray, q: float) -> float:
    q = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(vals, q, method="nearest"))
    except TypeError:
        return float(np.quantile(vals, q, interpolation="nearest"))

def calibrate_boundaries(lmax: np.ndarray, alpha_k: np.ndarray) -> np.ndarray:
    K, B0 = lmax.shape
    c = np.zeros(K, float)
    alive = np.ones(B0, dtype=bool)
    for k in range(K):
        target = float(alpha_k[k])
        if target <= 0 or alive.sum() == 0:
            c[k] = float('inf'); alive[:] = False; continue
        m = int(alive.sum())
        cond_tail = target * (B0 / float(m))
        cond_tail = float(np.clip(cond_tail, 0.0, 1.0))
        vals = lmax[k, alive]
        c[k] = _q_nearest(vals, 1.0 - cond_tail)
        alive[alive] &= (lmax[k, alive] < c[k])
    return c

# ---------------- RR 注入 & 随机 Ω（方案一：组级） ----------------
def inject_rr_to_q(q_l_k: np.ndarray, omega_idx: Iterable[int], rr: float) -> np.ndarray:
    q = np.asarray(q_l_k, float).copy()
    omega_idx = np.asarray(list(omega_idx), int)
    if omega_idx.size == 0: return q
    for k in range(q.shape[0]):
        v = q[k,:].copy(); v[omega_idx] *= rr
        s = v.sum(); q[k,:] = v/s if s>0 else np.full(q.shape[1], 1.0/q.shape[1])
    return q

def sample_random3_omega(rng: np.random.Generator,
                         membership: np.ndarray) -> np.ndarray:
    g = rng.integers(0, membership.shape[0])  # pick one group
    return np.where(membership[g] == 1)[0]

# ---------------- 功效模拟 ----------------
def simulate_power(z_k: np.ndarray, U_k: np.ndarray, q0_l_k: np.ndarray,
                   membership: Optional[np.ndarray], c_k: np.ndarray,
                   leaves: List[str], groups: List[str],
                   rr: float, A: int, seed: int,
                   omega_mode: str, fixed_omega_idx: np.ndarray,
                   omega_nleaf: int) -> float:
    rng = np.random.default_rng(seed)
    K, L = q0_l_k.shape
    crosses = 0
    scan_groups = membership is not None
    if scan_groups: M = membership

    for a in range(A):
        # 方案一：若 groups 扫描，则优先使用整组注入
        if omega_mode == "fixed":
            omega_idx = fixed_omega_idx
        elif scan_groups:
            omega_idx = sample_random3_omega(rng, M)  # group-level injection
        else:
            # 叶级扫描时，允许 random_leaf（或 fixed）
            if omega_mode == "random_leaf":
                nleaf = max(1, min(int(omega_nleaf), L))
                omega_idx = rng.choice(np.arange(L), size=nleaf, replace=False)
            else:
                # fallback: 整组随机（对叶级扫描也可用）
                omega_idx = np.array([], dtype=int)

        p_l_k = inject_rr_to_q(q0_l_k, omega_idx, rr)
        for k in range(K):
            z = int(z_k[k]); C = float(U_k[k]); p = p_l_k[k,:]
            counts = rng.multinomial(z, p)  # (L,)
            if scan_groups:
                c = counts @ M.T
                qG = q0_l_k[k,:] @ M.T
                u = qG * float(z)
                lam = float(llr_one_sided(c, u, np.full_like(c, C, float)).max())
            else:
                u = q0_l_k[k,:] * float(z)
                lam = float(llr_one_sided(counts, u, np.full(counts.shape, C, float)).max())
            if lam >= c_k[k]:
                crosses += 1
                break
    return crosses / float(A)

# ---------------- 单点评估（给定 U） ----------------
def evaluate_power_for_U(U: int, shape: np.ndarray, q_l_k: np.ndarray,
                         membership: Optional[np.ndarray], alpha: float, spend: str,
                         B0: int, A: int, seed: int,
                         leaves: List[str], groups: List[str],
                         rr: float, omega_mode: str,
                         fixed_omega_idx: np.ndarray, omega_nleaf: int
                         ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    z_k = build_zk_for_U(U, shape)   # (K,)
    U_k = z_k.cumsum()               # (K,)
    alpha_k = alpha_splits(U_k, U, alpha, spend)  # (K,)
    lmax = simulate_null_paths(z_k, q_l_k, U_k, membership, B0=B0, seed=seed)
    c_k = calibrate_boundaries(lmax, alpha_k)
    power = simulate_power(z_k, U_k, q_l_k, membership, c_k,
                           leaves, groups, rr, A, seed+7,
                           omega_mode, fixed_omega_idx, omega_nleaf)
    return power, c_k, U_k, alpha_k

# ---------------- CSV 追加日志 ----------------
def append_power_log(csv_path: str, phase: str, U: int, power: float):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["phase","U","power","timestamp"])
        w.writerow([phase, int(U), float(power), time.strftime("%Y-%m-%d %H:%M:%S")])

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Part3 (scheme-1 aligned): power & U_lim")
    ap.add_argument("--outdir", required=True, type=str)
    ap.add_argument("--prepared", required=True, type=str)
    ap.add_argument("--baseline-mode", choices=["ext_qk","ext_w","internal"], default="ext_qk")
    ap.add_argument("--scan-level", choices=["groups","leaves"], default="groups")

    ap.add_argument("--time-shape", choices=["from-prepared","uniform-K","csv"], default="from-prepared")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--z-shape-csv", type=str, default=None)

    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--spend", choices=["obf","pocock","linear","power1p5","power2"], default="obf")
    ap.add_argument("--B0", type=int, default=20000)
    ap.add_argument("--A", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--omega-mode", choices=["fixed","random3","random_leaf"], default="random3")
    ap.add_argument("--omega", type=str, default="")
    ap.add_argument("--omega-nleaf", type=int, default=1)

    ap.add_argument("--target-power", type=float, default=0.90)
    ap.add_argument("--U-start", type=int, required=True)
    ap.add_argument("--U-step", type=int, required=True)
    ap.add_argument("--U-max", type=int, default=None)

    args = ap.parse_args()

    # 输出目录
    step5_dir = os.path.join(args.outdir, "step5")
    os.makedirs(step5_dir, exist_ok=True)
    out_power = os.path.join(step5_dir, "power_curve.csv")
    out_bounds = os.path.join(step5_dir, "boundaries.csv")

    # 读 prepared
    data = np.load(args.prepared, allow_pickle=True)
    leaves = list(data["leaves"])

    # groups / membership
    if args.scan_level == "groups":
        if "groups" in data and data["groups"].size > 0:
            groups = list(data["groups"])
        else:
            groups = build_groups_from_leaves(leaves)
        membership = build_membership_GxL(leaves, groups)
    else:
        groups = build_groups_from_leaves(leaves)  # 仅供 fixed/random_leaf 解析用
        membership = None

    # 时间形状
    if args.time_shape == "from-prepared":
        if "z_k" not in data:
            raise RuntimeError("prepared.npz missing 'z_k' for from-prepared")
        z_obs = np.asarray(data["z_k"], int).reshape(-1)
        if z_obs.sum() <= 0: raise RuntimeError("prepared.z_k has non-positive sum")
        shape = z_obs / z_obs.sum()
    elif args.time_shape == "uniform-K":
        if not args.K or args.K <= 0:
            raise RuntimeError("Provide --K for uniform-K")
        shape = np.full(args.K, 1.0/args.K, float)
    else:
        if not args.z_shape_csv:
            raise RuntimeError("--z-shape-csv required when time-shape=csv")
        dfz = pd.read_csv(args.z_shape_csv).sort_values("k")
        z = dfz["z"].to_numpy(dtype=float)
        if z.sum() <= 0: raise RuntimeError("time-shape csv has non-positive sum")
        shape = z / z.sum()

    K = shape.shape[0]
    q_l_k = get_q_l_k(data, args.baseline_mode, K)

    # 固定 Ω（仅 fixed 用）
    if args.omega_mode == "fixed":
        fixed_omega_idx = parse_omega_to_leaf_idx([s for s in args.omega.split(",") if s], leaves)
    else:
        fixed_omega_idx = np.array([], dtype=int)

    # 若 groups 扫描，但用户传了 random_leaf，则覆盖为 random3（方案一）
    if (args.scan_level == "groups") and (args.omega_mode == "random_leaf"):
        print("[INFO] scan-level=groups & omega-mode=random_leaf detected; "
              "switching to omega-mode=random3 to align with scheme-1.", flush=True)
        args.omega_mode = "random3"

    # U_max 缺省：≥ 已观测总量的 2.5 倍，或起点 + 10*步长，或 100000
    if args.U_max is None:
        total_obs = int(np.asarray(data["z_k"]).sum()) if "z_k" in data else 1000
        args.U_max = max(int(2.5 * total_obs), args.U_start + 10*args.U_step, 100000)

    cache: Dict[int, Tuple[float,np.ndarray,np.ndarray,np.ndarray]] = {}

    def get_power(U: int):
        if U in cache: return cache[U], 0.0
        t0 = time.perf_counter()
        res = evaluate_power_for_U(
            U, shape, q_l_k, membership, args.alpha, args.spend,
            args.B0, args.A, args.seed,
            leaves, groups, args.rr, args.omega_mode,
            fixed_omega_idx, args.omega_nleaf
        )
        dt = time.perf_counter() - t0
        cache[U] = res
        return res, dt

    # 逐步提升
    U = int(args.U_start)
    prevU = None
    reached = False
    while U <= args.U_max:
        (power, c_k, U_k, alpha_k), dt = get_power(U)
        append_power_log(out_power, "escalate", U, power)
        print(f"[step5] escalate  U={U:>5d}  power={power:.3f}  dt={dt:.1f}s", flush=True)
        if power >= args.target_power:
            U_hi = U
            U_lo = prevU if prevU is not None else max(1, U - args.U_step)
            reached = True
            break
        prevU = U
        U += args.U_step

    if not reached:
        print("[WARN] Escalation did not reach target power within U_max.")
        print(f"[INFO] power_curve -> {out_power}")
        sys.exit(0)

    # 二分细化
    lo, hi = int(U_lo), int(U_hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        (p_mid, _, _, _), dt = get_power(mid)
        append_power_log(out_power, "binary", mid, p_mid)
        print(f"[step5] binary    U={mid:>5d}  power={p_mid:.3f}  dt={dt:.1f}s", flush=True)
        if p_mid >= args.target_power:
            hi = mid
        else:
            lo = mid + 1

    U_lim = hi
    power_star, c_k_star, U_k_star, alpha_k_star = cache[U_lim]
    append_power_log(out_power, "final", U_lim, power_star)
    print(f"[step5] FINAL     U={U_lim:>5d}  power={power_star:.3f}", flush=True)

    # 输出边界
    t_k = U_k_star / float(U_lim)
    dfb = pd.DataFrame({
        "k": np.arange(1, len(c_k_star)+1, dtype=int),
        "c_k": c_k_star,
        "U_k": U_k_star,
        "t_k": t_k,
        "alpha_k": alpha_k_star,
        "U_lim": U_lim,
        "alpha_total": args.alpha,
        "spend": args.spend,
        "baseline_mode": args.baseline_mode,
        "scan_level": args.scan_level,
        "target_power": args.target_power,
        "omega_mode": args.omega_mode,
        "omega_nleaf": args.omega_nleaf
    })
    dfb.to_csv(out_bounds, index=False)
    print(f"[OK] power_curve -> {out_power}")
    print(f"[OK] boundaries  -> {out_bounds}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", str(e))
        raise
