#!/usr/bin/env python3
"""CRISP-DM 多元線性回歸流程（Multiple Linear Regression pipeline）

用法：
    python mlr_pipeline.py --url <CSV 網址或路徑> --target <目標欄位> \
        --name "<資料集名稱>" --slug <輸出資料夾名稱> [--business "<商業問題>"]

輸出（reports/<slug>/）：
    report.md      CRISP-DM 六步驟分析報告
    results.json   係數、檢定、指標等數值結果
    figures/*.png  相關係數熱圖、實際 vs 預測、殘差圖、標準化係數

型態判斷、One-hot、資料切分（mulberry32 + Fisher–Yates）與統計方法
都和網頁版 assets/js/mlr.js 相同，所以兩邊的數字可以互相對照。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

M32 = 0xFFFFFFFF
MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-"}
NUM_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
ID_NAME_RE = re.compile(r"(id|index|unnamed: ?\d+)", re.IGNORECASE)


# ----------------------------------------------------------------- 基本工具

def is_missing(v) -> bool:
    return v is None or str(v).strip().lower() in MISSING


def to_num(v) -> float:
    s = "" if v is None else str(v).strip()
    return float(s) if NUM_RE.fullmatch(s) else math.nan


def mulberry32(seed: int):
    """與 JavaScript 版逐位元相同的 32-bit 亂數產生器。"""
    a = seed & M32

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (a | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return rand


def train_test_split_idx(n: int, test_ratio: float, seed: int):
    rand = mulberry32(seed)
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = math.floor(rand() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    n_test = math.floor(n * test_ratio + 0.5)  # 同 JavaScript Math.round
    return idx[n_test:], idx[:n_test]


# ------------------------------------------------------------ 資料理解與準備

def load_table(src: str, sep: str) -> pd.DataFrame:
    raw = pd.read_csv(src, sep=sep, header=None, dtype=str, keep_default_na=False, skip_blank_lines=True)
    raw = raw.apply(lambda s: s.str.strip())
    first = raw.iloc[0].tolist()
    has_header = not all(math.isfinite(to_num(v)) for v in first)
    if has_header:
        df = raw.iloc[1:].reset_index(drop=True)
        df.columns = [h if h else f"col_{i + 1}" for i, h in enumerate(first)]
    else:
        df = raw
        df.columns = [f"col_{i + 1}" for i in range(raw.shape[1])]
    return df


def infer_columns(df: pd.DataFrame) -> list[dict]:
    cols = []
    n = len(df)
    for j, name in enumerate(df.columns):
        present = [str(v).strip() for v in df[name] if not is_missing(v)]
        nums = [to_num(v) for v in present]
        numeric = bool(present) and all(math.isfinite(x) for x in nums)
        unique = len(set(present))
        col = dict(name=name, index=j, type="numeric" if numeric else "categorical",
                   missing=n - len(present), unique=unique, exclude=None)
        if not present:
            col.update(type="empty", exclude="全部為空值")
        elif unique == 1:
            col["exclude"] = "常數欄位（只有一種值）"
        elif numeric:
            all_int = all(float(x).is_integer() for x in nums)
            if unique == len(present) and (ID_NAME_RE.fullmatch(name) or (all_int and max(nums) - min(nums) == len(present) - 1)):
                col["exclude"] = "疑似 ID / 流水號"
        elif unique > 30 or (len(present) > 10 and unique / len(present) > 0.5):
            col["exclude"] = "類別過多（疑似 ID 或文字欄）"
        cols.append(col)
    return cols


def build_design(df: pd.DataFrame, cols: list[dict], target: str, features: list[str]):
    info = {c["name"]: c for c in cols}
    ok = df[target].map(lambda v: math.isfinite(to_num(v)))
    for f in features:
        if info[f]["type"] == "numeric":
            ok &= df[f].map(lambda v: math.isfinite(to_num(v)))
        else:
            ok &= ~df[f].map(is_missing)
    used = df[ok].reset_index(drop=True)

    names, blocks, encodings = [], [], []
    for f in features:
        if info[f]["type"] == "numeric":
            blocks.append(used[f].map(to_num).to_numpy(dtype=float))
            names.append(f)
            encodings.append(dict(feature=f, type="numeric", columns=[f]))
        else:
            levels = sorted(set(used[f]))
            dummies = [f"{f}_{lv}" for lv in levels[1:]]
            for lv in levels[1:]:
                blocks.append((used[f] == lv).astype(float).to_numpy())
            names.extend(dummies)
            encodings.append(dict(feature=f, type="categorical", levels=levels, baseline=levels[0], columns=dummies))
    X = np.column_stack(blocks) if blocks else np.empty((len(used), 0))
    y = used[target].map(to_num).to_numpy(dtype=float)
    return dict(X=X, y=y, names=names, encodings=encodings, rows=used, n_dropped=int((~ok).sum()))


# -------------------------------------------------------------------- 建模

def fit_ols(X: np.ndarray, y: np.ndarray, names: list[str], cols: list[int] | None = None) -> dict:
    n = len(y)
    keep = list(range(X.shape[1])) if cols is None else list(cols)
    if n <= len(keep) + 1:
        raise ValueError(f"資料只有 {n} 筆，不足以估計 {len(keep) + 1} 個參數")
    dropped = []
    while True:
        A = np.column_stack([np.ones(n)] + [X[:, j] for j in keep])
        Q, R = np.linalg.qr(A)
        col_norm = np.linalg.norm(A, axis=0)
        bad = next((k for k in range(A.shape[1]) if not abs(R[k, k]) > 1e-10 * col_norm[k]), -1)
        if bad == -1:
            break
        if bad == 0:
            raise ValueError("無法估計截距")
        dropped.append(names[keep[bad - 1]])
        keep.pop(bad - 1)

    p = A.shape[1]
    beta = np.linalg.solve(R, Q.T @ y)
    fitted = A @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    df_resid, df_model = n - p, p - 1
    sigma2 = rss / df_resid
    Rinv = np.linalg.inv(R)
    xtx_inv = Rinv @ Rinv.T
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), df_resid)
    tcrit = stats.t.ppf(0.975, df_resid)
    r2 = 1 - rss / tss
    F = ((tss - rss) / df_model) / sigma2 if df_model > 0 else math.nan
    llf = -n / 2 * (math.log(2 * math.pi) + math.log(rss / n) + 1)
    coefs = [dict(name="(Intercept)" if j == 0 else names[keep[j - 1]], estimate=float(beta[j]), se=float(se[j]),
                  t=float(t[j]), p=float(pv[j]), lower=float(beta[j] - tcrit * se[j]), upper=float(beta[j] + tcrit * se[j]))
             for j in range(p)]
    return dict(feature_index=keep, features=[names[j] for j in keep], dropped_collinear=dropped, coefficients=coefs,
                beta=beta, n=n, p=p, df_resid=df_resid, df_model=df_model, rss=rss, tss=tss, r2=r2,
                adj_r2=1 - (1 - r2) * (n - 1) / df_resid, sigma=math.sqrt(sigma2), F=F,
                F_p=float(stats.f.sf(F, df_model, df_resid)) if df_model > 0 else math.nan,
                aic=-2 * llf + 2 * p, bic=-2 * llf + math.log(n) * p)


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    idx = model["feature_index"]
    out = np.full(X.shape[0], model["beta"][0], dtype=float)
    if idx:
        out += X[:, idx] @ model["beta"][1:]
    return out


def metrics(y: np.ndarray, yhat: np.ndarray, k: int) -> dict:
    n = len(y)
    r2 = float(r2_score(y, yhat))
    return dict(n=n, r2=r2, adj_r2=1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else math.nan,
                rmse=float(math.sqrt(mean_squared_error(y, yhat))), mae=float(mean_absolute_error(y, yhat)))


def vif(X: np.ndarray, names: list[str], cols: list[int]) -> list[float]:
    if len(cols) < 2:
        return [1.0] * len(cols)
    out = []
    for j in cols:
        m = fit_ols(X, X[:, j], names, [k for k in cols if k != j])
        out.append(math.inf if m["r2"] >= 1 else 1 / (1 - m["r2"]))
    return out


def standardized(model: dict, X: np.ndarray, y: np.ndarray) -> list[float]:
    sd_y = y.std(ddof=1)
    return [float(model["beta"][k + 1] * X[:, j].std(ddof=1) / sd_y) for k, j in enumerate(model["feature_index"])]


def backward_elimination(X, y, names, sl, cols):
    active, steps = list(cols), []
    while True:
        m = fit_ols(X, y, names, active)
        pv = [c["p"] for c in m["coefficients"][1:]]
        if not pv or not max(pv) > sl:
            return m, steps
        worst = int(np.argmax(pv))
        steps.append(dict(removed=m["coefficients"][worst + 1]["name"], p=pv[worst], r2=m["r2"],
                          adj_r2=m["adj_r2"], n_features=len(active)))
        active = [j for k, j in enumerate(m["feature_index"]) if k != worst]


# ------------------------------------------------- 特徵選擇：依特徵數 k 比較

SELECTION_METHODS = [
    ("backward_p", "逆向淘汰（p 值）"),
    ("forward", "前向選擇 Forward"),
    ("rfe", "RFE（|標準化係數|）"),
    ("best", "最佳子集 Exhaustive"),
    ("lasso", "Lasso（L1 排序）"),
    ("enet", "ElasticNet（α = 0.5）"),
]


def standardized_gram(X, y, cols):
    """A = ZᵀZ、c = Zᵀ(y − ȳ)，Z 為標準化特徵；任一子集的 RSS 只需解 k×k 系統。"""
    Z = X[:, cols]
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
    yc = y - y.mean()
    return dict(A=Z.T @ Z, c=Z.T @ yc, syy=float(yc @ yc), n=len(y), K=len(cols))


def subset_solve(g, S):
    if not S:
        return g["syy"], np.array([])
    try:
        L = np.linalg.cholesky(g["A"][np.ix_(S, S)])
    except np.linalg.LinAlgError:
        return math.inf, None
    z = solve_triangular(L, g["c"][S], lower=True)
    b = solve_triangular(L.T, z, lower=False)
    return g["syy"] - float(z @ z), b


def forward_path(g):
    chosen, left, subsets = [], list(range(g["K"])), {}
    while left:
        best, best_rss = -1, math.inf
        for a in left:
            rss, _ = subset_solve(g, chosen + [a])
            if best == -1 or rss < best_rss:
                best, best_rss = a, rss
        chosen.append(best)
        left.remove(best)
        subsets[len(chosen)] = list(chosen)
    return subsets


def rfe_path(g):
    S = list(range(g["K"]))
    subsets = {len(S): list(S)}
    while len(S) > 1:
        _, b = subset_solve(g, S)
        worst = int(np.argmin(np.abs(b)))
        S = [a for i, a in enumerate(S) if i != worst]
        subsets[len(S)] = list(S)
    return subsets


def pvalue_path(X, y, names, cols):
    active = list(cols)
    subsets = {len(active): [cols.index(j) for j in active]}
    while len(active) > 1:
        m = fit_ols(X, y, names, active)
        worst = int(np.argmax([c["p"] for c in m["coefficients"][1:]]))
        active = [j for k, j in enumerate(m["feature_index"]) if k != worst]
        subsets[len(active)] = [cols.index(j) for j in active]
    return subsets


def best_subset_path(g, max_k=15):
    K = g["K"]
    if K > max_k:
        return None
    best = {}
    for mask in range(1, 1 << K):
        S = [a for a in range(K) if mask >> a & 1]
        rss, _ = subset_solve(g, S)
        if len(S) not in best or rss < best[len(S)][0]:
            best[len(S)] = (rss, S)
    return {k: v[1] for k, v in best.items()}


def enet_path(g, l1_ratio, n_lambda=100, eps=1e-4):
    """座標下降走 λ 路徑，依特徵第一次進入模型的順序排序（Lasso: l1_ratio = 1）。"""
    K, n = g["K"], g["n"]
    A = g["A"].tolist()
    sd_y = math.sqrt(g["syy"] / (n - 1))
    c = [v / sd_y for v in g["c"]]
    lam_max = max(abs(v) / (n * l1_ratio) for v in c)
    coef, enter = [0.0] * K, [math.inf] * K
    for s in range(n_lambda):
        lam = lam_max * eps ** (s / (n_lambda - 1))
        for _ in range(2000):
            max_delta = 0.0
            for j in range(K):
                r = c[j]
                for l_ in range(K):
                    if l_ != j:
                        r -= A[j][l_] * coef[l_]
                z = r / n
                nb = math.copysign(max(abs(z) - lam * l1_ratio, 0.0), z) / (A[j][j] / n + lam * (1 - l1_ratio))
                max_delta = max(max_delta, abs(nb - coef[j]))
                coef[j] = nb
            if max_delta < 1e-10:
                break
        for j in range(K):
            if enter[j] == math.inf and abs(coef[j]) > 1e-12:
                enter[j] = s
    order = sorted(range(K), key=lambda a: (enter[a], -abs(coef[a]), a))
    return {k: order[:k] for k in range(1, K + 1)}


def feature_selection(Xtr, ytr, Xte, yte, names, cols):
    g = standardized_gram(Xtr, ytr, cols)
    subsets_by = dict(backward_p=pvalue_path(Xtr, ytr, names, cols), forward=forward_path(g), rfe=rfe_path(g),
                      best=best_subset_path(g), lasso=enet_path(g, 1.0), enet=enet_path(g, 0.5))
    cache = {}

    def evaluate(S):
        sub = sorted(cols[a] for a in S)
        if tuple(sub) not in cache:
            m = fit_ols(Xtr, ytr, names, sub)
            cache[tuple(sub)] = dict(cols=sub, features=m["features"], k=len(sub),
                                     train=metrics(ytr, predict(m, Xtr), m["p"] - 1),
                                     test=metrics(yte, predict(m, Xte), m["p"] - 1))
        return cache[tuple(sub)]

    methods = [dict(key=key, label=label, results={k: evaluate(subsets_by[key][k]) for k in range(1, len(cols) + 1)})
               for key, label in SELECTION_METHODS if subsets_by[key] is not None]
    best = None
    for m in methods:
        for k in range(1, len(cols) + 1):
            res = m["results"][k]
            score = res["test"]["rmse"]
            if best is None or score < best["score"] or (score == best["score"] and k < best["k"]):
                best = {**res, "score": score, "method": m["key"], "label": m["label"]}
    best["same_methods"] = [m["label"] for m in methods if m["results"][best["k"]]["cols"] == best["cols"]]
    return dict(K=len(cols), methods=methods, best=best)


# -------------------------------------------------------------------- 圖表

COLORS = dict(train="#3b5bdb", test="#e8590c", pos="#0c8599", neg="#d6336c", muted="#868e96")


def plot_corr(corr: pd.DataFrame, path: Path):
    k = len(corr)
    fig, ax = plt.subplots(figsize=(max(4.8, 0.62 * k + 2), max(4, 0.55 * k + 1.4)))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(k), corr.columns)
    if k <= 16:
        for i in range(k):
            for j in range(k):
                v = corr.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7 if k > 10 else 9,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix (Pearson)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_actual_vs_pred(ytr, ptr, yte, pte, target, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(ytr, ptr, s=22, alpha=0.7, color=COLORS["train"], label="Train")
    if len(yte):
        ax.scatter(yte, pte, s=30, alpha=0.9, color=COLORS["test"], label="Test", marker="D")
    allv = np.concatenate([ytr, ptr, yte, pte])
    lo, hi = allv.min(), allv.max()
    ax.plot([lo, hi], [lo, hi], ls="--", color=COLORS["muted"], lw=1, label="y = x")
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title("Actual vs Predicted (full model)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(ptr, rtr, pte, rte, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(ptr, rtr, s=22, alpha=0.7, color=COLORS["train"], label="Train")
    if len(pte):
        ax.scatter(pte, rte, s=30, alpha=0.9, color=COLORS["test"], label="Test", marker="D")
    ax.axhline(0, ls="--", color=COLORS["muted"], lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals vs Predicted (full model)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_std_coef(names, values, pvals, sl, path: Path):
    order = np.argsort(np.abs(values))
    fig, ax = plt.subplots(figsize=(6, max(2.6, 0.38 * len(names) + 1.2)))
    for rank, i in enumerate(order):
        color = COLORS["pos"] if values[i] >= 0 else COLORS["neg"]
        ax.barh(rank, values[i], color=color, alpha=1.0 if pvals[i] < sl else 0.35)
    ax.set_yticks(range(len(names)), [names[i] for i in order])
    ax.axvline(0, color=COLORS["muted"], lw=1)
    ax.set_xlabel("Standardized coefficient (faded = not significant)")
    ax.set_title("Feature influence (full model)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


LINE_COLORS = ["#3b5bdb", "#e8590c", "#0c8599", "#d6336c", "#5c940d", "#7048e8"]
LINE_MARKERS = ["o", "s", "^", "D", "x", "*"]


def plot_selection(sel: dict, path: Path):
    ks = list(range(1, sel["K"] + 1))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, m in enumerate(sel["methods"]):
        ax.plot(ks, [m["results"][k]["test"]["rmse"] for k in ks], marker=LINE_MARKERS[i % 6], ms=5, lw=1.6,
                color=LINE_COLORS[i % 6], alpha=0.85, label=m["key"])
    b = sel["best"]
    ax.scatter([b["k"]], [b["test"]["rmse"]], marker="*", s=260, color="black", zorder=5, label=f"best k={b['k']}")
    ax.set_xticks(ks)
    ax.set_xlabel("Number of features k")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Feature selection: test RMSE vs k")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- 報告

def fmt(x, sig: int = 5) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e9 or ax < 1e-4:
        return f"{x:.2e}"
    digits = min(max(0, sig - 1 - math.floor(math.log10(ax))), 8)
    return f"{x:,.{digits}f}"


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


def md_table(header: list[str], rows: list[list[str]]) -> str:
    def cell(s: str) -> str:
        return str(s).replace("|", "\\|")  # 儲存格內的 | 會破壞 Markdown 表格

    lines = ["| " + " | ".join(map(cell, header)) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(map(cell, r)) + " |" for r in rows]
    return "\n".join(lines)


def equation(model: dict, target: str) -> str:
    parts = [fmt(model["coefficients"][0]["estimate"])]
    for c in model["coefficients"][1:]:
        sign = "−" if c["estimate"] < 0 else "+"
        parts.append(f"{sign} {fmt(abs(c['estimate']))} × {c['name']}")
    return f"{target} = " + " ".join(parts)


def coef_table(model: dict, sl: float, vifs=None, stdc=None) -> str:
    header = ["變數", "係數", "標準誤", "t", "p 值", "95% 信賴區間"]
    if vifs is not None:
        header += ["VIF", "標準化係數"]
    rows = []
    for k, c in enumerate(model["coefficients"]):
        name = f"**{c['name']}**" if k > 0 and c["p"] < sl else c["name"]
        row = [name, fmt(c["estimate"]), fmt(c["se"]), f"{c['t']:.3f}", f"{fmt_p(c['p'])} {stars(c['p'])}".strip(),
               f"[{fmt(c['lower'])}, {fmt(c['upper'])}]"]
        if vifs is not None:
            row += (["—", "—"] if k == 0 else [fmt(vifs[k - 1], 4), f"{stdc[k - 1]:.3f}"])
        rows.append(row)
    return md_table(header, rows)


def main():
    ap = argparse.ArgumentParser(description="CRISP-DM multiple linear regression report")
    ap.add_argument("--url", required=True, help="CSV 網址或本機路徑")
    ap.add_argument("--target", help="目標欄位（預設：最後一個數值欄位）")
    ap.add_argument("--name", default="Dataset", help="資料集名稱（報告標題）")
    ap.add_argument("--slug", default="dataset", help="輸出資料夾名稱 reports/<slug>")
    ap.add_argument("--business", default="", help="商業理解：要解決的問題（一段文字）")
    ap.add_argument("--exclude", nargs="*", default=[], help="不放入模型的欄位")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sl", type=float, default=0.05, help="逆向淘汰顯著水準")
    ap.add_argument("--sep", default=",", help="分隔符號（預設逗號）")
    ap.add_argument("--out-root", default="reports")
    ap.add_argument("--site", default="", help="互動式網頁網址（寫進報告的部署段落）")
    args = ap.parse_args()

    out = Path(args.out_root) / args.slug
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # 2. Data Understanding
    df = load_table(args.url, args.sep)
    cols = infer_columns(df)
    numeric = [c for c in cols if c["type"] == "numeric" and not c["exclude"]]
    target = args.target or numeric[-1]["name"]
    if target not in df.columns:
        raise SystemExit(f"找不到目標欄位 {target}；可用欄位：{list(df.columns)}")
    features = [c["name"] for c in cols if c["name"] != target and not c["exclude"]
                and c["type"] != "empty" and c["name"] not in args.exclude]

    # 3. Data Preparation
    d = build_design(df, cols, target, features)
    X, y, names = d["X"], d["y"], d["names"]
    train_idx, test_idx = train_test_split_idx(len(y), args.test_ratio, args.seed)
    if len(test_idx) < 2:
        raise SystemExit("測試集少於 2 筆，請提高 --test-ratio")
    Xtr, ytr, Xte, yte = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    # 4. Modeling
    full = fit_ols(Xtr, ytr, names)
    reduced, steps = backward_elimination(Xtr, ytr, names, args.sl, full["feature_index"])
    vifs = vif(Xtr, names, full["feature_index"])
    std_full = standardized(full, Xtr, ytr)
    sel = feature_selection(Xtr, ytr, Xte, yte, names, full["feature_index"])
    sweet = fit_ols(Xtr, ytr, names, sel["best"]["cols"])

    # 與 scikit-learn 交叉驗證係數
    sk = LinearRegression().fit(Xtr[:, full["feature_index"]], ytr)
    assert np.allclose(sk.coef_, full["beta"][1:], rtol=1e-6, atol=1e-8), "OLS 係數與 scikit-learn 不一致"
    assert math.isclose(sk.intercept_, full["beta"][0], rel_tol=1e-6, abs_tol=1e-8)

    # 5. Evaluation
    def evaluate(m):
        return dict(train=metrics(ytr, predict(m, Xtr), m["p"] - 1),
                    test=metrics(yte, predict(m, Xte), m["p"] - 1) if len(yte) > 1 else None)

    ev = dict(full=evaluate(full), reduced=evaluate(reduced), sweet=evaluate(sweet))

    numeric_used = [f for f in features if next(c for c in cols if c["name"] == f)["type"] == "numeric"] + [target]
    corr = d["rows"][numeric_used].apply(lambda s: s.map(to_num)).corr()
    plot_corr(corr, figs / "correlation.png")
    ptr, pte = predict(full, Xtr), predict(full, Xte)
    plot_actual_vs_pred(ytr, ptr, yte, pte, target, figs / "actual_vs_predicted.png")
    plot_residuals(ptr, ytr - ptr, pte, yte - pte, figs / "residuals.png")
    plot_std_coef(full["features"], np.array(std_full), np.array([c["p"] for c in full["coefficients"][1:]]),
                  args.sl, figs / "standardized_coefficients.png")
    plot_selection(sel, figs / "feature_selection.png")

    # ------------------------------------------------------------ report.md
    L = []
    L.append(f"# {args.name}：多元線性回歸分析報告（CRISP-DM）\n")
    L.append(f"> 由 `mlr-crisp-dm` skill 產生 · 資料來源：<{args.url}> · "
             f"{datetime.now():%Y-%m-%d %H:%M}\n")

    L.append("## 1. Business Understanding 商業理解\n")
    L.append((args.business or f"以多元線性回歸預測 **{target}**，並找出影響它的主要因素。") + "\n")
    L.append(f"- 目標變數 y：`{target}`\n- 候選特徵 X：{', '.join(f'`{f}`' for f in features)}\n")

    L.append("## 2. Data Understanding 資料理解\n")
    L.append(f"資料共 **{len(df)} 筆 × {df.shape[1]} 欄**，缺失值 {sum(c['missing'] for c in cols)} 個。\n")
    L.append(md_table(["欄位", "型態", "缺失", "唯一值", "備註"],
                      [[c["name"], {"numeric": "數值", "categorical": "類別", "empty": "空"}[c["type"]],
                        str(c["missing"]), str(c["unique"]),
                        "目標 y" if c["name"] == target else (c["exclude"] or "")] for c in cols]) + "\n")
    desc_rows = []
    for c in cols:
        if c["type"] != "numeric":
            continue
        v = pd.Series([to_num(x) for x in df[c["name"]] if not is_missing(x)])
        q = v.quantile([0.25, 0.5, 0.75])
        desc_rows.append([c["name"], str(len(v)), fmt(v.mean()), fmt(v.std()), fmt(v.min()), fmt(q[0.25]),
                          fmt(q[0.5]), fmt(q[0.75]), fmt(v.max())])
    L.append("**敘述統計**\n")
    L.append(md_table(["欄位", "count", "mean", "std", "min", "25%", "50%", "75%", "max"], desc_rows) + "\n")
    for c in cols:
        if c["type"] == "categorical" and not c["exclude"]:
            counts = df[c["name"]].value_counts().sort_index()
            L.append(f"**{c['name']} 類別分布**：" + "、".join(f"{k} {v} 筆" for k, v in counts.items()) + "\n")
    corr_t = corr[target].drop(target).sort_values(key=np.abs, ascending=False)
    L.append("**與目標的相關係數（依絕對值排序）**：" +
             "、".join(f"`{k}` {v:+.3f}" for k, v in corr_t.items()) + "\n")
    L.append("![Correlation](figures/correlation.png)\n")

    L.append("## 3. Data Preparation 資料準備\n")
    L.append(f"- 缺失值：{'刪除 ' + str(d['n_dropped']) + ' 筆含缺失值的資料' if d['n_dropped'] else '無缺失值，不需處理'}")
    for e in d["encodings"]:
        if e["type"] == "categorical":
            L.append(f"- One-hot encoding：`{e['feature']}` → {', '.join(f'`{c}`' for c in e['columns'])}"
                     f"（基準類別 `{e['baseline']}`，drop first 避免 dummy variable trap）")
    excluded = [c for c in cols if c["exclude"] and c["name"] != target]
    if excluded:
        L.append("- 排除欄位：" + "、".join(f"`{c['name']}`（{c['exclude']}）" for c in excluded))
    if args.exclude:
        L.append("- 手動排除：" + "、".join(f"`{x}`" for x in args.exclude))
    L.append(f"- 訓練/測試切分：訓練 {len(ytr)} 筆、測試 {len(yte)} 筆（test ratio {args.test_ratio}、"
             f"seed {args.seed}，mulberry32 洗牌，與網頁版相同）")
    L.append("- 模型直接使用原始單位（不標準化），另以標準化係數比較各特徵影響力\n")

    L.append("## 4. Modeling 建模\n")
    L.append("### 4.1 全特徵模型（OLS）\n")
    L.append(f"`{equation(full, target)}`\n")
    L.append(coef_table(full, args.sl, vifs, std_full) + "\n")
    L.append(f"R² = **{full['r2']:.4f}**、Adj R² = **{full['adj_r2']:.4f}**、F = {fmt(full['F'])}"
             f"（p = {fmt_p(full['F_p'])}）、殘差標準誤 σ = {fmt(full['sigma'])}、AIC = {fmt(full['aic'])}、"
             f"BIC = {fmt(full['bic'])}；顯著標記：*** p<0.001、** p<0.01、* p<0.05、. p<0.1\n")
    if full["dropped_collinear"]:
        L.append(f"> ⚠️ 完全共線而移除：{', '.join(full['dropped_collinear'])}\n")

    L.append(f"### 4.2 逆向淘汰（Backward Elimination，SL = {args.sl}）\n")
    if steps:
        L.append(md_table(["步驟", "移除變數", "p 值", "移除前 R²", "移除前 Adj R²"],
                          [[str(i + 1), f"`{s['removed']}`", fmt_p(s["p"]), f"{s['r2']:.4f}", f"{s['adj_r2']:.4f}"]
                           for i, s in enumerate(steps)]) + "\n")
    else:
        L.append("所有特徵的 p 值都小於 SL，不需移除任何特徵。\n")
    L.append(f"保留特徵：{', '.join(f'`{f}`' for f in reduced['features']) or '（只剩截距）'}\n")
    L.append(f"`{equation(reduced, target)}`\n")
    L.append(coef_table(reduced, args.sl) + "\n")

    L.append("## 5. Evaluation 評估\n")
    rows = []
    models = dict(full=full, reduced=reduced, sweet=sweet)
    for key, label in (("full", "全特徵模型"), ("reduced", f"逆向淘汰模型（SL = {args.sl}）"),
                       ("sweet", f"最佳 k 模型（k = {sel['best']['k']}）")):
        tr, te = ev[key]["train"], ev[key]["test"]
        rows.append([label, str(len(models[key]["features"])), f"{tr['r2']:.4f}",
                     f"{tr['adj_r2']:.4f}", f"{te['r2']:.4f}", fmt(te["rmse"]), fmt(te["mae"])])
    L.append(md_table(["模型", "特徵數", "Train R²", "Train Adj R²", "Test R²", "Test RMSE", "Test MAE"], rows) + "\n")
    L.append("![Actual vs Predicted](figures/actual_vs_predicted.png)\n")
    L.append("![Residuals](figures/residuals.png)\n")
    L.append("![Standardized coefficients](figures/standardized_coefficients.png)\n")

    b = sel["best"]
    L.append("### 5.1 特徵選擇比較（依特徵數 k）\n")
    L.append("六種方法各自決定「用 k 個特徵時選哪些」，每個組合都在訓練集用 OLS 配適、在測試集評估：\n")
    header = ["k"] + [m["label"] for m in sel["methods"]]
    trows = []
    for k in range(1, sel["K"] + 1):
        cells = [str(k)]
        for m in sel["methods"]:
            res = m["results"][k]
            v = fmt(res["test"]["rmse"])
            cells.append(f"**{v}**" if res["cols"] == b["cols"] else v)
        trows.append(cells)
    L.append("**Test RMSE**（粗體 = 最佳組合）\n")
    L.append(md_table(header, trows) + "\n")
    ref = next((m for m in sel["methods"] if m["key"] == "best"), sel["methods"][0])
    L.append(f"**{ref['label']} 每個 k 選出的特徵**\n")
    L.append(md_table(["k", "特徵", "Train Adj R²", "Test R²", "Test RMSE"],
                      [[str(k), ", ".join(f"`{f}`" for f in res["features"]), f"{res['train']['adj_r2']:.4f}",
                        f"{res['test']['r2']:.4f}", fmt(res["test"]["rmse"])]
                       for k, res in sorted(ref["results"].items())]) + "\n")
    L.append(f"最佳組合（測試 RMSE 最低）：**k = {b['k']}**，特徵 {', '.join(f'`{f}`' for f in b['features'])}，"
             f"Test RMSE = {fmt(b['test']['rmse'])}、Test R² = {b['test']['r2']:.4f}"
             f"（選出此組合的方法：{'、'.join(b['same_methods'])}）。\n")
    L.append(f"`{equation(sweet, target)}`\n")
    L.append("![Feature selection](figures/feature_selection.png)\n")
    L.append("> 註：依作業範例以測試集比較 k；實務上建議用交叉驗證（cross-validation）選 k，避免對測試集過度調整。\n")

    sig = [c["name"] for c in full["coefficients"][1:] if c["p"] < args.sl]
    top = sorted(zip(full["features"], std_full), key=lambda t: -abs(t[1]))[:3]
    high_vif = [f"`{n}`（{v:.1f}）" for n, v in zip(full["features"], vifs) if v > 10]
    best = "reduced" if ev["reduced"]["test"]["rmse"] < ev["full"]["test"]["rmse"] else "full"
    L.append("**結論**\n")
    L.append(f"- 全特徵模型解釋了訓練資料 {full['r2'] * 100:.1f}% 的變異；測試集 R² = {ev['full']['test']['r2']:.4f}、"
             f"RMSE = {fmt(ev['full']['test']['rmse'])}。")
    L.append(f"- 在 α = {args.sl} 下顯著的特徵：{', '.join(f'`{s}`' for s in sig) or '無'}。")
    L.append("- 影響力最大（|標準化係數|）的特徵：" +
             "、".join(f"`{n}`（{'正' if v > 0 else '負'}向，{v:+.3f}）" for n, v in top) + "。")
    L.append(f"- 逆向淘汰後保留 {len(reduced['features'])} 個特徵，測試集 R² 從 {ev['full']['test']['r2']:.4f} "
             f"變為 {ev['reduced']['test']['r2']:.4f}；以測試集 RMSE 來看"
             f"{'逆向淘汰模型' if best == 'reduced' else '全特徵模型'}較佳。")
    L.append(f"- 特徵選擇比較：k = {b['k']} 時測試 RMSE 最低（{fmt(b['test']['rmse'])}，R² = {b['test']['r2']:.4f}），"
             f"特徵為 {', '.join(f'`{f}`' for f in b['features'])}"
             + (f"，比全特徵模型更精簡（{len(full['features'])} → {b['k']} 個特徵）。" if b["k"] < len(full["features"]) else "。"))
    L.append("- 共線性：" + ("、".join(high_vif) + " 的 VIF > 10，有明顯共線性。" if high_vif else "所有特徵 VIF < 10，共線性不嚴重。") + "\n")

    L.append("## 6. Deployment 部署\n")
    site = f"[互動式網頁（GitHub Pages）]({args.site})" if args.site else "互動式網頁（GitHub Pages）"
    L.append(f"- {site}：下拉選單切換 Problem、可輸入任意 CSV 網址；同樣的切分與參數會得到與本報告相同的數字。")
    L.append(f"- 最佳 k 模型（k = {b['k']}）：`{equation(sweet, target)}`")
    L.append(f"- 逆向淘汰模型（SL = {args.sl}）：`{equation(reduced, target)}`\n")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")

    def model_json(m):
        return {k: v for k, v in m.items() if k not in ("beta",)}

    (out / "results.json").write_text(json.dumps(dict(
        name=args.name, url=args.url, target=target, features=features, names=names,
        config=dict(test_ratio=args.test_ratio, seed=args.seed, sl=args.sl),
        n_rows=len(df), n_used=len(y), n_dropped=d["n_dropped"], train_idx=train_idx, test_idx=test_idx,
        full=model_json(full), reduced=model_json(reduced), sweet=model_json(sweet), steps=steps, vif=vifs,
        std_coef=std_full, evaluation=ev,
        selection=dict(K=sel["K"], best=sel["best"],
                       methods={m["key"]: {str(k): dict(features=res["features"], test_rmse=res["test"]["rmse"],
                                                        test_r2=res["test"]["r2"])
                                           for k, res in sorted(m["results"].items())} for m in sel["methods"]})),
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(f"[mlr-crisp-dm] {args.name}: n={len(y)} (train {len(ytr)} / test {len(yte)})")
    print(f"  full    : {len(full['features'])} features, test R2={ev['full']['test']['r2']:.4f}, "
          f"RMSE={ev['full']['test']['rmse']:.4f}")
    print(f"  reduced : {reduced['features']}, test R2={ev['reduced']['test']['r2']:.4f}, "
          f"RMSE={ev['reduced']['test']['rmse']:.4f}")
    print(f"  report  : {out / 'report.md'}")


if __name__ == "__main__":
    main()
