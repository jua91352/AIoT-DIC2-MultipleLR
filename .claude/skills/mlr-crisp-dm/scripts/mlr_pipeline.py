#!/usr/bin/env python3
"""CRISP-DM 多元線性回歸 × 5 種特徵選擇方法（Multiple Linear Regression pipeline）

用法：
    python mlr_pipeline.py --url <CSV 網址或路徑> --target <目標欄位> \
        --name "<資料集名稱>" --slug <輸出資料夾名稱> [--business "<商業問題>"]

輸出（reports/<slug>/）：
    report.md      CRISP-DM 六步驟分析報告
    results.json   5 種方法在每個 k 的特徵與指標、LassoCV、全特徵 OLS 檢定
    figures/*.png  5 種方法的 RMSE / R² 雙軌跡折線圖、相關係數熱圖、實際 vs 預測、殘差圖

5 種方法（每個方法對 k = 1 ~ K 各選出 k 個特徵，再用 OLS 在訓練集重新配適、於測試集評估）：
    1. Pearson（Filter）         訓練集 |r(x, y)| 由大到小
    2. SelectKBest（Filter）     f_regression 的 F 值由大到小（F 是 |r| 的單調函數，排序與 Pearson 相同）
    3. RFE（Wrapper）            RFE(LinearRegression) 於標準化特徵，每輪移除 |係數| 最小者
    4. Forward SFS（Wrapper）    SequentialFeatureSelector(forward, cv=KFold(5, shuffle=True))
    5. Lasso（Embedded）         LassoCV(cv=KFold(5, shuffle=True)) 收縮後的 |係數| 由大到小

切分使用 scikit-learn train_test_split(test_size, random_state)，與網頁版 assets/js/mlr.js 逐筆相同，
兩邊的每一個數字都可以互相對照。
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
from sklearn.feature_selection import RFE, SequentialFeatureSelector, f_regression  # noqa: E402
from sklearn.linear_model import LassoCV, LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-"}
NUM_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
ID_NAME_RE = re.compile(r"(id|index|unnamed: ?\d+)", re.IGNORECASE)

METHODS = [
    ("pearson", "Pearson", "Filter", "訓練集 |Pearson r(x, y)| 由大到小，取前 k 個"),
    ("kbest", "SelectKBest F-test", "Filter", "`SelectKBest(f_regression)`：F = (n−2)·r²/(1−r²) 最大的 k 個（與 Pearson 排序相同）"),
    ("rfe", "RFE", "Wrapper", "`RFE(LinearRegression)` 於標準化特徵，每輪移除 |係數| 最小的特徵"),
    ("sfs", "Forward SFS", "Wrapper", "`SequentialFeatureSelector(forward, cv=KFold(5, shuffle=True))`，每步加入 CV 平均 R² 最高者"),
    ("lasso", "Lasso", "Embedded", "`LassoCV(cv=KFold(5, shuffle=True))` 收縮後的 |係數| 由大到小（係數為 0 者排最後）"),
]
STYLE = {  # 與網頁相同的顏色 / 線型 / 標記
    "pearson": dict(color="#2a78d6", ls="-", marker="o", fill="full"),
    "kbest": dict(color="#eb6834", ls="--", marker="s", fill="none"),
    "rfe": dict(color="#1baf7a", ls="-.", marker="^", fill="full"),
    "sfs": dict(color="#e34948", ls=":", marker="D", fill="none"),
    "lasso": dict(color="#4a3aa7", ls="-", marker="p", fill="none"),
}
JITTER = 0.06


# ----------------------------------------------------------------- 基本工具

def is_missing(v) -> bool:
    return v is None or str(v).strip().lower() in MISSING


def to_num(v) -> float:
    s = "" if v is None else str(v).strip()
    return float(s) if NUM_RE.fullmatch(s) else math.nan


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
    """與 pandas get_dummies(drop_first=True) 相同：數值欄在前，類別欄的 dummy 接在後面。"""
    info = {c["name"]: c for c in cols}
    ok = df[target].map(lambda v: math.isfinite(to_num(v)))
    for f in features:
        if info[f]["type"] == "numeric":
            ok &= df[f].map(lambda v: math.isfinite(to_num(v)))
        else:
            ok &= ~df[f].map(is_missing)
    used = df[ok].reset_index(drop=True)

    names, blocks, encodings = [], [], []
    ordered = sorted(features, key=lambda f: (info[f]["type"] != "numeric", info[f]["index"]))
    for f in ordered:
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
    resid = y - A @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    df_resid, df_model = n - p, p - 1
    sigma2 = rss / df_resid
    Rinv = np.linalg.inv(R)
    se = np.sqrt(sigma2 * np.diag(Rinv @ Rinv.T))
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), df_resid)
    tcrit = stats.t.ppf(0.975, df_resid)
    r2 = 1 - rss / tss
    F = ((tss - rss) / df_model) / sigma2 if df_model > 0 else math.nan
    coefs = [dict(name="(Intercept)" if j == 0 else names[keep[j - 1]], estimate=float(beta[j]), se=float(se[j]),
                  t=float(t[j]), p=float(pv[j]), lower=float(beta[j] - tcrit * se[j]), upper=float(beta[j] + tcrit * se[j]))
             for j in range(p)]
    return dict(feature_index=keep, features=[names[j] for j in keep], dropped_collinear=dropped, coefficients=coefs,
                beta=beta, n=n, p=p, r2=r2, adj_r2=1 - (1 - r2) * (n - 1) / df_resid, sigma=math.sqrt(sigma2), F=F,
                F_p=float(stats.f.sf(F, df_model, df_resid)) if df_model > 0 else math.nan)


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


# ------------------------------------------------ 特徵選擇：5 種方法 × k

def rankings(Xtr: pd.DataFrame, ytr: pd.Series, kf: KFold) -> tuple[dict, dict, dict]:
    """回傳 (各方法的特徵排序, LassoCV 資訊, 單變數統計)。排序的前 k 個 = 該方法選 k 個特徵的結果。"""
    cols, K = list(Xtr.columns), Xtr.shape[1]
    scaler = StandardScaler().fit(Xtr)
    Z = pd.DataFrame(scaler.transform(Xtr), columns=cols, index=Xtr.index)

    F, Fp = f_regression(Xtr, ytr)
    r = Xtr.corrwith(ytr)
    rfe = RFE(LinearRegression(), n_features_to_select=1).fit(Z, ytr)
    # tol 調小讓座標下降收斂到精確解（預設 tol=1e-4 時相鄰 α 的 CV 誤差幾乎同分，可能選到隔壁的 α）
    lcv = LassoCV(cv=kf, tol=1e-10, max_iter=100_000).fit(Z, ytr)

    order: list[str] = []
    for k in range(1, K):
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward", cv=kf)
        chosen = Xtr.columns[sfs.fit(Xtr, ytr).get_support()]
        order += [c for c in chosen if c not in order]
    order += [c for c in cols if c not in order]

    rank = {
        "pearson": list(r.abs().fillna(0).sort_values(ascending=False, kind="stable").index),
        "kbest": [cols[i] for i in np.argsort(np.nan_to_num(F), kind="mergesort")[::-1]],
        "rfe": [cols[i] for i in np.argsort(rfe.ranking_, kind="stable")],
        "sfs": order,
        "lasso": [cols[i] for i in np.argsort(np.abs(lcv.coef_), kind="stable")[::-1]],
    }
    lasso = dict(model=lcv, scaler=scaler, alpha=float(lcv.alpha_),
                 features=[c for c, b in zip(cols, lcv.coef_) if b != 0], coef_std=lcv.coef_.tolist(),
                 coef_raw=(lcv.coef_ / scaler.scale_).tolist(),
                 intercept_raw=float(lcv.intercept_ - (lcv.coef_ / scaler.scale_) @ scaler.mean_))
    uni = dict(r=r.to_dict(), F=dict(zip(cols, F.tolist())), F_p=dict(zip(cols, Fp.tolist())))
    return rank, lasso, uni


def feature_selection(Xtr, ytr, Xte, yte, names, seed):
    Xtr_df, Xte_df = pd.DataFrame(Xtr, columns=names), pd.DataFrame(Xte, columns=names)
    ytr_s = pd.Series(ytr)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rank, lasso, uni = rankings(Xtr_df, ytr_s, kf)
    K = len(names)
    cache: dict[tuple, dict] = {}

    def evaluate(feats):
        key = tuple(sorted(names.index(f) for f in feats))
        if key not in cache:
            m = fit_ols(Xtr, ytr, names, list(key))
            cv = cross_val_score(LinearRegression(), Xtr_df[list(feats)], ytr_s, cv=kf, scoring="neg_root_mean_squared_error")
            cache[key] = dict(cols=list(key), model=m, train=metrics(ytr, predict(m, Xtr), len(key)),
                              test=metrics(yte, predict(m, Xte), len(key)), cv_rmse=float(-cv.mean()))
        return cache[key]

    methods = []
    for key, label, family, rule in METHODS:
        results = {}
        for k in range(1, K + 1):
            res = evaluate(rank[key][:k])
            results[k] = {**res, "order": rank[key][:k], "added": rank[key][k - 1]}
        methods.append(dict(key=key, label=label, family=family, rule=rule, ranking=rank[key], results=results))

    best = None
    for m in methods:
        for k in range(1, K + 1):
            s = m["results"][k]["test"]["rmse"]
            if best is None or s < best["score"] * (1 - 1e-12) or (abs(s - best["score"]) <= 1e-12 * best["score"] and k < best["k"]):
                best = dict(score=s, k=k, method=m["key"], res=m["results"][k])
    best["same_methods"] = [m["label"] for m in methods if m["results"][best["k"]]["cols"] == best["res"]["cols"]]

    p = lasso["model"].predict(pd.DataFrame(lasso["scaler"].transform(Xte_df), columns=names))
    ptr = lasso["model"].predict(pd.DataFrame(lasso["scaler"].transform(Xtr_df), columns=names))
    lasso.update(k=len(lasso["features"]), test=metrics(yte, p, len(lasso["features"])),
                 train=metrics(ytr, ptr, len(lasso["features"])))
    return dict(K=K, methods=methods, best=best, lasso=lasso, uni=uni)


# -------------------------------------------------------------------- 圖表

def plot_trajectory(sel: dict, path: Path, basis: str = "test"):
    """1×2 雙軌跡折線圖：RMSE / R² vs 特徵數 k（5 種顏色 + 專屬線型與標記 + 水平微偏移 + LassoCV ★）。"""
    K, ks = sel["K"], np.arange(1, sel["K"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    b, lcv = sel["best"], sel["lasso"]
    for ax, metric, title in [(axes[0], "rmse", "RMSE"), (axes[1], "r2", "R-squared")]:
        for i, m in enumerate(sel["methods"]):
            st = STYLE[m["key"]]
            vals = [m["results"][k][basis][metric] for k in ks]
            ax.plot(ks + (i - 2) * JITTER, vals, ls=st["ls"], marker=st["marker"], fillstyle=st["fill"], color=st["color"],
                    lw=2, ms=8, markerfacecolor="white" if st["fill"] == "none" else st["color"],
                    label=f"{i + 1}. {m['label']} ({m['family']})")
        ax.scatter([lcv["k"]], [lcv[basis][metric]], marker="*", s=320, color="#f2b01e", edgecolor="black", zorder=6,
                   label=f"LassoCV native shrinkage (k={lcv['k']})")
        bi = [m["key"] for m in sel["methods"]].index(b["method"])
        ax.scatter([b["k"] + (bi - 2) * JITTER], [b["res"][basis][metric]], s=420, facecolor="none", edgecolor="#f2b01e",
                   lw=2, zorder=5, label=f"Best: lowest test RMSE (k={b['k']})")
        ax.set_xticks(ks)
        ax.set_xlim(0.5, K + 0.5)
        ax.set_title(f"{title} by Number of Features", fontweight="bold")
        ax.set_xlabel("Number of Features (k)")
        ax.set_ylabel(f"Test {title}" if basis == "test" else title)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5, frameon=False, markerscale=0.8)
    fig.suptitle("Comparison of 5 Feature Selection Schemes (k = 1 to %d)" % K, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


def plot_actual_vs_pred(ytr, ptr, yte, pte, target, title, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(ytr, ptr, s=22, alpha=0.6, color="#8a93a8", label=f"Train ({len(ytr)})")
    ax.scatter(yte, pte, s=36, alpha=0.95, color="#4f46e5", label=f"Test ({len(yte)})")
    allv = np.concatenate([ytr, ptr, yte, pte])
    lo, hi = allv.min(), allv.max()
    ax.plot([lo, hi], [lo, hi], ls="--", color="#868e96", lw=1, label="y = x")
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(ptr, rtr, pte, rte, title, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(ptr, rtr, s=22, alpha=0.6, color="#8a93a8", label="Train")
    ax.scatter(pte, rte, s=36, alpha=0.95, color="#4f46e5", label="Test")
    ax.axhline(0, ls="--", color="#868e96", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
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


def coef_table(model: dict, vifs=None) -> str:
    header = ["變數", "係數", "標準誤", "t", "p 值", "95% 信賴區間"] + (["VIF"] if vifs is not None else [])
    rows = []
    for k, c in enumerate(model["coefficients"]):
        name = f"**{c['name']}**" if k > 0 and c["p"] < 0.05 else c["name"]
        row = [name, fmt(c["estimate"]), fmt(c["se"]), f"{c['t']:.3f}", f"{fmt_p(c['p'])} {stars(c['p'])}".strip(),
               f"[{fmt(c['lower'])}, {fmt(c['upper'])}]"]
        if vifs is not None:
            row.append("—" if k == 0 else fmt(vifs[k - 1], 4))
        rows.append(row)
    return md_table(header, rows)


def main():
    ap = argparse.ArgumentParser(description="CRISP-DM multiple linear regression × 5 feature selection schemes")
    ap.add_argument("--url", required=True, help="CSV 網址或本機路徑")
    ap.add_argument("--target", help="目標欄位（預設：最後一個數值欄位）")
    ap.add_argument("--name", default="Dataset", help="資料集名稱（報告標題）")
    ap.add_argument("--slug", default="dataset", help="輸出資料夾名稱 reports/<slug>")
    ap.add_argument("--business", default="", help="商業理解：要解決的問題（一段文字）")
    ap.add_argument("--exclude", nargs="*", default=[], help="不放入模型的欄位")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42, help="train_test_split 與 KFold 的 random_state")
    ap.add_argument("--sep", default=",", help="分隔符號（預設逗號）")
    ap.add_argument("--out-root", default="reports")
    ap.add_argument("--site", default="", help="互動式網頁網址（寫進報告的部署段落）")
    args = ap.parse_args()

    out = Path(args.out_root) / args.slug
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for old in figs.glob("*.png"):
        old.unlink()

    # 2. Data Understanding
    df = load_table(args.url, args.sep)
    cols = infer_columns(df)
    numeric = [c for c in cols if c["type"] == "numeric" and not c["exclude"]]
    target = args.target or numeric[-1]["name"]
    if target not in df.columns:
        raise SystemExit(f"找不到目標欄位 {target}；可用欄位：{list(df.columns)}")
    features = [c["name"] for c in cols if c["name"] != target and not c["exclude"]
                and c["type"] != "empty" and c["name"] not in args.exclude]

    # 3. Data Preparation：One-hot、train_test_split（與網頁相同）
    d = build_design(df, cols, target, features)
    X, y, names = d["X"], d["y"], d["names"]
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_ratio, random_state=args.seed)
    Xtr, ytr, Xte, yte = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    # 4. Modeling：全特徵 OLS + 5 種特徵選擇方法
    full = fit_ols(Xtr, ytr, names)
    if full["dropped_collinear"]:
        keep = full["feature_index"]
        X, Xtr, Xte, names = X[:, keep], Xtr[:, keep], Xte[:, keep], [names[j] for j in keep]
        full = fit_ols(Xtr, ytr, names)
    sk = LinearRegression().fit(Xtr, ytr)
    assert np.allclose(sk.coef_, full["beta"][1:], rtol=1e-6, atol=1e-8), "OLS 係數與 scikit-learn 不一致"
    vifs = vif(Xtr, names, full["feature_index"])
    sel = feature_selection(Xtr, ytr, Xte, yte, names, args.seed)
    b, K = sel["best"], sel["K"]
    bm = b["res"]["model"]
    full_res = sel["methods"][0]["results"][K]

    # 5. Evaluation：圖表
    numeric_used = [f for f in features if next(c for c in cols if c["name"] == f)["type"] == "numeric"] + [target]
    corr = d["rows"][numeric_used].apply(lambda s: s.map(to_num)).corr()
    plot_corr(corr, figs / "correlation.png")
    plot_trajectory(sel, figs / "feature_selection_5_schemes.png")
    ptr, pte = predict(bm, Xtr), predict(bm, Xte)
    best_title = f"Best model: k = {b['k']} ({', '.join(b['res']['order'])})"
    if len(best_title) > 60:
        best_title = f"Best model: k = {b['k']}"
    plot_actual_vs_pred(ytr, ptr, yte, pte, target, best_title, figs / "actual_vs_predicted.png")
    plot_residuals(ptr, ytr - ptr, pte, yte - pte, f"Residuals (best model, k = {b['k']})", figs / "residuals.png")

    # ------------------------------------------------------------ report.md
    code = lambda xs: ", ".join(f"`{x}`" for x in xs)  # noqa: E731
    L = []
    L.append(f"# {args.name}：CRISP-DM × 5 種特徵選擇方法報告\n")
    L.append(f"> 由 `mlr-crisp-dm` skill 產生 · 資料來源：<{args.url}> · {datetime.now():%Y-%m-%d %H:%M}\n")

    L.append("## 1. Business Understanding 商業理解\n")
    L.append((args.business or f"以多元線性回歸預測 **{target}**，並找出最精簡又準確的特徵組合。") + "\n")
    L.append(f"- 目標變數 y：`{target}`；候選特徵：{code(features)}")
    L.append(f"- 比較 5 種特徵選擇方法在 **k = 1 ~ {K}** 的 Test RMSE（越低越好）與 Test R²（越高越好），"
             "表現相同時選特徵較少的模型。\n")

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
    uni = sel["uni"]
    order_r = sorted(names, key=lambda f: -abs(uni["r"][f]) if math.isfinite(uni["r"][f]) else 0)
    L.append("**訓練集單變數統計（兩種 Filter 方法的依據）**\n")
    L.append(md_table(["特徵", "Pearson r", "F（f_regression）", "p 值"],
                      [[f"`{f}`", f"{uni['r'][f]:+.4f}", fmt(uni["F"][f], 5), fmt_p(uni["F_p"][f])] for f in order_r]) + "\n")
    L.append("F = (n − 2)·r² / (1 − r²) 是 |r| 的單調函數，所以 Pearson 與 SelectKBest 的排序完全相同，兩條線重疊是正確結果。\n")
    L.append("![Correlation](figures/correlation.png)\n")

    L.append("## 3. Data Preparation 資料準備\n")
    L.append(f"- 缺失值：{'刪除 ' + str(d['n_dropped']) + ' 筆含缺失值的資料' if d['n_dropped'] else '無缺失值，不需處理'}")
    for e in d["encodings"]:
        if e["type"] == "categorical":
            L.append(f"- One-hot（drop_first）：`{e['feature']}` → {code(e['columns'])}（基準類別 `{e['baseline']}`，"
                     "避免 dummy variable trap）")
    excluded = [c for c in cols if c["exclude"] and c["name"] != target]
    if excluded:
        L.append("- 排除欄位：" + "、".join(f"`{c['name']}`（{c['exclude']}）" for c in excluded))
    if args.exclude:
        L.append("- 手動排除：" + "、".join(f"`{x}`" for x in args.exclude))
    L.append(f"- 切分：`train_test_split(test_size={args.test_ratio}, random_state={args.seed})` → 訓練 {len(ytr)} 筆、"
             f"測試 {len(yte)} 筆（網頁以 numpy MT19937 重現同一個排列，逐筆相同）")
    L.append("- 標準化：`StandardScaler` 只用訓練集 fit，供 RFE 與 Lasso 比較係數大小；選出的特徵一律用 OLS 在原始單位重新配適")
    L.append(f"- 交叉驗證：`KFold(n_splits=5, shuffle=True, random_state={args.seed})`，供 Forward SFS、LassoCV 與 CV RMSE 共用\n")

    L.append("## 4. Modeling 建模\n")
    L.append("### 4.1 五種特徵選擇方法\n")
    L.append(md_table(["#", "方法", "類型", "規則", "特徵排序（前 k 個 = 選 k 個特徵）"],
                      [[str(i + 1), m["label"], m["family"], m["rule"], " › ".join(m["ranking"])]
                       for i, m in enumerate(sel["methods"])]) + "\n")
    lc = sel["lasso"]
    L.append(f"LassoCV 以 5 折 CV 選出 α = {fmt(lc['alpha'], 6)}，保留 {lc['k']} 個非 0 係數：{code(lc['features'])}。\n")
    L.append("### 4.2 全特徵模型（OLS，訓練集）\n")
    L.append(f"`{equation(full, target)}`\n")
    L.append(coef_table(full, vifs) + "\n")
    L.append(f"R² = **{full['r2']:.4f}**、Adj R² = **{full['adj_r2']:.4f}**、F = {fmt(full['F'])}（p = {fmt_p(full['F_p'])}）、"
             f"殘差標準誤 σ = {fmt(full['sigma'])}；粗體 = p < 0.05（*** p<0.001、** p<0.01、* p<0.05、. p<0.1）\n")

    L.append("## 5. Evaluation 評估\n")
    L.append("![5 feature selection schemes](figures/feature_selection_5_schemes.png)\n")
    digits = 2 if abs(full_res["test"]["rmse"]) >= 100 else 4  # 誤差指標的小數位數（與網頁相同）
    fm = lambda v: f"{v:,.{digits}f}"  # noqa: E731
    for metric, title, f in (("rmse", "Test RMSE", fm), ("r2", "Test R²", lambda v: f"{v:.4f}")):
        rows = []
        for k in range(1, K + 1):
            cells = [str(k)]
            for m in sel["methods"]:
                res = m["results"][k]
                v = f(res["test"][metric])
                cells.append(f"**{v}**" if res["cols"] == b["res"]["cols"] else v)
            rows.append(cells)
        L.append(f"**{title}**（粗體 = 最佳組合）\n")
        L.append(md_table(["k"] + [m["label"] for m in sel["methods"]], rows) + "\n")
    L.append("**每個 k 新增的特徵**（前 k 列累積起來就是選出的 k 個特徵）\n")
    L.append(md_table(["k"] + [m["label"] for m in sel["methods"]],
                      [[str(k)] + [f"+{m['results'][k]['added']}" for m in sel["methods"]] for k in range(1, K + 1)]) + "\n")
    cv_rows = [[str(k)] + [fm(m["results"][k]["cv_rmse"]) for m in sel["methods"]] for k in range(1, K + 1)]
    L.append("**5-fold CV RMSE（訓練集內）**\n")
    L.append(md_table(["k"] + [m["label"] for m in sel["methods"]], cv_rows) + "\n")
    cv_best = min(((m["results"][k]["cv_rmse"], k, m["results"][k]["cols"]) for m in sel["methods"] for k in range(1, K + 1)),
                  key=lambda t: (t[0], t[1]))
    cv_methods = "、".join(m["label"] for m in sel["methods"] if m["results"][cv_best[1]]["cols"] == cv_best[2])

    L.append("**結論**\n")
    L.append(f"- 最佳組合（Test RMSE 最低）：**k = {b['k']}**，{code(b['res']['order'])}，Test RMSE = **{fm(b['res']['test']['rmse'])}**、"
             f"R² = **{b['res']['test']['r2']:.4f}**（選出這組特徵的方法：{'、'.join(b['same_methods'])}）。")
    L.append(f"- 全特徵模型（k = {K}）：Test RMSE {fm(full_res['test']['rmse'])}、R² {full_res['test']['r2']:.4f}"
             + (f" → 最佳組合的 RMSE 降低 {(1 - b['res']['test']['rmse'] / full_res['test']['rmse']) * 100:.1f}%。"
                if b["k"] < K else "。"))
    L.append(f"- LassoCV 原生收縮模型（★，k = {lc['k']}）：Test RMSE {fm(lc['test']['rmse'])}、R² {lc['test']['r2']:.4f}。")
    L.append(f"- 以訓練集 5-fold CV 評估時，RMSE 最低在 k = {cv_best[1]}（{cv_methods}，{fm(cv_best[0])}）；"
             f"測試集只有 {len(yte)} 筆，選 k 時宜同時參考 CV。")
    high_vif = [f"`{n}`（{v:.1f}）" for n, v in zip(full["features"], vifs) if v > 10]
    L.append("- 共線性：" + ("、".join(high_vif) + " 的 VIF > 10，有明顯共線性。" if high_vif else "所有特徵 VIF < 10，共線性不嚴重。") + "\n")
    L.append(f"**最佳模型（k = {b['k']}）**：`{equation(bm, target)}`\n")
    L.append(coef_table(bm) + "\n")
    L.append("![Actual vs Predicted](figures/actual_vs_predicted.png)\n")
    L.append("![Residuals](figures/residuals.png)\n")

    L.append("## 6. Deployment 部署\n")
    site = f"[互動式網頁（GitHub Pages）]({args.site})" if args.site else "互動式網頁（GitHub Pages）"
    L.append(f"- {site}：下拉選單切換 Problem、可輸入任意 CSV 網址；同樣的切分與參數會得到與本報告相同的數字，"
             "並可在「模型檢視」用滑桿即時預測。")
    L.append(f"- 最佳模型：`{equation(bm, target)}`\n")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")

    def res_json(res):
        return dict(features=res["order"], added=res["added"], train_r2=res["train"]["r2"], test_rmse=res["test"]["rmse"],
                    test_r2=res["test"]["r2"], test_mae=res["test"]["mae"], cv_rmse=res["cv_rmse"])

    (out / "results.json").write_text(json.dumps(dict(
        name=args.name, url=args.url, target=target, features=features, names=names,
        config=dict(test_ratio=args.test_ratio, seed=args.seed, cv="KFold(n_splits=5, shuffle=True, random_state=seed)"),
        n_rows=len(df), n_used=len(y), n_dropped=d["n_dropped"], train_idx=train_idx.tolist(), test_idx=test_idx.tolist(),
        full={k: v for k, v in full.items() if k != "beta"}, vif=vifs,
        selection=dict(K=K, best=dict(k=b["k"], method=b["method"], same_methods=b["same_methods"], **res_json(b["res"])),
                       methods={m["key"]: dict(label=m["label"], family=m["family"], ranking=m["ranking"],
                                               results={str(k): res_json(r) for k, r in m["results"].items()})
                                for m in sel["methods"]},
                       lasso_cv={k: lc[k] for k in ("alpha", "k", "features", "coef_raw", "intercept_raw", "train", "test")}),
    ), ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(f"[mlr-crisp-dm] {args.name}: n={len(y)} (train {len(ytr)} / test {len(yte)}), K={K}")
    for m in sel["methods"]:
        print(f"  {m['label']:<20}" + "  ".join(f"{m['results'][k]['test']['rmse']:.4f}" for k in range(1, K + 1)))
    print(f"  best    : k={b['k']} {b['res']['order']} RMSE={b['res']['test']['rmse']:.4f} R2={b['res']['test']['r2']:.4f}")
    print(f"  LassoCV : alpha={lc['alpha']:.6g} k={lc['k']} RMSE={lc['test']['rmse']:.4f} R2={lc['test']['r2']:.4f}")
    print(f"  report  : {out / 'report.md'}")


if __name__ == "__main__":
    main()
