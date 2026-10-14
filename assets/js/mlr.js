/*!
 * mlr.js — 多元線性回歸（OLS）計算引擎，無外部相依套件
 * 瀏覽器：window.MLR；Node：require('./mlr.js')
 *
 * 資料切分（mulberry32 + Fisher–Yates）、型態判斷、One-hot、OLS 檢定、VIF 與逆向淘汰
 * 都和 .claude/skills/mlr-crisp-dm/scripts/mlr_pipeline.py 相同，兩邊結果可以互相對照。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.MLR = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ============================== 基本工具 ============================== */

  const MISSING = new Set(['', 'na', 'n/a', 'nan', 'null', 'none', '?', '-']);
  const NUM_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
  const ID_NAME_RE = /^(id|index|unnamed: ?\d+)$/i;

  function isMissing(v) {
    return v == null || MISSING.has(String(v).trim().toLowerCase());
  }

  function toNumber(v) {
    if (v == null) return NaN;
    const s = String(v).trim();
    return NUM_RE.test(s) ? Number(s) : NaN;
  }

  function mean(a) {
    let s = 0;
    for (const v of a) s += v;
    return s / a.length;
  }

  // 樣本標準差（ddof = 1，與 pandas 相同）
  function std(a) {
    const m = mean(a);
    let s = 0;
    for (const v of a) s += (v - m) * (v - m);
    return Math.sqrt(s / (a.length - 1));
  }

  function minMax(a) {
    let lo = Infinity, hi = -Infinity;
    for (const v of a) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return [lo, hi];
  }

  // 線性內插分位數（與 pandas 預設相同）
  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  function describe(values) {
    const s = values.slice().sort((a, b) => a - b);
    return {
      count: s.length,
      mean: mean(s),
      std: s.length > 1 ? std(s) : NaN,
      min: s[0],
      q1: quantile(s, 0.25),
      median: quantile(s, 0.5),
      q3: quantile(s, 0.75),
      max: s[s.length - 1],
    };
  }

  function pearson(a, b) {
    const ma = mean(a), mb = mean(b);
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < a.length; i++) {
      const da = a[i] - ma, db = b[i] - mb;
      sab += da * db;
      saa += da * da;
      sbb += db * db;
    }
    return sab / Math.sqrt(saa * sbb);
  }

  /* ============================ 亂數與資料切分 ============================ */

  // mulberry32：可重現的 32-bit 亂數產生器（Python 版有逐位元相同的實作）
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffledIndices(n, seed) {
    const rand = mulberry32(seed);
    const idx = [];
    for (let i = 0; i < n; i++) idx.push(i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = idx[i];
      idx[i] = idx[j];
      idx[j] = t;
    }
    return idx;
  }

  // 洗牌後前 round(n × testRatio) 筆為測試集，其餘為訓練集
  function trainTestSplit(n, testRatio, seed) {
    const idx = shuffledIndices(n, seed);
    const nTest = Math.round(n * testRatio);
    return { test: idx.slice(0, nTest), train: idx.slice(nTest) };
  }

  /* =============================== CSV 解析 =============================== */

  // RFC 4180 風格：支援引號、引號內的分隔符號與換行、"" 跳脫
  function splitDelimited(text, delim) {
    const rows = [];
    let row = [], field = '', inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inQuotes) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; }
          else inQuotes = false;
        } else field += c;
      } else if (c === '"') inQuotes = true;
      else if (c === delim) { row.push(field); field = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        row.push(field);
        rows.push(row);
        row = [];
        field = '';
      } else field += c;
    }
    if (field !== '' || row.length) { row.push(field); rows.push(row); }
    return rows.map(r => r.map(s => s.trim())).filter(r => r.some(s => s !== ''));
  }

  // 依前幾列判斷分隔符號；都不像時回傳 null（改用連續空白分隔）
  function detectDelimiter(lines) {
    let best = null, bestScore = 0;
    for (const d of [',', ';', '\t', '|']) {
      const counts = lines.map(l => l.split(d).length - 1);
      if (counts[0] === 0) continue;
      const consistency = counts.filter(c => c === counts[0]).length / counts.length;
      const score = consistency * 1000 + counts[0];
      if (score > bestScore) { bestScore = score; best = d; }
    }
    return best;
  }

  function dedupeNames(names) {
    const seen = new Map();
    return names.map(n => {
      const k = seen.get(n) || 0;
      seen.set(n, k + 1);
      return k ? `${n}_${k + 1}` : n;
    });
  }

  function parseCSV(text) {
    text = String(text).replace(/^﻿/, '');
    const sample = text.split(/\r?\n/).filter(l => l.trim() !== '').slice(0, 30);
    if (!sample.length) throw new Error('檔案是空的。');
    const delim = detectDelimiter(sample);
    const rows = delim
      ? splitDelimited(text, delim)
      : text.split(/\r?\n/).map(l => l.trim()).filter(Boolean).map(l => l.split(/\s+/));

    // 第一列全部是數字 → 視為沒有標頭，自動命名 col_1, col_2, …
    const first = rows[0];
    const hasHeader = !first.every(v => Number.isFinite(toNumber(v)));
    const header = dedupeNames(first.map((h, i) => (hasHeader && h) ? h : `col_${i + 1}`));
    const width = header.length;
    const body = (hasHeader ? rows.slice(1) : rows).map(r => {
      const out = r.slice(0, width);
      while (out.length < width) out.push('');
      return out;
    });
    return { header, rows: body, delimiter: delim || 'whitespace', hasHeader };
  }

  /* ============================ 欄位型態判斷 ============================ */

  function inferColumns(table) {
    const n = table.rows.length;
    return table.header.map((name, j) => {
      const present = [];
      for (const r of table.rows) if (!isMissing(r[j])) present.push(String(r[j]).trim());
      const nums = present.map(toNumber);
      const numeric = present.length > 0 && nums.every(Number.isFinite);
      const unique = new Set(present).size;
      const col = {
        name, index: j,
        type: numeric ? 'numeric' : 'categorical',
        missing: n - present.length,
        unique,
        exclude: null,
      };
      if (!present.length) {
        col.type = 'empty';
        col.exclude = '全部為空值';
      } else if (unique === 1) {
        col.exclude = '常數欄位（只有一種值）';
      } else if (numeric) {
        const [lo, hi] = minMax(nums);
        const allInt = nums.every(Number.isInteger);
        if (unique === present.length && (ID_NAME_RE.test(name) || (allInt && hi - lo === present.length - 1))) {
          col.exclude = '疑似 ID / 流水號';
        }
      } else if (unique > 30 || (present.length > 10 && unique / present.length > 0.5)) {
        col.exclude = '類別過多（疑似 ID 或文字欄）';
      }
      return col;
    });
  }

  /* ===================== 資料準備：缺失值與 One-hot ===================== */

  function buildDesign(table, columns, target, features) {
    const byName = new Map(columns.map(c => [c.name, c]));
    const tc = byName.get(target);
    if (!tc) throw new Error(`找不到目標欄位「${target}」。`);
    const fcs = features.map(f => {
      const c = byName.get(f);
      if (!c) throw new Error(`找不到特徵欄位「${f}」。`);
      return c;
    });

    // 刪除目標或任一特徵缺失（或數值欄無法轉成數字）的資料列
    const used = [], rowIndex = [];
    let dropped = 0;
    table.rows.forEach((r, i) => {
      let ok = Number.isFinite(toNumber(r[tc.index]));
      for (const c of fcs) {
        if (!ok) break;
        const v = r[c.index];
        ok = c.type === 'numeric' ? Number.isFinite(toNumber(v)) : !isMissing(v);
      }
      if (ok) { used.push(r); rowIndex.push(i); }
      else dropped++;
    });

    // 類別欄位 One-hot（drop first）：依字母排序第一個類別當基準，避免 dummy variable trap
    const encodings = [], names = [];
    for (const c of fcs) {
      if (c.type === 'numeric') {
        encodings.push({ feature: c.name, type: 'numeric', index: c.index, columns: [c.name] });
        names.push(c.name);
      } else {
        const levels = [...new Set(used.map(r => String(r[c.index]).trim()))].sort();
        const dummies = levels.slice(1).map(l => `${c.name}_${l}`);
        encodings.push({ feature: c.name, type: 'categorical', index: c.index, levels, baseline: levels[0], columns: dummies });
        names.push(...dummies);
      }
    }

    const X = used.map(r => encodeRaw(r, encodings));
    const y = used.map(r => toNumber(r[tc.index]));
    return { X, y, names, encodings, rows: used, rowIndex, nDropped: dropped };
  }

  function encodeRaw(r, encodings) {
    const row = [];
    for (const e of encodings) {
      const v = String(r[e.index]).trim();
      if (e.type === 'numeric') row.push(toNumber(v));
      else for (let k = 1; k < e.levels.length; k++) row.push(v === e.levels[k] ? 1 : 0);
    }
    return row;
  }

  // values：{ 原始特徵名稱: 值 }，回傳與 design.names 對齊的一列
  function encodeRow(values, encodings) {
    const row = [];
    for (const e of encodings) {
      const v = values[e.feature];
      if (e.type === 'numeric') row.push(Number(v));
      else for (let k = 1; k < e.levels.length; k++) row.push(String(v) === e.levels[k] ? 1 : 0);
    }
    return row;
  }

  /* ===================== 線性代數：Householder QR ===================== */

  function householderQR(X, y) {
    const n = X.length, p = X[0].length;
    const A = [];
    for (let j = 0; j < p; j++) {
      const col = new Float64Array(n);
      for (let i = 0; i < n; i++) col[i] = X[i][j];
      A.push(col);
    }
    const colNorm = A.map(col => Math.sqrt(col.reduce((s, v) => s + v * v, 0)));
    const b = Float64Array.from(y);

    for (let k = 0; k < p; k++) {
      const a = A[k];
      let norm = 0;
      for (let i = k; i < n; i++) norm += a[i] * a[i];
      norm = Math.sqrt(norm);
      if (norm === 0) continue;
      const alpha = a[k] > 0 ? -norm : norm;
      const v = new Float64Array(n - k);
      v[0] = a[k] - alpha;
      for (let i = k + 1; i < n; i++) v[i - k] = a[i];
      let vv = 0;
      for (let i = 0; i < v.length; i++) vv += v[i] * v[i];
      if (vv === 0) continue;
      const reflect = c => {
        let s = 0;
        for (let i = k; i < n; i++) s += v[i - k] * c[i];
        s = (2 * s) / vv;
        for (let i = k; i < n; i++) c[i] -= s * v[i - k];
      };
      for (let j = k; j < p; j++) reflect(A[j]);
      reflect(b);
    }

    const R = [];
    for (let i = 0; i < p; i++) {
      const row = new Array(p).fill(0);
      for (let j = i; j < p; j++) row[j] = A[j][i];
      R.push(row);
    }
    return { R, qty: b, colNorm };
  }

  function solveUpper(R, b) {
    const p = R.length, x = new Array(p).fill(0);
    for (let i = p - 1; i >= 0; i--) {
      let s = b[i];
      for (let j = i + 1; j < p; j++) s -= R[i][j] * x[j];
      x[i] = s / R[i][i];
    }
    return x;
  }

  function invertUpper(R) {
    const p = R.length;
    const inv = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let c = 0; c < p; c++) {
      const e = new Array(p).fill(0);
      e[c] = 1;
      const x = solveUpper(R, e);
      for (let i = 0; i < p; i++) inv[i][c] = x[i];
    }
    return inv;
  }

  /* ============================ 機率分布 ============================ */

  // Lanczos 近似（g = 7）
  function logGamma(x) {
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
      1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    x -= 1;
    let a = c[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  // 不完全 Beta 函數的連分數（Lentz 法）
  function betacf(a, b, x) {
    const MAXIT = 500, EPS = 1e-15, FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  // 正規化不完全 Beta 函數 I_x(a, b)
  function ibeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const lbt = logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
    if (x < (a + 1) / (a + b + 2)) return (Math.exp(lbt) * betacf(a, b, x)) / a;
    return 1 - (Math.exp(lbt) * betacf(b, a, 1 - x)) / b;
  }

  // t 檢定雙尾 p 值
  function tTwoSidedP(t, df) {
    if (Number.isNaN(t)) return NaN;
    if (!Number.isFinite(t)) return 0;
    return ibeta(df / (df + t * t), df / 2, 0.5);
  }

  function tCDF(t, df) {
    const tail = 0.5 * ibeta(df / (df + t * t), df / 2, 0.5);
    return t >= 0 ? 1 - tail : tail;
  }

  // t 分布反函數（prob > 0.5），二分法
  function tInv(prob, df) {
    let lo = 0, hi = 1;
    while (tCDF(hi, df) < prob && hi < 1e8) hi *= 2;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (tCDF(mid, df) < prob) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  }

  // F 分布右尾機率 P(F > f)
  function fSurvival(f, d1, d2) {
    if (Number.isNaN(f)) return NaN;
    if (!(f > 0)) return 1;
    if (!Number.isFinite(f)) return 0;
    return ibeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);
  }

  /* ============================== OLS 模型 ============================== */

  // cols：要使用的 X 欄位索引（預設全部）。共線（無法估計）的欄位會被移除並記錄在 droppedCollinear。
  function fitOLS(X, y, names, cols) {
    const n = y.length;
    let keep = cols ? cols.slice() : names.map((_, j) => j);
    if (n <= keep.length + 1) {
      throw new Error(`資料只有 ${n} 筆，不足以估計 ${keep.length + 1} 個參數；請降低測試集比例或減少特徵。`);
    }
    const droppedCollinear = [];
    let design, qr;
    for (;;) {
      design = X.map(r => {
        const row = [1];
        for (const j of keep) row.push(r[j]);
        return row;
      });
      qr = householderQR(design, y);
      let bad = -1;
      for (let k = 0; k <= keep.length; k++) {
        if (!(Math.abs(qr.R[k][k]) > 1e-10 * qr.colNorm[k])) { bad = k; break; }
      }
      if (bad === -1) break;
      if (bad === 0) throw new Error('無法估計截距。');
      droppedCollinear.push(names[keep[bad - 1]]);
      keep = keep.filter((_, k) => k !== bad - 1);
    }

    const p = keep.length + 1;
    const beta = solveUpper(qr.R, qr.qty);
    const fitted = design.map(r => r.reduce((s, v, j) => s + v * beta[j], 0));
    const resid = y.map((v, i) => v - fitted[i]);
    const rss = resid.reduce((s, e) => s + e * e, 0);
    const yMean = mean(y);
    const tss = y.reduce((s, v) => s + (v - yMean) * (v - yMean), 0);
    const dfResid = n - p, dfModel = p - 1;
    const sigma2 = rss / dfResid;

    // (XᵀX)⁻¹ = R⁻¹ R⁻ᵀ
    const Rinv = invertUpper(qr.R);
    const XtXinv = Rinv.map(ri => Rinv.map(rj => ri.reduce((s, v, k) => s + v * rj[k], 0)));

    const tcrit = tInv(0.975, dfResid);
    const coefficients = beta.map((b, j) => {
      const se = Math.sqrt(sigma2 * XtXinv[j][j]);
      const t = b / se;
      return {
        name: j === 0 ? '(Intercept)' : names[keep[j - 1]],
        estimate: b, se, t,
        p: tTwoSidedP(t, dfResid),
        lower: b - tcrit * se,
        upper: b + tcrit * se,
      };
    });

    const r2 = 1 - rss / tss;
    const F = dfModel > 0 ? ((tss - rss) / dfModel) / sigma2 : NaN;
    const llf = (-n / 2) * (Math.log(2 * Math.PI) + Math.log(rss / n) + 1);
    return {
      featureIndex: keep,
      features: keep.map(j => names[j]),
      droppedCollinear,
      coefficients, beta, XtXinv,
      n, p, dfResid, dfModel, rss, tss,
      r2,
      adjR2: 1 - ((1 - r2) * (n - 1)) / dfResid,
      sigma: Math.sqrt(sigma2),
      F,
      Fp: dfModel > 0 ? fSurvival(F, dfModel, dfResid) : NaN,
      aic: -2 * llf + 2 * p,
      bic: -2 * llf + Math.log(n) * p,
      fitted, resid,
    };
  }

  function predict(model, X) {
    return X.map(r => {
      let s = model.beta[0];
      model.featureIndex.forEach((j, k) => { s += model.beta[k + 1] * r[j]; });
      return s;
    });
  }

  // 單筆預測值與 95% 預測區間
  function predictInterval(model, row, level = 0.95) {
    const x = [1, ...model.featureIndex.map(j => row[j])];
    const yhat = x.reduce((s, v, i) => s + v * model.beta[i], 0);
    let q = 0;
    for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < x.length; j++) q += x[i] * model.XtXinv[i][j] * x[j];
    }
    const half = tInv(0.5 + level / 2, model.dfResid) * model.sigma * Math.sqrt(1 + q);
    return { yhat, lower: yhat - half, upper: yhat + half };
  }

  // k：模型中的特徵數（不含截距）；R² 以該資料集自身的平均為基準（同 scikit-learn r2_score）
  function metrics(y, yhat, k) {
    const n = y.length, m = mean(y);
    let rss = 0, tss = 0, ae = 0;
    for (let i = 0; i < n; i++) {
      const e = y[i] - yhat[i];
      rss += e * e;
      tss += (y[i] - m) * (y[i] - m);
      ae += Math.abs(e);
    }
    const r2 = 1 - rss / tss;
    return {
      n, r2,
      adjR2: n - k - 1 > 0 ? 1 - ((1 - r2) * (n - 1)) / (n - k - 1) : NaN,
      rmse: Math.sqrt(rss / n),
      mae: ae / n,
    };
  }

  // VIF_j = 1 / (1 − R²_j)，R²_j 為第 j 個特徵對其他特徵回歸的決定係數
  function vif(X, names, cols) {
    const use = cols ? cols.slice() : names.map((_, j) => j);
    if (use.length < 2) return use.map(() => 1);
    return use.map(j => {
      const others = use.filter(k => k !== j);
      try {
        const m = fitOLS(X, X.map(r => r[j]), names, others);
        return m.r2 >= 1 ? Infinity : 1 / (1 - m.r2);
      } catch (e) {
        return NaN;
      }
    });
  }

  // 標準化係數 β_j × sd(x_j) / sd(y)：把不同單位的特徵放在同一尺度比較影響力
  function standardizedCoefficients(model, X, y) {
    const sdY = std(y);
    return model.featureIndex.map((j, k) => (model.beta[k + 1] * std(X.map(r => r[j]))) / sdY);
  }

  // 逆向淘汰：每次移除 p 值最大且 > SL 的特徵，直到全部顯著
  function backwardElimination(X, y, names, sl, cols) {
    let active = cols ? cols.slice() : names.map((_, j) => j);
    const steps = [];
    for (;;) {
      const model = fitOLS(X, y, names, active);
      let worst = -1;
      for (let k = 1; k < model.coefficients.length; k++) {
        if (worst === -1 || model.coefficients[k].p > model.coefficients[worst].p) worst = k;
      }
      if (worst === -1 || !(model.coefficients[worst].p > sl)) return { model, steps };
      steps.push({
        removed: model.coefficients[worst].name,
        p: model.coefficients[worst].p,
        r2: model.r2,
        adjR2: model.adjR2,
        nFeatures: active.length,
      });
      active = model.featureIndex.filter((_, k) => k !== worst - 1);
    }
  }

  /* ===================== 特徵選擇：依特徵數 k 比較各方法 ===================== */

  // 標準化 Gram 矩陣：A = ZᵀZ、c = Zᵀ(y − ȳ)，Z 為標準化特徵。截距由中心化處理，
  // 任一子集的 RSS 只需解 k×k 系統，最佳子集法才能快速窮舉。
  function standardizedGram(X, y, cols) {
    const n = y.length, K = cols.length;
    const mu = cols.map(j => mean(X.map(r => r[j])));
    const sd = cols.map(j => std(X.map(r => r[j])));
    const ym = mean(y);
    const A = Array.from({ length: K }, () => new Array(K).fill(0));
    const c = new Array(K).fill(0);
    let syy = 0;
    for (let i = 0; i < n; i++) {
      const z = cols.map((j, a) => (X[i][j] - mu[a]) / sd[a]);
      const yc = y[i] - ym;
      for (let a = 0; a < K; a++) {
        c[a] += z[a] * yc;
        for (let b = a; b < K; b++) A[a][b] += z[a] * z[b];
      }
      syy += yc * yc;
    }
    for (let a = 0; a < K; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];
    return { A, c, syy, n, K };
  }

  // Cholesky 解 A_S b = c_S，回傳 RSS 與標準化尺度的係數
  function subsetSolve(g, S) {
    const k = S.length;
    if (!k) return { rss: g.syy, b: [] };
    const L = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < k; i++) {
      for (let j = 0; j <= i; j++) {
        let s = g.A[S[i]][S[j]];
        for (let t = 0; t < j; t++) s -= L[i][t] * L[j][t];
        if (i === j) {
          if (!(s > 1e-10 * g.A[S[i]][S[i]])) return { rss: Infinity, b: null };
          L[i][i] = Math.sqrt(s);
        } else L[i][j] = s / L[j][j];
      }
    }
    const z = new Array(k);
    for (let i = 0; i < k; i++) {
      let s = g.c[S[i]];
      for (let t = 0; t < i; t++) s -= L[i][t] * z[t];
      z[i] = s / L[i][i];
    }
    const b = new Array(k);
    for (let i = k - 1; i >= 0; i--) {
      let s = z[i];
      for (let t = i + 1; t < k; t++) s -= L[t][i] * b[t];
      b[i] = s / L[i][i];
    }
    let zz = 0;
    for (const v of z) zz += v * v;
    return { rss: g.syy - zz, b };
  }

  // 以下各方法回傳 subsets[k] = 第 k 步選到的特徵位置（0..K−1）

  // 前向選擇（Forward SFS）：每步加入讓訓練 RSS 下降最多的特徵
  function forwardPath(g) {
    const chosen = [], subsets = [];
    const left = [];
    for (let a = 0; a < g.K; a++) left.push(a);
    while (left.length) {
      let best = -1, bestRss = Infinity;
      for (const a of left) {
        const { rss } = subsetSolve(g, [...chosen, a]);
        if (best === -1 || rss < bestRss) { best = a; bestRss = rss; }
      }
      chosen.push(best);
      left.splice(left.indexOf(best), 1);
      subsets[chosen.length] = chosen.slice();
    }
    return subsets;
  }

  // RFE：從全部特徵開始，每步移除 |標準化係數| 最小的特徵
  function rfePath(g) {
    let S = [];
    for (let a = 0; a < g.K; a++) S.push(a);
    const subsets = [];
    subsets[S.length] = S.slice();
    while (S.length > 1) {
      const { b } = subsetSolve(g, S);
      let worst = 0;
      for (let i = 1; i < S.length; i++) if (Math.abs(b[i]) < Math.abs(b[worst])) worst = i;
      S = S.filter((_, i) => i !== worst);
      subsets[S.length] = S.slice();
    }
    return subsets;
  }

  // 逆向淘汰（p 值）一路做到只剩 1 個特徵
  function pValuePath(X, y, names, cols) {
    let active = cols.slice();
    const subsets = [];
    subsets[active.length] = active.map(j => cols.indexOf(j));
    while (active.length > 1) {
      const m = fitOLS(X, y, names, active);
      let worst = 1;
      for (let k = 2; k < m.coefficients.length; k++) if (m.coefficients[k].p > m.coefficients[worst].p) worst = k;
      active = m.featureIndex.filter((_, k) => k !== worst - 1);
      subsets[active.length] = active.map(j => cols.indexOf(j));
    }
    return subsets;
  }

  // 最佳子集（Exhaustive）：每個 k 找訓練 RSS 最小的組合（特徵數 ≤ 15 才窮舉）
  function bestSubsetPath(g, maxK = 15) {
    const K = g.K;
    if (K > maxK) return null;
    const best = [];
    for (let mask = 1; mask < (1 << K); mask++) {
      const S = [];
      for (let a = 0; a < K; a++) if (mask & (1 << a)) S.push(a);
      const { rss } = subsetSolve(g, S);
      if (!best[S.length] || rss < best[S.length].rss) best[S.length] = { rss, S };
    }
    return best.map(v => v && v.S);
  }

  // Lasso / ElasticNet：沿 λ 由大到小的路徑（座標下降），依特徵「第一次進入模型」的順序排序
  function enetPath(g, l1Ratio, nLambda = 100, eps = 1e-4) {
    const K = g.K, n = g.n;
    const sdY = Math.sqrt(g.syy / (n - 1));
    const c = g.c.map(v => v / sdY);
    let lamMax = 0;
    for (const v of c) lamMax = Math.max(lamMax, Math.abs(v) / (n * l1Ratio));
    const coef = new Array(K).fill(0);
    const enter = new Array(K).fill(Infinity);
    for (let s = 0; s < nLambda; s++) {
      const lam = lamMax * Math.pow(eps, s / (nLambda - 1));
      for (let it = 0; it < 2000; it++) {
        let maxDelta = 0;
        for (let j = 0; j < K; j++) {
          let r = c[j];
          for (let l = 0; l < K; l++) if (l !== j) r -= g.A[j][l] * coef[l];
          const z = r / n;
          const nb = (Math.sign(z) * Math.max(Math.abs(z) - lam * l1Ratio, 0)) / (g.A[j][j] / n + lam * (1 - l1Ratio));
          maxDelta = Math.max(maxDelta, Math.abs(nb - coef[j]));
          coef[j] = nb;
        }
        if (maxDelta < 1e-10) break;
      }
      for (let j = 0; j < K; j++) if (enter[j] === Infinity && Math.abs(coef[j]) > 1e-12) enter[j] = s;
    }
    const order = [];
    for (let a = 0; a < K; a++) order.push(a);
    order.sort((a, b) => (enter[a] - enter[b]) || (Math.abs(coef[b]) - Math.abs(coef[a])) || (a - b));
    const subsets = [];
    for (let k = 1; k <= K; k++) subsets[k] = order.slice(0, k);
    return subsets;
  }

  const SELECTION_METHODS = [
    { key: 'backward_p', label: '逆向淘汰（p 值）', short: '逆向 p 值' },
    { key: 'forward', label: '前向選擇 Forward', short: 'Forward' },
    { key: 'rfe', label: 'RFE（|標準化係數|）', short: 'RFE' },
    { key: 'best', label: '最佳子集 Exhaustive', short: '最佳子集' },
    { key: 'lasso', label: 'Lasso（L1 排序）', short: 'Lasso' },
    { key: 'enet', label: 'ElasticNet（α = 0.5）', short: 'ElasticNet' },
  ];

  // 對每個方法、每個 k：在訓練集以 OLS 配適該子集，再用測試集評估
  function featureSelection(Xtr, ytr, Xte, yte, names, cols) {
    const g = standardizedGram(Xtr, ytr, cols);
    const subsetsByKey = {
      backward_p: pValuePath(Xtr, ytr, names, cols),
      forward: forwardPath(g),
      rfe: rfePath(g),
      best: bestSubsetPath(g),
      lasso: enetPath(g, 1),
      enet: enetPath(g, 0.5),
    };
    const cache = new Map();
    const evaluate = S => {
      const sub = S.map(a => cols[a]).sort((a, b) => a - b);
      const key = sub.join(',');
      if (!cache.has(key)) {
        const m = fitOLS(Xtr, ytr, names, sub);
        cache.set(key, {
          cols: sub, features: m.features, k: sub.length,
          train: metrics(ytr, predict(m, Xtr), m.p - 1),
          test: yte.length > 1 ? metrics(yte, predict(m, Xte), m.p - 1) : null,
        });
      }
      return cache.get(key);
    };
    const methods = SELECTION_METHODS.filter(m => subsetsByKey[m.key]).map(m => {
      const results = [];
      for (let k = 1; k <= cols.length; k++) results[k] = evaluate(subsetsByKey[m.key][k]);
      return { ...m, results };
    });
    // 最佳組合：測試集 RMSE 最小（同分取較少特徵）；沒有測試集時改用訓練集 Adj R²
    let best = null;
    for (const m of methods) {
      for (let k = 1; k <= cols.length; k++) {
        const res = m.results[k];
        const score = res.test ? res.test.rmse : -res.train.adjR2;
        if (!best || score < best.score || (score === best.score && k < best.k)) best = { score, k, method: m.key, label: m.label, ...res };
      }
    }
    best.sameMethods = methods.filter(m => m.results[best.k].cols.join() === best.cols.join()).map(m => m.label);
    return { K: cols.length, methods, best };
  }

  /* ========================= 完整流程（Modeling + Evaluation） ========================= */

  // cfg：{ target, features, testRatio, seed, sl }
  function runAnalysis(table, columns, cfg) {
    const design = buildDesign(table, columns, cfg.target, cfg.features);
    const { X, y, names } = design;
    if (!names.length) throw new Error('請至少選擇一個特徵（feature）。');
    if (y.length < 5) throw new Error(`可用的資料只有 ${y.length} 筆，太少了。`);

    const split = trainTestSplit(y.length, cfg.testRatio, cfg.seed);
    const take = (idx, arr) => idx.map(i => arr[i]);
    const Xtr = take(split.train, X), ytr = take(split.train, y);
    const Xte = take(split.test, X), yte = take(split.test, y);

    const full = fitOLS(Xtr, ytr, names);
    const be = backwardElimination(Xtr, ytr, names, cfg.sl, full.featureIndex);
    const reduced = be.model;
    const selection = featureSelection(Xtr, ytr, Xte, yte, names, full.featureIndex);
    const sweet = fitOLS(Xtr, ytr, names, selection.best.cols);
    const evaluate = m => ({
      train: metrics(ytr, predict(m, Xtr), m.p - 1),
      test: yte.length > 1 ? metrics(yte, predict(m, Xte), m.p - 1) : null,
    });

    return {
      cfg, design, split, Xtr, ytr, Xte, yte,
      full, reduced, sweet,
      steps: be.steps,
      selection,
      vif: vif(Xtr, names, full.featureIndex),
      stdCoef: {
        full: standardizedCoefficients(full, Xtr, ytr),
        reduced: standardizedCoefficients(reduced, Xtr, ytr),
        sweet: standardizedCoefficients(sweet, Xtr, ytr),
      },
      evaluation: { full: evaluate(full), reduced: evaluate(reduced), sweet: evaluate(sweet) },
    };
  }

  return {
    isMissing, toNumber, mean, std, describe, pearson,
    mulberry32, shuffledIndices, trainTestSplit,
    parseCSV, inferColumns, buildDesign, encodeRow,
    fitOLS, predict, predictInterval, metrics, vif, standardizedCoefficients, backwardElimination,
    featureSelection, SELECTION_METHODS,
    tTwoSidedP, tInv, fSurvival,
    runAnalysis,
  };
});
