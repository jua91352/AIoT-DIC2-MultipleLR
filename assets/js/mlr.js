/*!
 * mlr.js — 多元線性回歸（OLS）計算引擎，無外部相依套件
 * 瀏覽器：window.MLR；Node：require('./mlr.js')
 *
 * 資料切分與 scikit-learn train_test_split(random_state) 逐位元相同（numpy MT19937），
 * One-hot 與 pandas get_dummies(drop_first=True) 相同；5 種特徵選擇方法（Pearson、SelectKBest、
 * RFE、Forward SFS、Lasso）與 LassoCV 都依 scikit-learn 的定義實作，
 * 和 .claude/skills/mlr-crisp-dm/scripts/mlr_pipeline.py 的結果可以互相對照。
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

  // MT19937：與 numpy.random.RandomState(seed) 逐位元相同的 32-bit 亂數產生器
  function mt19937(seed) {
    const mt = new Uint32Array(624);
    let pos = 624;
    mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const s = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, s) + i) >>> 0;
    }
    function twist() {
      for (let i = 0; i < 624; i++) {
        const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
        let v = mt[(i + 397) % 624] ^ (y >>> 1);
        if (y & 1) v ^= 0x9908b0df;
        mt[i] = v >>> 0;
      }
      pos = 0;
    }
    return function next() {
      if (pos >= 624) twist();
      let y = mt[pos++];
      y ^= y >>> 11;
      y ^= (y << 7) & 0x9d2c5680;
      y ^= (y << 15) & 0xefc60000;
      y ^= y >>> 18;
      return y >>> 0;
    };
  }

  // numpy RandomState(seed).permutation(n)：Fisher–Yates，j 以位元遮罩 + 拒絕取樣產生（legacy random_interval）
  function permutation(n, seed) {
    const next = mt19937(seed);
    const idx = [];
    for (let i = 0; i < n; i++) idx.push(i);
    for (let i = n - 1; i > 0; i--) {
      let mask = i;
      mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16;
      let j;
      while ((j = (next() & mask) >>> 0) > i);
      const t = idx[i];
      idx[i] = idx[j];
      idx[j] = t;
    }
    return idx;
  }

  // 與 sklearn train_test_split(test_size, random_state=seed) 相同：排列的前 ceil(n × test_size) 筆為測試集，
  // 接下來 floor(n × (1 − test_size)) 筆為訓練集（兩者都保留排列順序，交叉驗證的折也依這個順序切）
  function trainTestSplit(n, testRatio, seed) {
    const perm = permutation(n, seed);
    const nTest = Math.ceil(testRatio * n), nTrain = Math.floor((1 - testRatio) * n);
    return { test: perm.slice(0, nTest), train: perm.slice(nTest, nTest + nTrain) };
  }

  // sklearn KFold(n_splits, shuffle=True, random_state=seed)：先以 RandomState(seed) 洗牌，
  // 再依序切段（前 n % n_splits 段多 1 筆）；seed 為 null 時不洗牌
  function kFold(n, nSplits, seed) {
    const idx = seed == null ? Array.from({ length: n }, (_, i) => i) : permutation(n, seed);
    const folds = [];
    let start = 0;
    for (let f = 0; f < nSplits; f++) {
      const size = Math.floor(n / nSplits) + (f < n % nSplits ? 1 : 0);
      const inTest = new Uint8Array(n);
      for (let i = start; i < start + size; i++) inTest[idx[i]] = 1;
      const test = idx.slice(start, start + size), train = [];
      for (let i = 0; i < n; i++) if (!inTest[i]) train.push(i);
      folds.push({ train, test });
      start += size;
    }
    return folds;
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
    // 與 pandas get_dummies 相同的欄位順序：數值欄在前，類別欄展開的 dummy 接在後面
    const fcs = features.map(f => {
      const c = byName.get(f);
      if (!c) throw new Error(`找不到特徵欄位「${f}」。`);
      return c;
    }).sort((a, b) => (a.type === 'numeric' ? 0 : 1) - (b.type === 'numeric' ? 0 : 1) || a.index - b.index);

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

  /* ===================== 特徵選擇：5 種方法 × 特徵數 k ===================== */

  const take = (idx, arr) => idx.map(i => arr[i]);

  // StandardScaler：只用訓練集配適，母體標準差（ddof = 0），常數欄的尺度設為 1
  function standardScaler(X) {
    const n = X.length, K = X[0].length;
    const center = [], scale = [];
    for (let j = 0; j < K; j++) {
      let m = 0;
      for (const r of X) m += r[j];
      m /= n;
      let v = 0;
      for (const r of X) v += (r[j] - m) * (r[j] - m);
      center.push(m);
      scale.push(v > 0 ? Math.sqrt(v / n) : 1);
    }
    return { center, scale, transform: A => A.map(r => r.map((v, j) => (v - center[j]) / scale[j])) };
  }

  // 只求係數的最小平方法（含截距），給交叉驗證、RFE 這類大量重複配適使用；共線欄位自動略過
  function lsFit(X, y, cols) {
    let keep = cols.slice();
    for (;;) {
      if (X.length <= keep.length) return null;
      const design = X.map(r => {
        const row = [1];
        for (const j of keep) row.push(r[j]);
        return row;
      });
      const qr = householderQR(design, y);
      let bad = -1;
      for (let k = 0; k <= keep.length; k++) {
        if (!(Math.abs(qr.R[k][k]) > 1e-10 * qr.colNorm[k])) { bad = k; break; }
      }
      if (bad === -1) return { cols: keep, beta: solveUpper(qr.R, qr.qty) };
      if (bad === 0) return null;
      keep = keep.filter((_, k) => k !== bad - 1);
    }
  }

  function lsPredict(fit, X) {
    return X.map(r => {
      let s = fit.beta[0];
      fit.cols.forEach((j, k) => { s += fit.beta[k + 1] * r[j]; });
      return s;
    });
  }

  // k 折交叉驗證（sklearn cross_val_score 的定義：每折算一次分數再平均）
  function cvScore(folds, cols) {
    let r2 = 0, rmse = 0;
    for (const f of folds) {
      const fit = lsFit(f.Xtr, f.ytr, cols);
      if (!fit) return { r2: NaN, rmse: NaN };
      const m = metrics(f.yte, lsPredict(fit, f.Xte), cols.length);
      r2 += m.r2;
      rmse += m.rmse;
    }
    return { r2: r2 / folds.length, rmse: rmse / folds.length };
  }

  function makeFolds(X, y, nSplits, seed) {
    return kFold(y.length, nSplits, seed).map(f => ({
      Xtr: take(f.train, X), ytr: take(f.train, y), Xte: take(f.test, X), yte: take(f.test, y),
    }));
  }

  // 1. Pearson（Filter）：訓練集 |r(x_j, y)| 由大到小
  function pearsonRanking(r) {
    const idx = r.map((_, j) => j);
    const score = j => (Number.isFinite(r[j]) ? Math.abs(r[j]) : 0);
    return idx.sort((a, b) => score(b) - score(a) || a - b);
  }

  // 2. SelectKBest(f_regression)（Filter）：F_j = r² / (1 − r²) × (n − 2)，取 F 最大的 k 個。
  //    同分時與 sklearn 相同（mergesort 由小到大排序後取最後 k 個 → 索引大的優先）
  function kbestRanking(F) {
    const idx = F.map((_, j) => j);
    const score = j => (Number.isFinite(F[j]) ? F[j] : 0);
    return idx.sort((a, b) => score(b) - score(a) || b - a);
  }

  // 3. RFE（Wrapper）：標準化特徵上配適 OLS，每輪移除 |係數| 最小的特徵，直到剩 1 個
  function rfeRanking(Z, y) {
    let S = Z[0].map((_, j) => j);
    const removed = [];
    while (S.length > 1) {
      const fit = lsFit(Z, y, S);
      let worst = 0;
      if (fit && fit.cols.length === S.length) {
        for (let i = 1; i < S.length; i++) if (Math.abs(fit.beta[i + 1]) < Math.abs(fit.beta[worst + 1])) worst = i;
      }
      removed.push(S[worst]);
      S = S.filter((_, i) => i !== worst);
    }
    return [...S, ...removed.reverse()];
  }

  // 4. Forward SFS（Wrapper）：sklearn SequentialFeatureSelector(LinearRegression(), direction="forward",
  //    cv=KFold(5, shuffle=True, random_state))，每一步加入讓 5 折交叉驗證平均 R² 最高的特徵
  function forwardRanking(folds, K) {
    const chosen = [], left = [];
    for (let j = 0; j < K; j++) left.push(j);
    while (left.length) {
      let best = -1, bestScore = -Infinity;
      for (const j of left) {
        const s = cvScore(folds, [...chosen, j]).r2;
        if (best === -1 || s > bestScore) { best = j; bestScore = s; }
      }
      chosen.push(best);
      left.splice(left.indexOf(best), 1);
    }
    return chosen;
  }

  // 中心化後的 Gram 矩陣：G = XcᵀXc、c = Xcᵀyc（Lasso 座標下降只需要這兩個）
  function centeredGram(X, y) {
    const n = y.length, K = X[0].length;
    const mx = new Array(K).fill(0);
    for (const r of X) for (let j = 0; j < K; j++) mx[j] += r[j] / n;
    const my = mean(y);
    const G = Array.from({ length: K }, () => new Array(K).fill(0));
    const c = new Array(K).fill(0);
    for (let i = 0; i < n; i++) {
      const xc = X[i].map((v, j) => v - mx[j]);
      const yc = y[i] - my;
      for (let a = 0; a < K; a++) {
        c[a] += xc[a] * yc;
        for (let b = a; b < K; b++) G[a][b] += xc[a] * xc[b];
      }
    }
    for (let a = 0; a < K; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
    return { G, c, n, K, mx, my };
  }

  // Lasso 座標下降：min (1/2n)‖yc − Xc b‖² + α‖b‖₁（與 sklearn Lasso 的目標函數相同），b 為暖啟動初值
  function lassoCD(g, alpha, b) {
    const t = g.n * alpha;
    for (let it = 0; it < 100000; it++) {
      let maxDelta = 0, maxB = 0;
      for (let j = 0; j < g.K; j++) {
        const d = g.G[j][j];
        if (!(d > 0)) continue;
        let r = g.c[j];
        for (let l = 0; l < g.K; l++) if (l !== j) r -= g.G[j][l] * b[l];
        const nb = r > t ? (r - t) / d : r < -t ? (r + t) / d : 0;
        maxDelta = Math.max(maxDelta, Math.abs(nb - b[j]));
        maxB = Math.max(maxB, Math.abs(nb));
        b[j] = nb;
      }
      if (maxDelta === 0 || maxDelta <= 1e-13 * maxB) break;
    }
    return b;
  }

  const alphaMax = g => Math.max(...g.c.map(Math.abs)) / g.n;

  // np.geomspace(αmax, αmax × eps, num)
  function alphaGrid(amax, eps, num) {
    const out = [];
    for (let i = 0; i < num; i++) out.push(amax * Math.pow(eps, i / (num - 1)));
    return out;
  }

  // 5. Lasso（Embedded）：依 LassoCV 收縮後的 |係數| 由大到小排序；係數同為 0 的特徵依
  //    np.argsort(|coef|)[::-1] 的慣例排序（索引大的在前，與 SelectKBest 的同分規則相同）
  function lassoRanking(coef) {
    return coef.map((_, j) => j).sort((a, b) => Math.abs(coef[b]) - Math.abs(coef[a]) || b - a);
  }

  // LassoCV(cv=KFold(5, shuffle=True, random_state))：在標準化訓練集上以 5 折 CV 從 100 個 α 中選
  //    平均 MSE 最小者，再用整個訓練集重新配適。回傳 Lasso 原生收縮後的係數（不再用 OLS 重配），即圖中的 ★
  function lassoCV(Z, y, foldsZ, nAlphas = 100, eps = 1e-3) {
    const gAll = centeredGram(Z, y);
    const amax = alphaMax(gAll);
    if (!(amax > 0)) return null;
    const alphas = alphaGrid(amax, eps, nAlphas);
    const mse = new Array(nAlphas).fill(0);
    for (const f of foldsZ) {
      const g = centeredGram(f.Xtr, f.ytr);
      const b = new Array(g.K).fill(0);
      alphas.forEach((alpha, a) => {
        lassoCD(g, alpha, b);
        const b0 = g.my - g.mx.reduce((s, m, j) => s + m * b[j], 0);
        let e2 = 0;
        f.Xte.forEach((r, i) => {
          const p = r.reduce((s, v, j) => s + v * b[j], b0);
          e2 += (f.yte[i] - p) * (f.yte[i] - p);
        });
        mse[a] += e2 / f.yte.length / foldsZ.length;
      });
    }
    let best = 0;
    for (let a = 1; a < nAlphas; a++) if (mse[a] < mse[best]) best = a;
    const b = new Array(gAll.K).fill(0);
    for (let a = 0; a <= best; a++) lassoCD(gAll, alphas[a], b);
    const intercept = gAll.my - gAll.mx.reduce((s, m, j) => s + m * b[j], 0);
    return { alphas, mse, alpha: alphas[best], index: best, coef: b, intercept };
  }

  const SELECTION_METHODS = [
    { key: 'pearson', label: 'Pearson', family: 'Filter', short: 'Pearson' },
    { key: 'kbest', label: 'SelectKBest F-test', family: 'Filter', short: 'SelectKBest' },
    { key: 'rfe', label: 'RFE', family: 'Wrapper', short: 'RFE' },
    { key: 'sfs', label: 'Forward SFS', family: 'Wrapper', short: 'Forward SFS' },
    { key: 'lasso', label: 'Lasso', family: 'Embedded', short: 'Lasso' },
  ];

  // cols：候選特徵（design 欄位索引）；seed：交叉驗證 KFold 的 random_state。
  // 每個方法對每個 k 選出特徵後，都在訓練集用 OLS 重新配適，再評估測試集、訓練集與 5 折 CV（訓練集內）
  function featureSelection(Xtr, ytr, Xte, yte, names, cols, seed, nFolds = 5) {
    const K = cols.length;
    const Xc = Xtr.map(r => cols.map(j => r[j]));
    const Xc_te = Xte.map(r => cols.map(j => r[j]));
    const scaler = standardScaler(Xc);
    const Z = scaler.transform(Xc), Zte = scaler.transform(Xc_te);
    const folds = makeFolds(Xc, ytr, nFolds, seed);
    const foldsZ = makeFolds(Z, ytr, nFolds, seed);

    const r = cols.map((_, a) => pearson(Xc.map(row => row[a]), ytr));
    const n = ytr.length;
    const F = r.map(v => (v * v) / (1 - v * v) * (n - 2));
    const Fp = F.map(f => fSurvival(f, 1, n - 2));
    let lcv = lassoCV(Z, ytr, foldsZ);
    const rankings = {
      pearson: pearsonRanking(r),
      kbest: kbestRanking(F),
      rfe: rfeRanking(Z, ytr),
      sfs: forwardRanking(folds, K),
      lasso: lcv ? lassoRanking(lcv.coef) : pearsonRanking(r),
    };

    const cache = new Map();
    const evaluate = local => {
      const sorted = local.slice().sort((a, b) => a - b);
      const key = sorted.join(',');
      if (!cache.has(key)) {
        const global = sorted.map(a => cols[a]);
        const model = fitOLS(Xtr, ytr, names, global);
        cache.set(key, {
          key, local: sorted, cols: global, k: sorted.length, model,
          features: model.features,
          train: metrics(ytr, predict(model, Xtr), sorted.length),
          test: yte.length > 1 ? metrics(yte, predict(model, Xte), sorted.length) : null,
          cv: cvScore(folds, sorted),
        });
      }
      return cache.get(key);
    };

    const methods = SELECTION_METHODS.map(m => {
      const order = rankings[m.key];
      const results = [null];
      for (let k = 1; k <= K; k++) {
        results[k] = { ...evaluate(order.slice(0, k)), added: names[cols[order[k - 1]]], order: order.slice(0, k).map(a => names[cols[a]]) };
      }
      return { ...m, ranking: order.map(a => names[cols[a]]), results };
    });

    // LassoCV 原生收縮模型（係數換回原始單位）
    if (lcv) {
      const coefRaw = lcv.coef.map((b, a) => b / scaler.scale[a]);
      const intercept = lcv.intercept - coefRaw.reduce((s, b, a) => s + b * scaler.center[a], 0);
      const pred = A => A.map(row => row.reduce((s, v, a) => s + v * coefRaw[a], intercept));
      const active = cols.filter((_, a) => lcv.coef[a] !== 0);
      lcv = {
        ...lcv, coefRaw, interceptRaw: intercept, k: active.length,
        features: active.map(j => names[j]),
        train: metrics(ytr, pred(Xc), active.length),
        test: yte.length > 1 ? metrics(yte, pred(Xc_te), active.length) : null,
      };
    }

    return {
      K, cols, names: cols.map(j => names[j]), methods, lassoCV: lcv, nFolds,
      scaler, corr: r, F, Fp, seed,
    };
  }

  // 依指定的評估基準（test / train / cv）找 RMSE 最低的組合；同分取特徵較少者，再依方法順序
  function bestBy(selection, basis) {
    let best = null;
    for (const m of selection.methods) {
      for (let k = 1; k <= selection.K; k++) {
        const res = m.results[k];
        const s = res[basis];
        if (!s || !Number.isFinite(s.rmse)) continue;
        const better = !best || s.rmse < best.score * (1 - 1e-12) ||
          (Math.abs(s.rmse - best.score) <= 1e-12 * best.score && k < best.k);
        if (better) best = { score: s.rmse, k, method: m.key, res };
      }
    }
    if (best) {
      best.sameMethods = selection.methods.filter(m => m.results[best.k].key === best.res.key).map(m => m.key);
    }
    return best;
  }

  /* ========================= 完整流程（Modeling + Evaluation） ========================= */

  // cfg：{ target, features, testRatio, seed }
  function runAnalysis(table, columns, cfg) {
    const design = buildDesign(table, columns, cfg.target, cfg.features);
    const { X, y, names } = design;
    if (!names.length) throw new Error('請至少選擇一個特徵（feature）。');
    if (y.length < 15) throw new Error(`可用的資料只有 ${y.length} 筆，太少了（至少 15 筆）。`);

    const split = trainTestSplit(y.length, cfg.testRatio, cfg.seed);
    const Xtr = take(split.train, X), ytr = take(split.train, y);
    const Xte = take(split.test, X), yte = take(split.test, y);

    const full = fitOLS(Xtr, ytr, names);
    const selection = featureSelection(Xtr, ytr, Xte, yte, names, full.featureIndex, cfg.seed);
    return {
      cfg, design, split, Xtr, ytr, Xte, yte, full, selection,
      vif: vif(Xtr, names, full.featureIndex),
      best: { test: bestBy(selection, 'test'), train: bestBy(selection, 'train'), cv: bestBy(selection, 'cv') },
    };
  }

  return {
    isMissing, toNumber, mean, std, describe, pearson,
    mt19937, permutation, trainTestSplit, kFold,
    parseCSV, inferColumns, buildDesign, encodeRow,
    fitOLS, predict, predictInterval, metrics, vif, standardizedCoefficients,
    featureSelection, bestBy, SELECTION_METHODS,
    tTwoSidedP, tInv, fSurvival,
    runAnalysis,
  };
});
