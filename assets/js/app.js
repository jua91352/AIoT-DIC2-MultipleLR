/* app.js — 介面：下拉選單切換 Problem、任意 CSV 網址 / 上傳檔案、CRISP-DM 六步驟結果 */
(function () {
  'use strict';

  const PRESETS = {
    startups: {
      label: 'Problem 1 — 50 Startups（預測 Profit）',
      url: 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv',
      target: 'Profit',
      business: `
        <p>創投公司想知道新創公司的<strong>支出結構</strong>與<strong>所在州別</strong>如何影響年度利潤：研發（R&amp;D Spend）、行政（Administration）、行銷（Marketing Spend）支出與所在州（State），哪些因素真正影響利潤（Profit）？</p>
        <ul>
          <li>目標：建立多元線性回歸模型預測 <code>Profit</code>。</li>
          <li>成功標準：測試集 R² 高、RMSE 小，並用逆向淘汰找出最精簡且顯著的特徵組合，作為投資判斷依據。</li>
        </ul>`,
      notes: {
        'R&D Spend': '研發支出', Administration: '行政支出', 'Marketing Spend': '行銷支出',
        State: '所在州（New York / California / Florida）', Profit: '年度利潤',
      },
    },
    boston: {
      label: 'Problem 2 — Boston Housing（預測房價 medv）',
      url: 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
      target: 'medv',
      business: `
        <p>以 1970 年代波士頓地區 506 個普查區的 13 項特徵，預測<strong>自住房屋價值中位數 medv</strong>（單位：千美元），找出影響房價的關鍵因素，作為不動產估價與都市規劃的參考。</p>
        <ul>
          <li>目標：建立多元線性回歸模型預測 <code>medv</code>。</li>
          <li>成功標準：測試集 R² 高、RMSE 小，並理解各特徵對房價的方向與強度。</li>
        </ul>
        <p class="note">註：變數 <code>b</code> 由黑人人口比例計算而來，有倫理爭議（scikit-learn 1.2 起已移除此資料集）。這裡沿用作業指定的經典資料集；可以在上方取消勾選 <code>b</code> 比較結果。</p>`,
      notes: {
        crim: '城鎮人均犯罪率', zn: '大面積住宅用地比例（> 25,000 ft²）', indus: '非零售商業用地比例',
        chas: '是否臨查爾斯河（1 = 是）', nox: '一氧化氮濃度（每千萬分之一）', rm: '每戶平均房間數',
        age: '1940 年前建成的自住房比例', dis: '到 5 個波士頓就業中心的加權距離', rad: '放射狀公路可達性指數',
        tax: '每 1 萬美元的全額財產稅率', ptratio: '城鎮師生比', b: '1000(Bk − 0.63)²，Bk 為黑人人口比例',
        lstat: '低社經地位人口比例（%）', medv: '自住房屋價值中位數（千美元）',
      },
    },
    custom: {
      label: '自訂資料集（任意 CSV 網址 / 上傳檔案）',
      url: '',
      target: null,
      business: `
        <p>自訂資料集：以選擇的目標欄位為 y、勾選的欄位為 X，套用與 Problem 1、2 相同的 CRISP-DM 多元線性回歸流程（型態判斷 → One-hot → 切分 → OLS → 逆向淘汰 → 評估）。</p>`,
      notes: {},
    },
  };

  const TYPE_LABEL = { numeric: '數值', categorical: '類別', empty: '空' };
  const DELIM_LABEL = { ',': '逗號', ';': '分號', '\t': 'Tab', '|': '直線', whitespace: '空白' };

  const $ = sel => document.querySelector(sel);
  const state = {
    key: 'startups', source: '', sourceIsUrl: true,
    table: null, columns: null, result: null,
    view: 'full', charts: {}, pendingTarget: null, timer: null, loadToken: 0, loadedMsg: '',
  };

  /* ============================== 格式化 ============================== */

  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = s => String(s).replace(/[&<>"']/g, c => ESC[c]);

  function fmt(x, sig = 5) {
    if (x == null || Number.isNaN(x)) return '—';
    if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
    if (x === 0) return '0';
    const ax = Math.abs(x);
    if (ax >= 1e9 || ax < 1e-4) return x.toExponential(2);
    const digits = Math.min(Math.max(0, sig - 1 - Math.floor(Math.log10(ax))), 8);
    return x.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }
  const fmtP = p => (Number.isNaN(p) ? '—' : p < 1e-4 ? p.toExponential(2) : p.toFixed(4));
  const stars = p => (p < 0.001 ? '***' : p < 0.01 ? '**' : p < 0.05 ? '*' : p < 0.1 ? '.' : '');
  const f4 = x => (Number.isFinite(x) ? x.toFixed(4) : '—');
  const codeList = names => names.map(n => `<code>${esc(n)}</code>`).join('、');

  // left：要靠左對齊的欄位索引（文字欄）；cell 可以是字串或 { html, cls, title }
  function htmlTable(headers, rows, left = [0]) {
    const L = new Set(left);
    const head = '<thead><tr>' + headers.map((h, i) => `<th${L.has(i) ? ' class="l"' : ''}>${h}</th>`).join('') + '</tr></thead>';
    const body = '<tbody>' + rows.map(r => `<tr${r.cls ? ` class="${r.cls}"` : ''}>` + r.cells.map((c, i) => {
      const obj = c !== null && typeof c === 'object';
      const cls = [L.has(i) ? 'l' : '', obj && c.cls ? c.cls : ''].filter(Boolean).join(' ');
      const title = obj && c.title ? ` title="${esc(c.title)}"` : '';
      return `<td${cls ? ` class="${cls}"` : ''}${title}>${obj ? c.html : c}</td>`;
    }).join('') + '</tr>').join('') + '</tbody>';
    return `<table>${head}${body}</table>`;
  }

  const VIEW_LABEL = { full: '全特徵模型', reduced: '逆向淘汰模型', sweet: '最佳 k 模型' };
  const LINE_COLORS = ['#3b5bdb', '#e8590c', '#0c8599', '#d6336c', '#5c940d', '#7048e8'];
  const LINE_STYLES = ['circle', 'rect', 'triangle', 'rectRot', 'crossRot', 'star'];

  const kpis = items => items.map(([v, label]) => `<div class="kpi"><b>${v}</b><span>${label}</span></div>`).join('');

  function extent(arr) {
    let lo = Infinity, hi = -Infinity;
    for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return [lo, hi];
  }

  const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function alpha(hex, a) {
    let h = hex.replace('#', '');
    if (h.length === 3) h = h.split('').map(ch => ch + ch).join('');
    const n = parseInt(h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }

  function setStatus(msg, kind) {
    const s = $('#status');
    s.textContent = msg;
    s.className = 'status ' + (kind || '');
  }

  /* ============================== 載入資料 ============================== */

  function normalizeUrl(u) {
    const m = u.match(/^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/blob\/(.+)$/);
    return m ? `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${m[3]}` : u;
  }

  function selectProblem(key, autoload) {
    state.key = key;
    $('#problem').value = key;
    $('#url').value = PRESETS[key].url;
    if (key === 'custom') {
      state.table = null;
      state.result = null;
      $('#results').hidden = true;
      $('#config').hidden = true;
      setStatus('請貼上 CSV 網址後按「載入」，或上傳本機 CSV 檔。', 'info');
      history.replaceState(null, '', location.pathname + '?problem=custom');
      $('#url').focus();
      return;
    }
    if (autoload) loadUrl();
  }

  async function loadUrl() {
    let url = $('#url').value.trim();
    if (!url) { setStatus('請先輸入資料集網址。', 'error'); return; }
    if (!/^https?:\/\//i.test(url)) url = 'https://' + url;
    url = normalizeUrl(url);
    $('#url').value = url;
    if (state.key !== 'custom' && url !== PRESETS[state.key].url) {
      state.key = 'custom';
      $('#problem').value = 'custom';
    }
    const token = ++state.loadToken;
    setStatus('讀取資料中…', 'loading');
    let text;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      text = await res.text();
    } catch (err) {
      if (token !== state.loadToken) return;
      setStatus(`無法讀取資料集（${err.message}）。請確認網址是「原始 CSV」（例如 raw.githubusercontent.com/…），而且該網站允許跨來源讀取（CORS）。`, 'error');
      return;
    }
    if (token !== state.loadToken) return;
    ingest(text, url, true);
  }

  function onFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      state.key = 'custom';
      $('#problem').value = 'custom';
      $('#url').value = '';
      ++state.loadToken;
      ingest(String(reader.result), file.name, false);
    };
    reader.onerror = () => setStatus('檔案讀取失敗。', 'error');
    reader.readAsText(file);
    e.target.value = '';
  }

  function ingest(text, source, isUrl) {
    try {
      const table = MLR.parseCSV(text);
      if (table.rows.length < 5) throw new Error('資料少於 5 筆，無法建立回歸模型。');
      const columns = MLR.inferColumns(table);
      const numeric = columns.filter(c => c.type === 'numeric' && !c.exclude);
      if (!numeric.length) throw new Error('找不到可用的數值欄位，無法做回歸。');

      state.table = table;
      state.columns = columns;
      state.source = source;
      state.sourceIsUrl = isUrl;

      let target = state.pendingTarget || PRESETS[state.key].target;
      if (!numeric.some(c => c.name === target)) target = numeric[numeric.length - 1].name;
      state.pendingTarget = null;

      const sel = $('#target');
      sel.innerHTML = '';
      for (const c of numeric) sel.add(new Option(c.name, c.name));
      sel.value = target;
      fillFeatures(target);
      $('#config').hidden = false;

      state.loadedMsg = `已載入 ${table.rows.length} 筆 × ${table.header.length} 欄（分隔符號：${DELIM_LABEL[table.delimiter]}` +
        `${table.hasHeader ? '' : '；沒有標頭，自動命名欄位'}）。可調整目標、特徵與切分參數，結果會即時更新。`;
      run();
    } catch (err) {
      setStatus(`資料解析失敗：${err.message}`, 'error');
      $('#results').hidden = true;
    }
  }

  function fillFeatures(target) {
    const box = $('#features');
    box.innerHTML = '';
    for (const c of state.columns) {
      if (c.name === target || c.type === 'empty') continue;
      const label = document.createElement('label');
      label.className = 'chip';
      if (c.exclude) label.title = `預設排除：${c.exclude}`;
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = c.name;
      input.checked = !c.exclude;
      const name = document.createElement('span');
      name.textContent = c.name;
      label.append(input, name);
      if (c.type === 'categorical') label.insertAdjacentHTML('beforeend', '<span class="badge">類別 → One-hot</span>');
      if (c.exclude) label.insertAdjacentHTML('beforeend', '<span class="badge warn">預設排除</span>');
      box.append(label);
    }
  }

  function readConfig() {
    return {
      target: $('#target').value,
      features: [...document.querySelectorAll('#features input:checked')].map(i => i.value),
      testRatio: Number($('#testRatio').value),
      seed: Math.max(0, Math.floor(Number($('#seed').value) || 0)),
      sl: Math.min(0.5, Math.max(0.001, Number($('#sl').value) || 0.05)),
    };
  }

  function run() {
    if (!state.table) return;
    const cfg = readConfig();
    $('#testRatioOut').textContent = Math.round(cfg.testRatio * 100) + '%';
    try {
      state.result = MLR.runAnalysis(state.table, state.columns, cfg);
    } catch (err) {
      setStatus(err.message, 'error');
      $('#results').hidden = true;
      return;
    }
    setStatus(state.loadedMsg, 'ok');
    $('#results').hidden = false;
    renderAll();
    updateShareLink();
  }

  function scheduleRun() {
    clearTimeout(state.timer);
    state.timer = setTimeout(run, 120);
  }

  function updateShareLink() {
    const a = $('#shareLink');
    let qs = '';
    if (state.key !== 'custom') qs = `?problem=${state.key}`;
    else if (state.sourceIsUrl && state.source) {
      qs = `?url=${encodeURIComponent(state.source)}&target=${encodeURIComponent($('#target').value)}`;
    }
    if (state.key === 'custom' && !state.sourceIsUrl) {
      a.textContent = '（上傳的本機檔案無法產生分享連結）';
      a.removeAttribute('href');
    } else {
      const link = location.origin + location.pathname + qs;
      a.textContent = link;
      a.href = link;
    }
    history.replaceState(null, '', location.pathname + qs);
  }

  /* ============================== 呈現結果 ============================== */

  const colByName = name => state.columns.find(c => c.name === name);

  function renderAll() {
    const r = state.result;
    renderBusiness();
    renderData(r);
    renderPrep(r);
    renderModel(r);
    renderEvaluation(r);
    renderDeploy(r);
  }

  function renderBusiness() {
    const src = state.sourceIsUrl
      ? `<a href="${esc(state.source)}" target="_blank" rel="noopener">${esc(state.source)}</a>`
      : `上傳檔案 <code>${esc(state.source)}</code>`;
    $('#business').innerHTML = PRESETS[state.key].business + `<p class="note">資料來源：${src}</p>`;
  }

  function renderData(r) {
    const { table, columns } = state;
    const notes = PRESETS[state.key].notes || {};
    const used = new Set(r.cfg.features);

    $('#dataSummary').innerHTML = kpis([
      [table.rows.length, '資料筆數'],
      [table.header.length, '欄位數'],
      [columns.filter(c => c.type === 'numeric').length, '數值欄位'],
      [columns.filter(c => c.type === 'categorical').length, '類別欄位'],
      [columns.reduce((s, c) => s + c.missing, 0), '缺失值'],
      [r.design.rows.length, '建模使用筆數'],
    ]);

    $('#columnsTable').innerHTML = htmlTable(['欄位', '說明', '型態', '缺失', '唯一值', '角色'], columns.map(c => {
      const role = c.name === r.cfg.target ? '<span class="tag y">目標 y</span>'
        : used.has(c.name) ? '<span class="tag x">特徵 X</span>'
          : `<span class="tag">未使用${c.exclude ? '：' + esc(c.exclude) : ''}</span>`;
      return { cells: [esc(c.name), esc(notes[c.name] || ''), TYPE_LABEL[c.type], c.missing, c.unique, role] };
    }), [0, 1, 2, 5]);

    $('#preview').innerHTML = htmlTable(table.header.map(esc),
      table.rows.slice(0, 8).map(row => ({ cells: row.map(v => esc(v)) })));

    const numCols = columns.filter(c => c.type === 'numeric');
    $('#describe').innerHTML = htmlTable(['欄位', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
      numCols.map(c => {
        const vals = table.rows.map(row => MLR.toNumber(row[c.index])).filter(Number.isFinite);
        const d = MLR.describe(vals);
        return { cells: [esc(c.name), d.count, fmt(d.mean), fmt(d.std), fmt(d.min), fmt(d.q1), fmt(d.median), fmt(d.q3), fmt(d.max)] };
      }));

    const catCols = columns.filter(c => c.type === 'categorical');
    $('#categorical').innerHTML = catCols.map(c => {
      const counts = new Map();
      for (const row of table.rows) {
        if (MLR.isMissing(row[c.index])) continue;
        counts.set(row[c.index], (counts.get(row[c.index]) || 0) + 1);
      }
      const items = [...counts].sort((a, b) => String(a[0]).localeCompare(String(b[0]))).slice(0, 12)
        .map(([k, v]) => `${esc(k)} <b>${v}</b>`).join(' · ');
      return `<p><strong>${esc(c.name)}</strong> 類別分布：${items}${counts.size > 12 ? ' …' : ''}</p>`;
    }).join('');

    // 相關係數：使用中的數值特徵 + 目標
    const vars = [...r.cfg.features.filter(f => colByName(f).type === 'numeric'), r.cfg.target];
    const series = vars.map(v => {
      const idx = colByName(v).index;
      return r.design.rows.map(row => MLR.toNumber(row[idx]));
    });
    const M = series.map(a => series.map(b => MLR.pearson(a, b)));
    const pos = cssVar('--heat-pos'), neg = cssVar('--heat-neg');
    const head = '<thead><tr><th></th>' + vars.map(v => `<th>${esc(v)}</th>`).join('') + '</tr></thead>';
    const body = vars.map((v, i) => '<tr><td><b>' + esc(v) + '</b></td>' + M[i].map(x => {
      const a = Number.isFinite(x) ? Math.min(1, Math.abs(x)) * 0.85 : 0;
      return `<td style="background:rgba(${x >= 0 ? pos : neg},${a.toFixed(3)});${a > 0.5 ? 'color:#fff;' : ''}">${Number.isFinite(x) ? x.toFixed(2) : '—'}</td>`;
    }).join('') + '</tr>').join('');
    $('#corr').innerHTML = `<table class="heat">${head}<tbody>${body}</tbody></table>`;

    const ti = vars.length - 1;
    const ranked = vars.slice(0, -1).map((v, i) => [v, M[ti][i]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    $('#corrNote').innerHTML = ranked.length
      ? `與 <code>${esc(r.cfg.target)}</code> 相關性最高：` + ranked.slice(0, 4).map(([v, x]) => `<code>${esc(v)}</code>（${x >= 0 ? '+' : ''}${x.toFixed(3)}）`).join('、') +
        '。相關係數只看單一變數，多元回歸會同時控制其他特徵。'
      : '';
  }

  function renderPrep(r) {
    const items = [];
    const nFeat = r.cfg.features.length;
    items.push(`目標 y：<code>${esc(r.cfg.target)}</code>；特徵 X：${nFeat} 個原始欄位，展開後共 ${r.design.names.length} 個模型欄位。`);
    items.push(r.design.nDropped
      ? `缺失值：刪除 ${r.design.nDropped} 筆含缺失值（或無法轉成數字）的資料，剩 ${r.design.rows.length} 筆。`
      : `缺失值：沒有缺失值，${r.design.rows.length} 筆資料全部使用。`);
    for (const e of r.design.encodings) {
      if (e.type !== 'categorical') continue;
      items.push(`One-hot encoding：<code>${esc(e.feature)}</code> → ${codeList(e.columns)}（基準類別 <code>${esc(e.baseline)}</code>，drop first 避免 dummy variable trap）。`);
    }
    const unused = state.columns.filter(c => c.name !== r.cfg.target && c.type !== 'empty' && !r.cfg.features.includes(c.name));
    if (unused.length) {
      items.push('未放入模型：' + unused.map(c => `<code>${esc(c.name)}</code>（${esc(c.exclude || '手動取消勾選')}）`).join('、') + '。');
    }
    items.push(`訓練/測試切分：訓練集 ${r.ytr.length} 筆、測試集 ${r.yte.length} 筆（測試比例 ${Math.round(r.cfg.testRatio * 100)}%、seed ${r.cfg.seed}，mulberry32 洗牌；Python 報告用相同方法，數字一致）。`);
    items.push('係數使用原始單位（不標準化），另計算標準化係數比較不同單位特徵的影響力。');
    let html = items.map(t => `<li>${t}</li>`).join('');
    if (r.full.droppedCollinear.length) {
      html += `<li class="warn-item">完全共線、無法估計而自動移除：${codeList(r.full.droppedCollinear)}。</li>`;
    }
    $('#prep').innerHTML = html;
  }

  function equationText(model, target) {
    let s = `${target} = ${fmt(model.coefficients[0].estimate)}`;
    for (const c of model.coefficients.slice(1)) {
      s += ` ${c.estimate < 0 ? '−' : '+'} ${fmt(Math.abs(c.estimate))} × ${c.name}`;
    }
    return s;
  }

  function coefTable(model, sl, vifs, stdc) {
    const headers = ['變數', '係數', '標準誤', 't', 'p 值', '95% 信賴區間'];
    if (vifs) headers.push('VIF', '標準化係數');
    return htmlTable(headers, model.coefficients.map((c, k) => {
      const cells = [esc(c.name), fmt(c.estimate), fmt(c.se), Number.isFinite(c.t) ? c.t.toFixed(3) : '—',
        `${fmtP(c.p)} <span class="stars">${stars(c.p)}</span>`, `[${fmt(c.lower)}, ${fmt(c.upper)}]`];
      if (vifs) {
        cells.push(k ? `<span class="${vifs[k - 1] > 10 ? 'warn' : ''}">${fmt(vifs[k - 1], 4)}</span>` : '—',
          k ? stdc[k - 1].toFixed(3) : '—');
      }
      return { cells, cls: k && c.p < sl ? 'sig' : '' };
    }));
  }

  function renderModel(r) {
    const { full, reduced, cfg } = r;
    $('#eqFull').textContent = equationText(full, cfg.target);
    $('#statsFull').innerHTML = kpis([
      [f4(full.r2), 'R²'],
      [f4(full.adjR2), 'Adj R²'],
      [fmt(full.F, 4), `F（p = ${fmtP(full.Fp)}）`],
      [fmt(full.sigma, 4), '殘差標準誤 σ'],
      [fmt(full.aic, 5), 'AIC'],
      [full.n, '訓練筆數'],
    ]);
    $('#coefFull').innerHTML = coefTable(full, cfg.sl, r.vif, r.stdCoef.full);

    $('#steps').innerHTML = r.steps.length
      ? `<p>每一步移除 p 值最大且大於 SL = ${cfg.sl} 的特徵，再重新配適，直到所有特徵都顯著：</p>` +
        '<div class="table-wrap">' + htmlTable(['步驟', '移除變數', 'p 值', '移除前 R²', '移除前 Adj R²'],
          r.steps.map((s, i) => ({ cells: [i + 1, `<code>${esc(s.removed)}</code>`, fmtP(s.p), f4(s.r2), f4(s.adjR2)] })), [0, 1]) + '</div>' +
        `<p>保留 ${reduced.features.length} 個特徵：${codeList(reduced.features) || '（只剩截距）'}</p>`
      : `<p>所有特徵的 p 值都小於 SL = ${cfg.sl}，不需要移除任何特徵。</p>`;
    $('#eqReduced').textContent = equationText(reduced, cfg.target);
    $('#coefReduced').innerHTML = coefTable(reduced, cfg.sl);
  }

  function renderEvaluation(r) {
    const ev = r.evaluation;
    const hasTest = !!ev.full.test;
    const keys = ['full', 'reduced', 'sweet'];
    const minRmse = hasTest ? Math.min(...keys.map(k => ev[k].test.rmse)) : NaN;
    const isBest = key => hasTest && ev[key].test.rmse <= minRmse * (1 + 1e-12);
    const labels = {
      full: '全特徵模型',
      reduced: `逆向淘汰模型（SL = ${r.cfg.sl}）`,
      sweet: `最佳 k 模型（k = ${r.selection.best.k}）`,
    };
    $('#metrics').innerHTML = htmlTable(['模型', '特徵數', 'Train R²', 'Train Adj R²', 'Test R²', 'Test Adj R²', 'Test RMSE', 'Test MAE'],
      keys.map(key => {
        const m = r[key], tr = ev[key].train, te = ev[key].test;
        return {
          cls: key === state.view ? 'current' : '',
          cells: [labels[key] + (isBest(key) ? '<span class="best">測試 RMSE 最低</span>' : ''),
            m.features.length, f4(tr.r2), f4(tr.adjR2),
            te ? f4(te.r2) : '—', te ? f4(te.adjR2) : '—', te ? fmt(te.rmse) : '—', te ? fmt(te.mae) : '—'],
        };
      }));
    renderCharts(r);
    renderResiduals(r);
    renderSelection(r);
    $('#conclusion').innerHTML = conclusionHTML(r);
  }

  function renderResiduals(r) {
    const model = r[state.view];
    if (!r.yte.length) { $('#residTable').innerHTML = ''; return; }
    const pte = MLR.predict(model, r.Xte);
    const rows = r.yte.map((y, i) => ({ i, y, p: pte[i], e: y - pte[i] }))
      .sort((a, b) => Math.abs(b.e) - Math.abs(a.e)).slice(0, 10);
    $('#residTable').innerHTML = htmlTable(['排名', '資料列', '實際值', '預測值', '殘差', '|誤差| %', '狀態'],
      rows.map((d, rank) => {
        const z = Math.abs(d.e) / model.sigma;
        const status = z > 2 ? '<span class="warn">離群（> 2σ）</span>' : z > 1 ? '偏高（> 1σ）' : '正常';
        return {
          cells: [`#${rank + 1}`, `第 ${r.design.rowIndex[r.split.test[d.i]] + 1} 筆`, fmt(d.y), fmt(d.p),
            (d.e >= 0 ? '+' : '') + fmt(d.e), d.y !== 0 ? (Math.abs(d.e / d.y) * 100).toFixed(2) + '%' : '—', status],
        };
      }), [0, 1, 6]);
  }

  function renderSelection(r) {
    const sel = r.selection, best = sel.best, t = r.cfg.target;
    const te = best.test;
    $('#sweetSpot').innerHTML = kpis([
      [`k = ${best.k}`, '最佳特徵數'],
      [te ? fmt(te.rmse) : '—', '最低 Test RMSE'],
      [te ? f4(te.r2) : '—', 'Test R²'],
      [f4(best.train.adjR2), 'Train Adj R²'],
    ]) + `<div class="kpi" style="grid-column: 1 / -1"><b style="font-size:15px">${codeList(best.features)}</b>` +
      `<span>選出這組特徵的方法：${best.sameMethods.map(esc).join('、')}</span></div>`;
    $('#eqSweet').textContent = equationText(r.sweet, t);

    const c = palette();
    const ks = [];
    for (let k = 1; k <= sel.K; k++) ks.push(String(k));
    const lineSet = (metric, part) => sel.methods.map((m, i) => ({
      label: m.short,
      data: m.results.slice(1).map(res => (res[part] ? res[part][metric] : null)),
      borderColor: LINE_COLORS[i % LINE_COLORS.length],
      backgroundColor: LINE_COLORS[i % LINE_COLORS.length],
      pointStyle: LINE_STYLES[i % LINE_STYLES.length],
      pointRadius: 4,
      borderWidth: 2,
      tension: 0,
    }));
    const lineOptions = (yTitle, fmtY) => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: c.text, usePointStyle: true, boxWidth: 8, font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}：${fmtY(ctx.parsed.y)}` } },
      },
      scales: {
        x: { title: { display: true, text: '特徵數 k', color: c.muted }, ticks: { color: c.muted }, grid: { color: c.grid } },
        y: { title: { display: true, text: yTitle, color: c.muted }, ticks: { color: c.muted, callback: v => fmtY(v) }, grid: { color: c.grid } },
      },
    });
    const part = te ? 'test' : 'train';
    const star = metric => ({
      label: '最佳組合',
      data: ks.map((_, i) => (i === best.k - 1 ? best[part][metric] : null)),
      showLine: false,
      pointStyle: 'star', pointRadius: 12, pointHoverRadius: 13, borderWidth: 2, borderColor: c.text, backgroundColor: c.text,
    });
    makeChart('chartKRmse', {
      type: 'line',
      data: { labels: ks, datasets: [...lineSet('rmse', part), star('rmse')] },
      options: lineOptions(`${te ? 'Test' : 'Train'} RMSE`, v => fmt(v, 4)),
    });
    makeChart('chartKR2', {
      type: 'line',
      data: { labels: ks, datasets: [...lineSet('r2', part), star('r2')] },
      options: lineOptions(`${te ? 'Test' : 'Train'} R²`, v => (Number.isFinite(v) ? v.toFixed(3) : '—')),
    });

    // k × 方法的 Test RMSE 表（滑鼠移到格子上可看該組特徵）
    $('#selectionTable').innerHTML = htmlTable(['k', ...sel.methods.map(m => `<span title="${esc(m.label)}">${esc(m.short)}</span>`)],
      ks.map((_, idx) => {
        const k = idx + 1;
        return {
          cells: [`k = ${k}`, ...sel.methods.map(m => {
            const res = m.results[k];
            const isBest = res.cols.join() === best.cols.join();
            return { html: res[part] ? fmt(res[part].rmse) : '—', cls: isBest ? 'best-cell' : '', title: res.features.join(', ') };
          })],
        };
      }));

    // 每個 k 的特徵組合（最佳子集；沒有時用前向選擇）
    const ref = sel.methods.find(m => m.key === 'best') || sel.methods.find(m => m.key === 'forward');
    $('#subsetTable').innerHTML = htmlTable(['k', `${esc(ref.label)} 選出的特徵`, 'Train Adj R²', 'Test R²', 'Test RMSE'],
      ref.results.slice(1).map(res => ({
        cls: res.cols.join() === best.cols.join() ? 'sig' : '',
        cells: [res.k, { html: codeList(res.features), cls: 'left' }, f4(res.train.adjR2),
          res.test ? f4(res.test.r2) : '—', res.test ? fmt(res.test.rmse) : '—'],
      })), [0, 1]);
  }

  function conclusionHTML(r) {
    const { full, reduced, cfg, evaluation: ev } = r;
    const sig = full.coefficients.slice(1).filter(c => c.p < cfg.sl).map(c => c.name);
    const top = full.features.map((f, k) => [f, r.stdCoef.full[k]])
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 3);
    const highVif = full.features.map((f, k) => [f, r.vif[k]]).filter(([, v]) => v > 10);
    const li = [];
    li.push(`全特徵模型解釋了訓練資料 ${(full.r2 * 100).toFixed(1)}% 的變異` +
      (ev.full.test ? `；測試集 R² = ${f4(ev.full.test.r2)}、RMSE = ${fmt(ev.full.test.rmse)}。` : '。'));
    li.push(`在 α = ${cfg.sl} 下顯著的特徵：${codeList(sig) || '無'}。`);
    li.push('影響力最大（|標準化係數|）的特徵：' +
      top.map(([f, v]) => `<code>${esc(f)}</code>（${v > 0 ? '正' : '負'}向，${v >= 0 ? '+' : ''}${v.toFixed(3)}）`).join('、') + '。');
    if (ev.full.test) {
      const better = ev.reduced.test.rmse < ev.full.test.rmse;
      li.push(`逆向淘汰後保留 ${reduced.features.length} 個特徵，測試集 R² 從 ${f4(ev.full.test.r2)} 變為 ${f4(ev.reduced.test.r2)}；` +
        `以測試集 RMSE 來看${better ? '逆向淘汰模型較佳（更精簡也更準）' : '全特徵模型較佳'}。`);
      const b = r.selection.best;
      li.push(`特徵選擇比較：k = ${b.k} 時測試 RMSE 最低（${fmt(b.test.rmse)}，R² = ${f4(b.test.r2)}），特徵為 ${codeList(b.features)}` +
        `${b.k < full.features.length ? `，比全特徵模型更精簡（${full.features.length} → ${b.k} 個特徵）` : ''}。`);
    }
    li.push(highVif.length
      ? `共線性：${highVif.map(([f, v]) => `<code>${esc(f)}</code>（VIF ${v.toFixed(1)}）`).join('、')} 的 VIF > 10，係數解讀需小心。`
      : '共線性：所有特徵 VIF < 10，共線性不嚴重。');
    return '<h3>結論</h3><ul>' + li.map(t => `<li>${t}</li>`).join('') + '</ul>';
  }

  /* ============================== 圖表 ============================== */

  function palette() {
    return {
      train: cssVar('--c-train'), test: cssVar('--c-test'), pos: cssVar('--c-pos'), neg: cssVar('--c-neg'),
      muted: cssVar('--muted'), grid: cssVar('--grid'), text: cssVar('--text'),
    };
  }

  function makeChart(id, config) {
    if (state.charts[id]) { state.charts[id].destroy(); delete state.charts[id]; }
    const canvas = document.getElementById(id);
    const box = canvas.parentElement;
    box.querySelector('.fallback')?.remove();
    if (typeof Chart === 'undefined') {
      box.insertAdjacentHTML('beforeend', '<div class="fallback">圖表套件（Chart.js）載入失敗，請檢查網路後重新整理。</div>');
      return;
    }
    if (!config) return;
    state.charts[id] = new Chart(canvas, config);
  }

  function scatterOptions(xTitle, yTitle, label) {
    const c = palette();
    const axis = title => ({
      title: { display: true, text: title, color: c.muted },
      ticks: { color: c.muted, callback: v => Number(Number(v).toPrecision(8)).toLocaleString('en-US', { maximumFractionDigits: 6 }) },
      grid: { color: c.grid },
    });
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      plugins: {
        legend: { labels: { color: c.text, usePointStyle: true, boxWidth: 8 } },
        tooltip: {
          filter: item => !item.dataset.isGuide,
          callbacks: { label: ctx => label(ctx.parsed.x, ctx.parsed.y) },
        },
      },
      scales: { x: axis(xTitle), y: axis(yTitle) },
    };
  }

  const pts = (xs, ys) => xs.map((x, i) => ({ x, y: ys[i] }));

  function renderCharts(r) {
    const model = r[state.view];
    const c = palette();
    const t = r.cfg.target;
    const ptr = MLR.predict(model, r.Xtr), pte = MLR.predict(model, r.Xte);
    const [lo, hi] = extent([...r.ytr, ...r.yte, ...ptr, ...pte]);

    makeChart('chartAvP', {
      type: 'scatter',
      data: {
        datasets: [
          { label: `訓練集（${r.ytr.length}）`, data: pts(r.ytr, ptr), backgroundColor: alpha(c.train, 0.55), pointRadius: 3.5 },
          { label: `測試集（${r.yte.length}）`, data: pts(r.yte, pte), backgroundColor: c.test, pointStyle: 'rectRot', pointRadius: 5.5 },
          { label: 'y = x', data: [{ x: lo, y: lo }, { x: hi, y: hi }], showLine: true, isGuide: true, borderColor: c.muted, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
        ],
      },
      options: scatterOptions(`實際 ${t}`, `預測 ${t}`, (x, y) => `實際 ${fmt(x)}，預測 ${fmt(y)}`),
    });

    const rtr = r.ytr.map((v, i) => v - ptr[i]), rte = r.yte.map((v, i) => v - pte[i]);
    const [plo, phi] = extent([...ptr, ...pte]);
    makeChart('chartResid', {
      type: 'scatter',
      data: {
        datasets: [
          { label: '訓練集', data: pts(ptr, rtr), backgroundColor: alpha(c.train, 0.55), pointRadius: 3.5 },
          { label: '測試集', data: pts(pte, rte), backgroundColor: c.test, pointStyle: 'rectRot', pointRadius: 5.5 },
          { label: '殘差 = 0', data: [{ x: plo, y: 0 }, { x: phi, y: 0 }], showLine: true, isGuide: true, borderColor: c.muted, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
        ],
      },
      options: scatterOptions(`預測 ${t}`, '殘差（實際 − 預測）', (x, y) => `預測 ${fmt(x)}，殘差 ${fmt(y)}`),
    });

    const sc = r.stdCoef[state.view];
    if (!model.features.length) {
      makeChart('chartCoef', null);
      document.getElementById('chartCoef').parentElement.insertAdjacentHTML('beforeend', '<div class="fallback">模型只剩截距，沒有特徵可比較。</div>');
    } else {
      const order = model.features.map((_, k) => k).sort((a, b) => Math.abs(sc[b]) - Math.abs(sc[a]));
      makeChart('chartCoef', {
        type: 'bar',
        data: {
          labels: order.map(k => model.features[k]),
          datasets: [{
            data: order.map(k => sc[k]),
            backgroundColor: order.map(k => alpha(sc[k] >= 0 ? c.pos : c.neg, model.coefficients[k + 1].p < r.cfg.sl ? 0.9 : 0.35)),
            borderRadius: 4,
          }],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 250 },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: ctx => `標準化係數 ${ctx.parsed.x.toFixed(3)}，p = ${fmtP(model.coefficients[order[ctx.dataIndex] + 1].p)}` } },
          },
          scales: {
            x: { title: { display: true, text: '標準化係數（淡色 = 不顯著）', color: c.muted }, ticks: { color: c.muted }, grid: { color: c.grid } },
            y: { ticks: { color: c.text }, grid: { display: false } },
          },
        },
      });
    }
    renderScatter(r);
  }

  function renderScatter(r) {
    const sel = $('#scatterFeature');
    const feats = r.design.encodings.filter(e => e.type === 'numeric').map(e => e.feature);
    if (!feats.length) {
      sel.innerHTML = '';
      makeChart('chartScatter', null);
      document.getElementById('chartScatter').parentElement.insertAdjacentHTML('beforeend', '<div class="fallback">沒有數值特徵。</div>');
      return;
    }
    let chosen = sel.value;
    if (!feats.includes(chosen)) {
      // 預設選與目標相關性最高的特徵
      chosen = feats.map(f => {
        const j = r.design.names.indexOf(f);
        return [f, Math.abs(MLR.pearson(r.Xtr.map(row => row[j]), r.ytr))];
      }).sort((a, b) => b[1] - a[1])[0][0];
    }
    sel.innerHTML = '';
    for (const f of feats) sel.add(new Option(f, f));
    sel.value = chosen;

    const c = palette();
    const j = r.design.names.indexOf(chosen);
    makeChart('chartScatter', {
      type: 'scatter',
      data: {
        datasets: [
          { label: '訓練集', data: pts(r.Xtr.map(row => row[j]), r.ytr), backgroundColor: alpha(c.train, 0.55), pointRadius: 3.5 },
          { label: '測試集', data: pts(r.Xte.map(row => row[j]), r.yte), backgroundColor: c.test, pointStyle: 'rectRot', pointRadius: 5.5 },
        ],
      },
      options: scatterOptions(chosen, r.cfg.target, (x, y) => `${chosen} = ${fmt(x)}，${r.cfg.target} = ${fmt(y)}`),
    });
  }

  /* ============================== 部署：即時預測 ============================== */

  function renderDeploy(r) {
    const model = r[state.view];
    const form = $('#predictForm');
    form.innerHTML = '';
    const usedEnc = r.design.encodings.filter(e => e.columns.some(col => model.features.includes(col)));
    if (!usedEnc.length) form.innerHTML = '<p class="note">模型只剩截距，預測值固定為訓練集平均。</p>';
    for (const e of usedEnc) {
      const wrap = document.createElement('label');
      wrap.className = 'field';
      const title = document.createElement('span');
      title.textContent = e.feature;
      let input;
      if (e.type === 'numeric') {
        input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        const j = r.design.names.indexOf(e.feature);
        input.value = String(Number(MLR.mean(r.Xtr.map(row => row[j])).toPrecision(6)));
      } else {
        input = document.createElement('select');
        for (const lv of e.levels) input.add(new Option(lv, lv));
      }
      input.dataset.feature = e.feature;
      input.addEventListener('input', updatePrediction);
      wrap.append(title, input);
      form.append(wrap);
    }
    updatePrediction();
  }

  function updatePrediction() {
    const r = state.result, model = r[state.view];
    const values = {};
    for (const e of r.design.encodings) values[e.feature] = e.type === 'numeric' ? 0 : e.baseline;
    let invalid = false;
    document.querySelectorAll('#predictForm [data-feature]').forEach(inp => {
      if (inp.tagName === 'INPUT' && (inp.value.trim() === '' || !Number.isFinite(Number(inp.value)))) invalid = true;
      values[inp.dataset.feature] = inp.value;
    });
    if (invalid) { $('#predictOut').textContent = '請在每個欄位輸入有效的數字。'; return; }
    const pi = MLR.predictInterval(model, MLR.encodeRow(values, r.design.encodings));
    $('#predictOut').innerHTML = `預測 <code>${esc(r.cfg.target)}</code> ≈<b>${fmt(pi.yhat)}</b>` +
      `<span>95% 預測區間：${fmt(pi.lower)} ～ ${fmt(pi.upper)}（${VIEW_LABEL[state.view]}）</span>`;
  }

  /* ============================== 初始化 ============================== */

  function init() {
    const sel = $('#problem');
    for (const [k, p] of Object.entries(PRESETS)) sel.add(new Option(p.label, k));

    sel.addEventListener('change', () => selectProblem(sel.value, true));
    $('#load').addEventListener('click', loadUrl);
    $('#url').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); loadUrl(); } });
    $('#file').addEventListener('change', onFile);
    $('#target').addEventListener('change', () => { fillFeatures($('#target').value); scheduleRun(); });
    $('#features').addEventListener('change', scheduleRun);
    ['#testRatio', '#seed', '#sl'].forEach(s => $(s).addEventListener('input', scheduleRun));
    document.querySelectorAll('input[name="view"]').forEach(radio => radio.addEventListener('change', () => {
      state.view = radio.value;
      renderEvaluation(state.result);
      renderDeploy(state.result);
    }));
    $('#scatterFeature').addEventListener('change', () => renderScatter(state.result));
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.result) renderAll(); });

    const qs = new URLSearchParams(location.search);
    const url = qs.get('url'), problem = qs.get('problem');
    if (url) {
      state.key = 'custom';
      sel.value = 'custom';
      $('#url').value = url;
      state.pendingTarget = qs.get('target');
      loadUrl();
    } else {
      selectProblem(PRESETS[problem] ? problem : 'startups', true);
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
