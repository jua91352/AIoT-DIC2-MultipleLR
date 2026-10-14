/* app.js — 介面：任務下拉選單、任意 CSV 網址 / 上傳、CRISP-DM 逐步解析、5 種特徵選擇方法的雙軌跡折線圖 */
(function () {
  'use strict';

  const PRESETS = {
    startups: {
      label: 'Problem 1：50 Startups（預測 Profit）',
      title: '50 Startups',
      url: 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv',
      target: 'Profit',
      business: `<p>創投公司想知道新創公司的<b>支出結構</b>與<b>所在州別</b>如何影響年度利潤：研發（R&amp;D Spend）、行政（Administration）、行銷（Marketing Spend）支出與所在州（State）中，哪些因素真正影響利潤（Profit）？目標是找出<b>最精簡又準確</b>的特徵組合，作為投資判斷依據。</p>`,
      notes: {
        'R&D Spend': '研發支出', Administration: '行政支出', 'Marketing Spend': '行銷支出',
        State: '所在州（New York / California / Florida）', Profit: '年度利潤',
      },
    },
    boston: {
      label: 'Problem 2：Boston Housing（預測房價 medv）',
      title: 'Boston Housing',
      url: 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
      target: 'medv',
      business: `<p>以 1970 年代波士頓地區 506 個普查區的 13 項特徵，預測<b>自住房屋價值中位數 medv</b>（千美元），找出影響房價的關鍵因素，作為不動產估價與都市規劃的參考。註：變數 <code>b</code> 由黑人人口比例計算而來，有倫理爭議（scikit-learn 1.2 起已移除此資料集），可在「進階設定」取消勾選比較結果。</p>`,
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
      title: '自訂資料集',
      url: '',
      target: null,
      business: '<p>自訂資料集：以選擇的目標欄位為 y、勾選的欄位為 X，套用與 Problem 1、2 相同的 CRISP-DM 流程與 5 種特徵選擇方法。</p>',
      notes: {},
    },
  };

  // 每種方法固定的顏色 / 線型 / 標記（顏色依實體固定，不隨排名或篩選改變）
  const STYLE = {
    pearson: { color: 'var(--m1)', dash: '', shape: 'circle', hollow: false, desc: '實線 · 實心圓' },
    kbest: { color: 'var(--m2)', dash: '7 5', shape: 'square', hollow: true, desc: '虛線 · 空心方形' },
    rfe: { color: 'var(--m3)', dash: '11 4 2 4', shape: 'triangle', hollow: false, desc: '點虛線 · 實心三角' },
    sfs: { color: 'var(--m4)', dash: '0.5 4.5', shape: 'diamond', hollow: true, desc: '點線 · 空心菱形' },
    lasso: { color: 'var(--m5)', dash: '', shape: 'pentagon', hollow: true, desc: '實線 · 空心五邊形' },
  };

  const RULES = {
    pearson: '依訓練集 |Pearson r(x, y)| 由大到小排序，取前 k 個特徵。',
    kbest: '<code>SelectKBest(f_regression)</code>：取 F 統計量最大的 k 個。F = (n − 2)·r² / (1 − r²) 是 |r| 的單調函數，所以排序與 Pearson 完全相同。',
    rfe: '<code>RFE(LinearRegression)</code>：在標準化特徵上反覆配適，每輪移除 |係數| 最小的特徵，直到剩 k 個。',
    sfs: '<code>SequentialFeatureSelector(direction="forward", cv=KFold(5, shuffle=True))</code>：每一步加入讓 5 折交叉驗證平均 R² 最高的特徵。',
    lasso: '<code>LassoCV(cv=KFold(5, shuffle=True))</code>（標準化特徵）：以 CV 選出 α 後，依收縮後的 |係數| 由大到小取前 k 個；被壓成 0 的特徵排在最後。',
  };

  const BASIS = {
    test: { name: '測試集', rmse: 'Test RMSE', r2: 'Test R²' },
    train: { name: '訓練集', rmse: 'Train RMSE', r2: 'Train R²' },
    cv: { name: '5-fold CV', rmse: 'CV RMSE（5-fold）', r2: 'CV R²（5-fold）' },
  };

  const STEPS = ['商業理解', '資料理解', '資料準備', '建模', '評估', '部署'];
  const STEPS_EN = ['Business Understanding', 'Data Understanding', 'Data Preparation', 'Modeling', 'Evaluation', 'Deployment'];
  const TYPE_LABEL = { numeric: '數值', categorical: '類別', empty: '空' };
  const DELIM_LABEL = { ',': '逗號', ';': '分號', '\t': 'Tab', '|': '直線', whitespace: '空白' };
  const JITTER = 0.06;

  const $ = sel => document.querySelector(sel);
  const state = {
    key: 'startups', source: '', sourceIsUrl: true,
    table: null, columns: null, result: null,
    pendingTarget: null, timer: null, loadToken: 0, loadedMsg: '',
    basis: 'test', jitter: true, star: true, hidden: new Set(),
    focusK: 1, model: { method: 'pearson', k: 1 }, step: 0, tab: 'pearson', metricDigits: 2,
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
  // 誤差指標：依資料集的誤差量級固定小數位數（RMSE ≥ 100 → 2 位，否則 4 位）
  function fmtM(x) {
    if (x == null || !Number.isFinite(x)) return '—';
    const d = state.metricDigits;
    return x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  const f4 = x => (Number.isFinite(x) ? x.toFixed(4) : '—');
  const fmtP = p => (Number.isNaN(p) ? '—' : p < 1e-4 ? p.toExponential(2) : p.toFixed(4));
  const stars = p => (p < 0.001 ? '***' : p < 0.01 ? '**' : p < 0.05 ? '*' : p < 0.1 ? '.' : '');
  const codeList = names => names.map(n => `<code>${esc(n)}</code>`).join('、');
  const fchips = (names, added) => names.map(n => `<span class="fchip${n === added ? ' new' : ''}">${esc(n)}</span>`).join('');
  const checks = items => `<ul class="checks">${items.filter(Boolean).map(t => `<li>${t}</li>`).join('')}</ul>`;
  const pct = (a, b) => `${a < b ? '−' : '+'}${(Math.abs(a - b) / b * 100).toFixed(1)}%`;

  // left：靠左對齊的欄位索引；cell 可以是字串或 { html, cls, title }
  function htmlTable(headers, rows, left = [0]) {
    const L = new Set(left);
    const head = '<thead><tr>' + headers.map((h, i) => `<th${L.has(i) ? ' class="l"' : ''}>${h}</th>`).join('') + '</tr></thead>';
    const body = '<tbody>' + rows.map(r => `<tr${r.cls ? ` class="${r.cls}"` : ''}${r.attrs || ''}>` + r.cells.map((c, i) => {
      const obj = c !== null && typeof c === 'object';
      const cls = [L.has(i) ? 'l' : '', obj && c.cls ? c.cls : ''].filter(Boolean).join(' ');
      const title = obj && c.title ? ` title="${esc(c.title)}"` : '';
      return `<td${cls ? ` class="${cls}"` : ''}${title}>${obj ? c.html : c}</td>`;
    }).join('') + '</tr>').join('') + '</tbody>';
    return `<table>${head}${body}</table>`;
  }

  function setStatus(msg, kind) {
    const s = $('#status');
    s.textContent = msg;
    s.className = 'status ' + (kind || '');
  }

  const sel = () => state.result.selection;
  const methodOf = key => sel().methods.find(m => m.key === key);
  const methodIndex = key => MLR.SELECTION_METHODS.findIndex(m => m.key === key);
  const methodName = key => `${methodIndex(key) + 1}. ${methodOf(key).label}`;
  const fullRes = () => sel().methods[0].results[sel().K];
  // 選出同一組特徵的方法：「5 種方法一致」或列出方法名稱
  const sameLabel = b => (b.sameMethods.length === 5 ? '5 種方法一致' : b.sameMethods.map(k => methodOf(k).label).join('、'));

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
      $('#dsTitle').textContent = PRESETS.custom.title;
      setStatus('請貼上 CSV 網址後按「載入資料」，或上傳本機 CSV 檔。', 'info');
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
      if (table.rows.length < 15) throw new Error('資料少於 15 筆，無法比較特徵選擇方法。');
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

      const tsel = $('#target');
      tsel.innerHTML = '';
      for (const c of numeric) tsel.add(new Option(c.name, c.name));
      tsel.value = target;
      fillFeatures(target);
      $('#config').hidden = false;

      state.loadedMsg = `已載入 ${table.rows.length} 筆 × ${table.header.length} 欄（分隔符號：${DELIM_LABEL[table.delimiter]}` +
        `${table.hasHeader ? '' : '；沒有標頭，自動命名欄位'}）。可在「進階設定」調整目標、特徵與切分參數，結果會即時更新。`;
      state.step = 0;
      run(true);
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
      seed: Math.min(4294967295, Math.max(0, Math.floor(Number($('#seed').value) || 0))),
    };
  }

  function run(resetView) {
    if (!state.table) return;
    const cfg = readConfig();
    $('#testRatioOut').textContent = cfg.testRatio.toFixed(2);
    try {
      state.result = MLR.runAnalysis(state.table, state.columns, cfg);
    } catch (err) {
      setStatus(err.message, 'error');
      $('#results').hidden = true;
      return;
    }
    setStatus(state.loadedMsg, 'ok');
    $('#results').hidden = false;
    const ref = fullRes().test || fullRes().train;
    state.metricDigits = Math.abs(ref.rmse) >= 100 ? 2 : 4;
    const best = state.result.best.test || state.result.best.train;
    if (resetView || state.focusK > sel().K) {
      state.focusK = best.k;
      state.model = { method: best.method, k: best.k };
      state.tab = best.method;
    } else {
      state.model.k = Math.min(state.model.k, sel().K);
    }
    renderAll();
    updateShareLink();
  }

  function scheduleRun() {
    clearTimeout(state.timer);
    state.timer = setTimeout(() => run(false), 150);
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

  /* ============================== 呈現 ============================== */

  function renderAll() {
    const r = state.result;
    $('#dsTitle').textContent = state.key === 'custom' ? `${PRESETS.custom.title}（${r.cfg.target}）` : PRESETS[state.key].title;
    $('#dsSub').textContent = `逐步解析 5 種特徵選擇方法在 k = 1 ~ ${sel().K} 的效能軌跡（RMSE / R²）、特徵演進與模型部署`;
    renderKpis(r);
    renderSteps();
    renderMethodCards();
    renderControls();
    renderView();
    renderAppendix(r);
  }

  // 聚焦的 k / 方法改變時需要更新的部分
  function renderView() {
    renderKButtons();
    renderPlots();
    renderTable();
    renderModel();
    $('#stepFocus').textContent = `k = ${state.focusK}`;
  }

  const ICONS = {
    rows: '<path d="M4 6h16M4 12h16M4 18h16" />',
    layers: '<path d="m12 3 9 5-9 5-9-5 9-5zM3 13l9 5 9-5" />',
    target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5"/>',
    down: '<path d="M4 7l6 6 4-4 6 6M20 10v5h-5" />',
    gauge: '<path d="M4 16a8 8 0 1 1 16 0M12 16l4-5" />',
  };
  const icon = name => `<span class="kpi-ico"><svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg></span>`;

  function renderKpis(r) {
    const b = r.best.test, full = fullRes();
    const cat = r.design.encodings.some(e => e.type === 'categorical');
    const tile = (label, ico, val, sub) => `<div class="kpi"><div class="kpi-top"><span>${label}</span>${icon(ico)}</div><div class="kpi-val">${val}</div><div class="kpi-sub">${sub}</div></div>`;
    let html = tile('有效樣本數 N', 'rows', r.design.rows.length.toLocaleString('en-US'),
      `訓練 ${r.ytr.length} ／ 測試 ${r.yte.length}（random_state = ${r.cfg.seed}）`);
    html += tile('候選特徵數 K', 'layers', sel().K, cat ? 'One-hot（drop_first）展開後' : '全部為數值特徵');
    if (b) {
      html += tile('最佳特徵數', 'target', `k = ${b.k}`, `${esc(b.res.order.join('、'))}<br>${esc(sameLabel(b))}`);
      const dr = b.res.test.rmse <= full.test.rmse;
      html += tile('最低 Test RMSE', 'down', fmtM(b.res.test.rmse),
        `全特徵 ${fmtM(full.test.rmse)}（<span class="${dr ? 'delta-good' : 'delta-bad'}">${pct(b.res.test.rmse, full.test.rmse)}</span>）`);
      html += tile('對應 Test R²', 'gauge', f4(b.res.test.r2), `全特徵 ${f4(full.test.r2)}`);
    }
    $('#kpis').innerHTML = html;
  }

  /* ------------------------------ 逐步解析（CRISP-DM 6 步） ------------------------------ */

  // 方法間分歧最大的 k：不同特徵組合數最多，同數量時 Test RMSE 差距最大
  function divergentK() {
    const s = sel();
    let best = { k: 1, n: 0, spread: -1 };
    for (let k = 1; k < s.K; k++) {
      const res = s.methods.map(m => m.results[k]);
      const n = new Set(res.map(x => x.key)).size;
      const v = res.map(x => (x.test ? x.test.rmse : x.train.rmse));
      const spread = Math.max(...v) - Math.min(...v);
      if (n > best.n || (n === best.n && spread > best.spread)) best = { k, n, spread };
    }
    return best;
  }

  function stepView(i) {
    const r = state.result, s = sel(), t = esc(r.cfg.target), K = s.K;
    const b = r.best.test, full = fullRes();
    const preset = PRESETS[state.key];
    switch (i) {
      case 0: {
        const src = state.sourceIsUrl
          ? `<a href="${esc(state.source)}" target="_blank" rel="noopener">${esc(state.source.replace(/^https?:\/\//, ''))}</a>`
          : `上傳檔案 <code>${esc(state.source)}</code>`;
        return {
          title: `用最少的特徵準確預測 <code>${t}</code>`,
          html: preset.business + checks([
            `目標 y：<code>${t}</code>；${r.design.encodings.some(e => e.type === 'categorical') ? 'One-hot 後' : ''}共 <b>K = ${K}</b> 個候選特徵，比較 5 種特徵選擇方法在 <b>k = 1 ~ ${K}</b> 的表現。`,
            '成功標準：<b>Test RMSE</b> 越低越好、<b>Test R²</b> 越高越好；表現相同時選特徵較少的模型（parsimony）。',
            '視覺結構：1×2 雙軌跡折線圖（左 RMSE、右 R²）＋ 特徵演進明細表；X 軸固定為整數 k，不會出現「1.5 個特徵」的刻度。',
            `資料來源：${src}`,
          ]),
          applyDesc: '顯示全部 5 條線、使用測試集，並把焦點移到 Test RMSE 最低的 k。',
          applyLabel: '套用 Step 1 檢視（全部 5 線 + 最佳 k）',
          apply() { state.hidden.clear(); state.basis = 'test'; if (b) focus(b.k, b.method); },
        };
      }
      case 1: {
        const cols = state.columns;
        const idx = s.corr.map((_, a) => a).sort((a, c) => Math.abs(s.corr[c]) - Math.abs(s.corr[a]));
        const bars = '<div class="rbars"><div class="rbar head"><span>特徵</span><span>Pearson r（訓練集）</span><span class="v">r</span><span class="f">F</span></div>' +
          idx.map(a => {
            const v = s.corr[a], w = Number.isFinite(v) ? Math.abs(v) * 50 : 0;
            return `<div class="rbar"><span class="nm" title="${esc(s.names[a])}">${esc(s.names[a])}</span>` +
              `<span class="track"><span class="fill ${v >= 0 ? 'pos' : 'neg'}" style="width:${w.toFixed(1)}%"></span></span>` +
              `<span class="v">${Number.isFinite(v) ? (v >= 0 ? '+' : '') + v.toFixed(3) : '—'}</span><span class="f">${fmt(s.F[a], 4)}</span></div>`;
          }).join('') + '</div>';
        const top = idx[0];
        return {
          title: `每個特徵與 <code>${t}</code> 的單變數關係`,
          html: `<p>資料共 <b>${state.table.rows.length}</b> 筆、<b>${cols.length}</b> 欄（數值 ${cols.filter(c => c.type === 'numeric').length}、類別 ${cols.filter(c => c.type === 'categorical').length}），缺失值 ${cols.reduce((x, c) => x + c.missing, 0)} 個。下面是訓練集中每個候選特徵與 <code>${t}</code> 的 Pearson r 與 F 統計量，兩種 Filter 方法就是依這個排序挑特徵：</p>` +
            bars + checks([
              `單變數相關最高的是 <code>${esc(s.names[top])}</code>（r = ${s.corr[top].toFixed(3)}），所以 Filter 方法在 k = 1 一定先選它。`,
              'Pearson 依 |r| 排序、SelectKBest 依 F 排序；F = (n − 2)·r² / (1 − r²) 隨 |r| 單調遞增，兩者選出的特徵完全相同 → 圖上藍線與橘線重疊是<b>正確結果</b>，不是 bug。',
              'Filter 方法只看單變數關係，不考慮特徵之間的共線性；這正是 Wrapper / Embedded 方法要補足的地方。',
            ]),
          applyDesc: '只顯示兩種 Filter 方法（Pearson、SelectKBest），開啟防重疊偏移，確認兩條線完全重疊。',
          applyLabel: '套用 Step 2 檢視（只看 Filter 兩條線）',
          apply() {
            state.hidden = new Set(['rfe', 'sfs', 'lasso']);
            state.jitter = true;
            state.basis = 'test';
            focus(state.focusK, 'pearson');
          },
        };
      }
      case 2: {
        const items = [];
        for (const e of r.design.encodings) {
          if (e.type !== 'categorical') continue;
          items.push(`One-hot（drop_first）：<code>${esc(e.feature)}</code> → ${codeList(e.columns)}，基準類別 <code>${esc(e.baseline)}</code>。若保留全部 ${e.levels.length} 個 dummy，它們加總恆等於 1，會和截距完全共線（dummy variable trap）。`);
        }
        if (!items.length) items.push('全部特徵都是數值，不需要 One-hot。');
        items.push(r.design.nDropped
          ? `缺失值：刪除 ${r.design.nDropped} 筆含缺失值（或無法轉成數字）的資料，剩 ${r.design.rows.length} 筆。`
          : `缺失值：沒有缺失值，${r.design.rows.length} 筆全部使用。`);
        items.push(`切分：<code>train_test_split(test_size=${r.cfg.testRatio}, random_state=${r.cfg.seed})</code> → 訓練 ${r.ytr.length} 筆、測試 ${r.yte.length} 筆。網頁用 numpy 的 MT19937 重現同一個亂數排列，和 scikit-learn 逐筆相同。`);
        items.push('標準化：<code>StandardScaler</code> 只用訓練集 fit（避免資訊洩漏），給 RFE 與 Lasso 比較係數大小；每組選出的特徵最後都用 <code>LinearRegression</code> 在原始單位重新配適。');
        items.push(`交叉驗證：<code>KFold(n_splits=5, shuffle=True, random_state=${r.cfg.seed})</code>，只在訓練集內切，供 Forward SFS、LassoCV 與「5-fold CV」曲線共用。`);
        const unused = state.columns.filter(c => c.name !== r.cfg.target && c.type !== 'empty' && !r.cfg.features.includes(c.name));
        if (unused.length) items.push('未放入模型：' + unused.map(c => `<code>${esc(c.name)}</code>（${esc(c.exclude || '手動取消勾選')}）`).join('、') + '。');
        if (r.full.droppedCollinear.length) items.push(`完全共線、無法估計而移除：${codeList(r.full.droppedCollinear)}。`);
        return {
          title: 'One-hot、切分與標準化',
          html: '<p>把原始欄位整理成模型可以使用的特徵矩陣，並保留一份模型從未看過的測試集：</p>' + checks(items),
          applyDesc: '把曲線切到「訓練集」：訓練誤差隨 k 一路下降（多加特徵不會讓訓練誤差變大），所以一定要用測試集判斷。',
          applyLabel: '套用 Step 3 檢視（改看訓練集曲線）',
          apply() { state.hidden.clear(); state.basis = 'train'; focus(K, null); },
        };
      }
      case 3: {
        const list = s.methods.map(m => `<div class="method-item">${Plot.legendKey({ ...STYLE[m.key] }, 28, 14)}<div><b>${esc(methodName(m.key))}</b> <span class="fam">${m.family}</span><br>${RULES[m.key]}<div class="rank">${m.ranking.map(esc).join(' › ')}</div></div></div>`).join('');
        const d = divergentK();
        return {
          title: '5 種方法各自挑出 k 個特徵',
          html: `<p><b>Filter</b> 只看單變數統計量；<b>Wrapper</b> 反覆配適模型來挑選；<b>Embedded</b> 在正則化的過程中自動篩選。每個方法對每個 k 選出特徵後，都在訓練集用 OLS 重新配適。下面是各方法的特徵排序（前 k 個 = 選 k 個特徵的結果）：</p><div class="method-list">${list}</div>`,
          applyDesc: `顯示全部 5 條線並開啟防重疊偏移，聚焦方法分歧最大的 k = ${d.k}（${d.n} 種不同的特徵組合）。`,
          applyLabel: `套用 Step 4 檢視（聚焦 k = ${d.k}）`,
          apply() {
            state.hidden.clear();
            state.jitter = true;
            state.basis = 'test';
            const res = s.methods.map(m => m.results[d.k]);
            const worst = s.methods[res.reduce((w, x, j) => (x.test && x.test.rmse > res[w].test.rmse ? j : w), 0)];
            focus(d.k, worst.key);
          },
        };
      }
      case 4: {
        if (!b) return { title: '評估', html: '<p>沒有測試集可以評估。</p>', applyDesc: '', applyLabel: '—', apply() {} };
        const methodsText = sameLabel(b);
        const d = divergentK();
        const items = [
          `最佳組合：${esc(methodsText)}，在 <b>k = ${b.k}</b> 選出 ${codeList(b.res.order)}，Test RMSE = <b>${fmtM(b.res.test.rmse)}</b>、R² = <b>${f4(b.res.test.r2)}</b>。`,
          `全特徵模型（k = ${K}）：Test RMSE ${fmtM(full.test.rmse)}、R² ${f4(full.test.r2)} → 精簡模型的 RMSE ${b.res.test.rmse <= full.test.rmse ? '降低' : '增加'} ${(Math.abs(b.res.test.rmse - full.test.rmse) / full.test.rmse * 100).toFixed(1)}%。`,
        ];
        if (d.n > 1) {
          items.push(`方法分歧最大在 k = ${d.k}：` + s.methods.map(m => {
            const x = m.results[d.k];
            return `${esc(m.label)} 加入 <code>${esc(x.added)}</code>（${fmtM(x.test.rmse)}）`;
          }).join('；') + '。');
        }
        const l = s.lassoCV;
        if (l && l.test) {
          items.push(`LassoCV（5 折 CV 選出 α = ${fmt(l.alpha, 4)}）保留 ${l.k} 個非 0 係數：${codeList(l.features)}；直接用收縮後的係數預測，Test RMSE = ${fmtM(l.test.rmse)}、R² = ${f4(l.test.r2)}（圖上的 ★）。`);
        }
        const cvb = r.best.cv;
        if (cvb) items.push(`改用訓練集內的 5-fold CV 評估，RMSE 最低在 k = ${cvb.k}（${esc(cvb.sameMethods.map(k => methodOf(k).label).join('、'))}）。測試集只有 ${r.yte.length} 筆，結論宜同時參考 CV。`);
        return {
          title: `k = ${b.k} 的 Test RMSE 最低`,
          html: '<p>比較每個 (方法, k) 在測試集的 RMSE 與 R²，找出誤差最低、又最精簡的特徵組合：</p>' + checks(items),
          applyDesc: '回到測試集曲線、顯示 LassoCV ★，聚焦最佳 k，並切到「方法對照」表比較同一個 k 下各方法選的特徵。',
          applyLabel: `套用 Step 5 檢視（最佳 k = ${b.k}）`,
          apply() { state.hidden.clear(); state.basis = 'test'; state.star = true; focus(b.k, b.method); state.tab = 'compare'; },
        };
      }
      default: {
        const bm = b ? methodOf(b.method).results[b.k] : null;
        return {
          title: '用最佳模型即時預測',
          html: '<p>把評估後的最佳模型部署成可互動的預測工具：</p>' + checks([
            bm ? `部署模型（${esc(sameLabel(b))} · k = ${b.k}）：<code>${esc(equationText(bm.model, r.cfg.target))}</code>` : '',
            '下方「模型檢視與即時預測」可以切換任一方法與 k，拖曳滑桿即時得到預測值與 95% 預測區間。',
            '網站部署在 GitHub Pages，是純前端靜態頁面：所有計算都在瀏覽器內完成，資料不會上傳。',
            '明細表的「Python 完整程式碼」分頁可用 scikit-learn 重現網頁上的每一個數字。',
          ]),
          applyDesc: '把最佳模型載入「模型檢視」，並捲動到即時預測器。',
          applyLabel: '套用 Step 6 檢視（前往即時預測器）',
          apply() {
            if (b) focus(b.k, b.method);
            setTimeout(() => $('#model').scrollIntoView({ behavior: 'smooth', block: 'start' }), 30);
          },
        };
      }
    }
  }

  function renderSteps() {
    $('#stepTabs').innerHTML = STEPS.map((name, i) =>
      `<button class="step-tab" type="button" role="tab" data-step="${i}" aria-selected="${i === state.step}"><span class="n">${i + 1}</span>Step ${i + 1} <small>${name}</small></button>`).join('');
    const v = stepView(state.step);
    $('#stepMain').innerHTML = `<h2><span class="phase">${STEPS[state.step]} ${STEPS_EN[state.step]}</span>Step ${state.step + 1}：${v.title}</h2>${v.html}`;
    $('#stepApplyDesc').textContent = v.applyDesc;
    $('#stepApply').innerHTML = `▶ ${esc(v.applyLabel)}`;
    $('#stepCount').textContent = `${state.step + 1} / ${STEPS.length}`;
    $('#stepPrev').disabled = state.step === 0;
    $('#stepNext').disabled = state.step === STEPS.length - 1;
    $('#stepNext2').hidden = state.step === STEPS.length - 1;
    $('#stepFocus').textContent = `k = ${state.focusK}`;
  }

  function goStep(i) {
    state.step = Math.max(0, Math.min(STEPS.length - 1, i));
    renderSteps();
  }

  function applyStep() {
    stepView(state.step).apply();
    renderMethodCards();
    renderControls();
    renderView();
    const grid = $('.chart-grid');
    if (state.step !== 5 && grid.getBoundingClientRect().top > innerHeight * 0.6) grid.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  /* ------------------------------ 控制列與 k 按鈕 ------------------------------ */

  function renderMethodCards() {
    $('#methodCards').innerHTML = MLR.SELECTION_METHODS.map((m, i) => {
      const st = STYLE[m.key];
      return `<button class="mcard" type="button" data-key="${m.key}" style="--mc:${st.color}" aria-pressed="${!state.hidden.has(m.key)}" title="點擊切換顯示 / 隱藏">` +
        `<span class="mcard-top"><span class="mname"><span class="dot"></span>${i + 1}. ${esc(m.label)}</span><span class="fam">${m.family}</span></span>` +
        `<span class="mcard-key">${Plot.legendKey(st)}<span>${st.desc}</span></span></button>`;
    }).join('');
  }

  function renderControls() {
    document.querySelectorAll('input[name="basis"]').forEach(i => { i.checked = i.value === state.basis; });
    $('#jitter').checked = state.jitter;
    $('#showStar').checked = state.star;
    $('#showStar').disabled = state.basis === 'cv' || !sel().lassoCV;
  }

  function renderKButtons() {
    const best = state.result.best[state.basis];
    let html = '';
    for (let k = 1; k <= sel().K; k++) {
      html += `<button class="kbtn" type="button" data-k="${k}" aria-pressed="${k === state.focusK}">k = ${k}${best && best.k === k ? '<span class="bst">★ Best</span>' : ''}</button>`;
    }
    $('#kButtons').innerHTML = html;
  }

  // 聚焦某個 k；指定方法時同時切換明細表分頁與模型檢視
  function focus(k, method) {
    state.focusK = k;
    if (method) {
      state.model = { method, k };
      if (state.tab !== 'compare' && state.tab !== 'rank' && state.tab !== 'code') state.tab = method;
    } else if (state.model.method !== 'lassocv') {
      state.model.k = k;
    }
  }

  /* ------------------------------ 雙軌跡折線圖 ------------------------------ */

  function renderPlots() {
    const r = state.result, s = sel(), basis = state.basis;
    const best = r.best[basis];
    const series = metric => s.methods.map((m, i) => ({
      key: m.key, label: `${i + 1}. ${m.label}`, ...STYLE[m.key],
      values: m.results.map(res => (res && res[basis] ? res[basis][metric] : NaN)),
      notes: m.results.map(res => (res ? `+ ${res.added}` : '')),
    }));
    const haloKey = best && best.sameMethods.find(k => !state.hidden.has(k));
    const l = s.lassoCV;
    const star = metric => (state.star && basis !== 'cv' && l && l[basis]
      ? { k: l.k, value: l[basis][metric], label: `LassoCV (k=${l.k})` } : null);
    const onPick = (k, key) => { focus(k, key); renderView(); };
    const common = {
      K: s.K, hidden: state.hidden, jitter: state.jitter ? JITTER : 0, focusK: state.focusK,
      best: haloKey ? { key: haloKey, k: best.k } : null, xTitle: 'Number of Features (k)', onPick,
    };
    $('#rmseTitle').textContent = 'RMSE by Number of Features';
    $('#r2Title').textContent = 'R-squared by Number of Features';
    document.querySelectorAll('.chart-grid .better').forEach((el, i) => {
      el.textContent = `${BASIS[basis].name} · ${i ? 'Higher is better ↑' : 'Lower is better ↓'}`;
    });
    Plot.lines($('#plotRmse'), { ...common, series: series('rmse'), star: star('rmse'), yTitle: BASIS[basis].rmse, fmt: fmtM });
    Plot.lines($('#plotR2'), { ...common, series: series('r2'), star: star('r2'), yTitle: BASIS[basis].r2, fmt: f4 });
  }

  /* ------------------------------ 明細表 ------------------------------ */

  function renderTable() {
    const s = sel();
    const tabs = [
      ...s.methods.map(m => ({ id: m.key, html: `<span class="dot" style="background:${STYLE[m.key].color}"></span>${esc(m.label)}` })),
      { id: 'compare', html: `方法對照（k = ${state.focusK}）` },
      { id: 'rank', html: '特徵排名矩陣' },
      { id: 'code', html: '&lt;/&gt; Python 完整程式碼' },
    ];
    $('#tableTabs').innerHTML = tabs.map(t => `<button class="tab" type="button" role="tab" data-tab="${t.id}" aria-selected="${t.id === state.tab}">${t.html}</button>`).join('');
    const panel = $('#tablePanel');
    if (state.tab === 'compare') panel.innerHTML = compareTable();
    else if (state.tab === 'rank') panel.innerHTML = rankTable();
    else if (state.tab === 'code') panel.innerHTML = codePanel();
    else panel.innerHTML = methodTable(state.tab);
  }

  function methodTable(key) {
    const s = sel(), m = methodOf(key), best = state.result.best.test;
    const rows = [];
    for (let k = 1; k <= s.K; k++) {
      const res = m.results[k], prev = m.results[k - 1];
      const d = prev && res.test && prev.test ? res.test.rmse - prev.test.rmse : NaN;
      const isBest = best && res.key === best.res.key;
      rows.push({
        cls: `click${k === state.focusK ? ' focus' : ''}`,
        attrs: ` data-k="${k}" data-method="${key}"`,
        cells: [`k = ${k}${isBest ? '<span class="bestb">Best</span>' : ''}`,
          { html: fchips(res.order, res.added), cls: 'wrap' },
          f4(res.train.r2),
          res.test ? fmtM(res.test.rmse) : '—',
          res.test ? f4(res.test.r2) : '—',
          !Number.isFinite(d) ? '—' : fmtM(Math.abs(d)) === fmtM(0) ? '±0' : `<span class="${d > 0 ? 'up' : 'down'}">${d > 0 ? '▲' : '▼'} ${fmtM(Math.abs(d))}</span>`,
          fmtM(res.cv.rmse)],
      });
    }
    return `<div class="panel-title"><h3><span class="dot" style="background:${STYLE[key].color}"></span>${esc(methodName(key))}（${m.family}）— 特徵數量 k = 1 ~ ${s.K} 演進明細表</h3><p>點擊列可在圖表與模型檢視中聚焦該 k</p></div>` +
      `<p class="rule">${RULES[key]}</p>` +
      '<div class="table-wrap">' + htmlTable(['k', '選出的特徵（<span class="fchip new">新增</span>）', 'Train R²', 'Test RMSE', 'Test R²', 'ΔTest RMSE', 'CV RMSE（5-fold）'], rows, [0, 1]) + '</div>';
  }

  function compareTable() {
    const s = sel(), k = state.focusK, best = state.result.best.test;
    const rows = s.methods.map(m => {
      const res = m.results[k];
      return {
        cls: `click${m.key === state.model.method && k === state.model.k ? ' focus' : ''}`,
        attrs: ` data-k="${k}" data-method="${m.key}"`,
        cells: [`<span class="dot" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${STYLE[m.key].color};margin-right:6px"></span>${esc(methodName(m.key))}${best && res.key === best.res.key ? '<span class="bestb">Best</span>' : ''}`,
          { html: fchips(res.order, null), cls: 'wrap' },
          res.test ? fmtM(res.test.rmse) : '—', res.test ? f4(res.test.r2) : '—', f4(res.train.r2), fmtM(res.cv.rmse)],
      };
    });
    const l = s.lassoCV;
    if (l) {
      rows.push({
        cls: `click${state.model.method === 'lassocv' ? ' focus' : ''}`,
        attrs: ' data-method="lassocv"',
        cells: [`★ LassoCV 原生收縮（k = ${l.k}）`, { html: fchips(l.features, null), cls: 'wrap' },
          l.test ? fmtM(l.test.rmse) : '—', l.test ? f4(l.test.r2) : '—', f4(l.train.r2), '—'],
      });
    }
    const n = new Set(s.methods.map(m => m.results[k].key)).size;
    return `<div class="panel-title"><h3>方法對照：k = ${k} 時各方法選出的特徵</h3><p>${n === 1 ? '5 種方法選出相同的特徵組合' : `共有 ${n} 種不同的特徵組合`}</p></div>` +
      '<p class="rule">同一個 k，不同方法可能選出不同特徵；★ LassoCV 不重新配適 OLS，而是直接使用 Lasso 收縮後的係數。</p>' +
      '<div class="table-wrap">' + htmlTable(['方法', '選出的特徵', 'Test RMSE', 'Test R²', 'Train R²', 'CV RMSE（5-fold）'], rows, [0, 1]) + '</div>';
  }

  function rankTable() {
    const s = sel(), k = state.focusK;
    const pos = s.methods.map(m => new Map(m.ranking.map((f, i) => [f, i + 1])));
    const feats = s.names.slice().sort((a, b) => pos.reduce((x, p) => x + p.get(a), 0) - pos.reduce((x, p) => x + p.get(b), 0));
    const rows = feats.map(f => ({
      cells: [`<code>${esc(f)}</code>`, ...pos.map(p => {
        const v = p.get(f);
        return { html: String(v), cls: `rank-cell${v <= k ? ' top' : ''}` };
      }), (pos.reduce((x, p) => x + p.get(f), 0) / pos.length).toFixed(1)],
    }));
    return `<div class="panel-title"><h3>特徵排名矩陣（1 = 最先被選入）</h3><p>底色 = 在 k = ${k} 時會被選入</p></div>` +
      '<p class="rule">每個方法對特徵的排序；同一列數字差異越大，代表方法之間對該特徵的看法越不一致。</p>' +
      '<div class="table-wrap">' + htmlTable(['特徵', ...s.methods.map(m => esc(m.label)), '平均名次'], rows) + '</div>';
  }

  /* ------------------------------ Python 程式碼 ------------------------------ */

  function pythonCode() {
    const r = state.result, cfg = r.cfg;
    const py = JSON.stringify;
    const sep = state.table.delimiter === ',' ? '' : state.table.delimiter === 'whitespace' ? ', sep=r"\\s+"' : `, sep=${py(state.table.delimiter)}`;
    return `"""${state.key === 'custom' ? '自訂資料集' : PRESETS[state.key].title} — CRISP-DM × 5 種特徵選擇方法（k = 1 ~ K）
與網頁上的數字相同（需要 pandas、scikit-learn ≥ 1.2、matplotlib）。
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SequentialFeatureSelector, f_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

DATA = ${py(state.source)}${state.sourceIsUrl ? '' : '  # 本機上傳的檔案'}
TARGET = ${py(cfg.target)}
FEATURES = [${cfg.features.map(f => py(f)).join(", ")}]

# 2-3. 資料理解與準備：刪除缺失值、One-hot（drop_first）、切分、標準化（只用訓練集 fit）
df = pd.read_csv(DATA${sep}).dropna(subset=FEATURES + [TARGET])
X = pd.get_dummies(df[FEATURES], drop_first=True, dtype=float)
y = df[TARGET].astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=${cfg.testRatio}, random_state=${cfg.seed})
scaler = StandardScaler().fit(X_tr)
Z_tr = pd.DataFrame(scaler.transform(X_tr), columns=X.columns, index=X_tr.index)
cols, K = list(X.columns), X.shape[1]
kf = KFold(n_splits=5, shuffle=True, random_state=${cfg.seed})  # Forward SFS、LassoCV 與 CV 評估共用

# 4. 五種方法各自給出特徵排序，前 k 個 = 「選 k 個特徵」的結果
F, _ = f_regression(X_tr, y_tr)
rfe = RFE(LinearRegression(), n_features_to_select=1).fit(Z_tr, y_tr)
# tol 調小讓座標下降收斂到精確解（預設 tol=1e-4 時，相鄰 α 的 CV 誤差幾乎同分可能選到隔壁的 α）
lcv = LassoCV(cv=kf, tol=1e-10, max_iter=100_000).fit(Z_tr, y_tr)


def forward_order():
    order = []
    for k in range(1, K):
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward", cv=kf)
        chosen = X.columns[sfs.fit(X_tr, y_tr).get_support()]
        order += [c for c in chosen if c not in order]
    return order + [c for c in cols if c not in order]


ranking = {
    "Pearson": list(X_tr.corrwith(y_tr).abs().sort_values(ascending=False, kind="stable").index),
    "SelectKBest": [cols[i] for i in np.argsort(F, kind="mergesort")[::-1]],
    "RFE": [cols[i] for i in np.argsort(rfe.ranking_, kind="stable")],
    "Forward SFS": forward_order(),
    "Lasso": [cols[i] for i in np.argsort(np.abs(lcv.coef_), kind="stable")[::-1]],
}

# 5. 每個 (方法, k) 都在訓練集用 OLS 重新配適，再評估測試集與 5-fold CV
rows = []
for name, order in ranking.items():
    for k in range(1, K + 1):
        feats = order[:k]
        pred = LinearRegression().fit(X_tr[feats], y_tr).predict(X_te[feats])
        cv = cross_val_score(LinearRegression(), X_tr[feats], y_tr, cv=kf, scoring="neg_root_mean_squared_error")
        rows.append({"method": name, "k": k, "features": ", ".join(feats),
                     "test_rmse": mean_squared_error(y_te, pred) ** 0.5,
                     "test_r2": r2_score(y_te, pred), "cv_rmse": -cv.mean()})
res = pd.DataFrame(rows)
print(res.pivot(index="k", columns="method", values="test_rmse").round(4))
best = res.sort_values(["test_rmse", "k"], kind="stable").iloc[0]
print(f"最佳：{best.method} k = {best.k}（{best.features}）Test RMSE = {best.test_rmse:.4f}、R² = {best.test_r2:.4f}")

# LassoCV 原生收縮模型：直接用收縮後的係數預測（圖上的 ★）
p = lcv.predict(pd.DataFrame(scaler.transform(X_te), columns=X.columns))
print(f"LassoCV α = {lcv.alpha_:.6g}，k = {np.count_nonzero(lcv.coef_)}，"
      f"Test RMSE = {mean_squared_error(y_te, p) ** 0.5:.4f}、R² = {r2_score(y_te, p):.4f}")

# 6. 1×2 雙軌跡折線圖（5 種顏色 + 專屬線型與標記，水平微偏移避免重疊）
style = {"Pearson": ("-", "o", "full"), "SelectKBest": ("--", "s", "none"), "RFE": ("-.", "^", "full"),
         "Forward SFS": (":", "D", "none"), "Lasso": ("-", "p", "none")}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in [(axes[0], "test_rmse", "RMSE"), (axes[1], "test_r2", "R-squared")]:
    for i, (name, (ls, mk, fill)) in enumerate(style.items()):
        d = res[res.method == name]
        ax.plot(d.k + (i - 2) * ${JITTER}, d[metric], ls=ls, marker=mk, fillstyle=fill, lw=2, ms=8, label=name)
    ax.set_xticks(range(1, K + 1))
    ax.set(title=f"{title} by Number of Features", xlabel="Number of Features", ylabel=title)
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.show()
`;
  }

  function highlightPy(src) {
    const re = /("""[\s\S]*?"""|f?"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(#[^\n]*)|\b(import|from|for|in|def|return|as|if|not|else|and|or|lambda|True|False|None)\b/g;
    let out = '', last = 0, m;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index));
      out += `<span class="${m[1] ? 'tok-s' : m[2] ? 'tok-c' : 'tok-k'}">${esc(m[0])}</span>`;
      last = re.lastIndex;
    }
    return out + esc(src.slice(last));
  }

  function codePanel() {
    return '<div class="panel-title"><h3>Python 完整程式碼（scikit-learn）</h3><p>使用目前的資料集、目標、特徵與切分參數</p></div>' +
      '<p class="rule">切分、One-hot、5 種方法的排序與 LassoCV 都使用 scikit-learn 的實作；網頁在瀏覽器內重現同樣的演算法，數字可以互相對照。</p>' +
      `<div class="code-box"><button class="btn copy" type="button" id="copyCode">複製</button><pre><code>${highlightPy(pythonCode())}</code></pre></div>`;
  }

  /* ------------------------------ 模型檢視與即時預測 ------------------------------ */

  function equationText(model, target) {
    let s = `${target} = ${fmt(model.coefficients[0].estimate)}`;
    for (const c of model.coefficients.slice(1)) s += ` ${c.estimate < 0 ? '−' : '+'} ${fmt(Math.abs(c.estimate))} × ${c.name}`;
    return s;
  }

  // 目前檢視的模型：{ kind: 'ols' | 'lasso', model?, lcv?, predict(designRow), label, res }
  function currentModel() {
    const r = state.result, s = sel(), mm = state.model;
    if (mm.method === 'lassocv' && s.lassoCV) {
      const l = s.lassoCV;
      return {
        kind: 'lasso', label: `★ LassoCV 原生收縮（k = ${l.k}）`, l,
        cols: s.cols.filter((_, a) => l.coef[a] !== 0),
        predict: row => s.cols.reduce((acc, j, a) => acc + l.coefRaw[a] * row[j], l.interceptRaw),
        train: l.train, test: l.test,
      };
    }
    const m = methodOf(mm.method) || s.methods[0];
    const k = Math.min(Math.max(1, mm.k), s.K);
    const res = m.results[k];
    return {
      kind: 'ols', label: `${methodName(m.key)} · k = ${k}`, res, model: res.model, cols: res.cols,
      predict: row => MLR.predict(res.model, [row])[0], train: res.train, test: res.test,
    };
  }

  function renderModel() {
    const r = state.result, s = sel(), cm = currentModel(), t = r.cfg.target;
    const ms = $('#modelMethod');
    ms.innerHTML = s.methods.map(m => `<option value="${m.key}">${esc(methodName(m.key))}</option>`).join('') +
      (s.lassoCV ? '<option value="lassocv">★ LassoCV 原生收縮</option>' : '');
    ms.value = state.model.method;
    const ks = $('#modelK');
    ks.innerHTML = Array.from({ length: s.K }, (_, i) => `<option value="${i + 1}">${i + 1}</option>`).join('');
    ks.value = cm.kind === 'lasso' ? cm.l.k : state.model.k;
    ks.disabled = cm.kind === 'lasso';

    if (cm.kind === 'ols') {
      $('#equation').textContent = equationText(cm.model, t);
    } else {
      let eq = `${t} = ${fmt(cm.l.interceptRaw)}`;
      s.cols.forEach((j, a) => {
        const c = cm.l.coefRaw[a];
        if (c !== 0) eq += ` ${c < 0 ? '−' : '+'} ${fmt(Math.abs(c))} × ${r.design.names[j]}`;
      });
      $('#equation').textContent = eq;
    }
    const mini = (v, l) => `<div class="mini"><b>${v}</b><span>${l}</span></div>`;
    $('#modelKpis').innerHTML = mini(esc(cm.label), '目前模型') + mini(f4(cm.train.r2), 'Train R²') +
      mini(cm.test ? f4(cm.test.r2) : '—', 'Test R²') + mini(cm.test ? fmtM(cm.test.rmse) : '—', 'Test RMSE') +
      mini(cm.test ? fmtM(cm.test.mae) : '—', 'Test MAE') + mini(cm.kind === 'ols' ? fmtM(cm.res.cv.rmse) : '—', 'CV RMSE（5-fold）');

    const rowNo = idx => `第 ${r.design.rowIndex[idx] + 1} 筆`;
    Plot.scatter($('#plotAvP'), {
      groups: [
        { label: `訓練集（${r.ytr.length}）`, cls: 'dot-train', r: 3.6, points: r.Xtr.map((row, i) => ({ x: r.ytr[i], y: cm.predict(row), note: rowNo(r.split.train[i]) })) },
        { label: `測試集（${r.yte.length}）`, cls: 'dot-test', r: 5, points: r.Xte.map((row, i) => ({ x: r.yte[i], y: cm.predict(row), note: rowNo(r.split.test[i]) })) },
      ],
      xTitle: `實際 ${t}`, yTitle: `預測 ${t}`, fmt: v => fmt(v, 6),
    });
    renderPredictor(cm);
    renderCoef(cm);
  }

  function renderPredictor(cm) {
    const r = state.result, form = $('#predictForm');
    const prev = {};
    form.querySelectorAll('[data-feature]').forEach(el => { prev[el.dataset.feature] = el.dataset.value ?? el.value; });
    form.innerHTML = '';
    const used = r.design.encodings.filter(e => e.columns.some(c => cm.cols.includes(r.design.names.indexOf(c))));
    if (!used.length) form.innerHTML = '<p class="note">模型沒有特徵，預測值固定為訓練集平均。</p>';
    for (const e of used) {
      const wrap = document.createElement('div');
      wrap.className = 'slider';
      if (e.type === 'numeric') {
        const j = r.design.names.indexOf(e.feature);
        const all = r.design.X.map(row => row[j]);
        const lo = Math.min(...all), hi = Math.max(...all);
        const step = Number(((hi - lo) / 200).toPrecision(1)) || 0.01;
        const mean = MLR.mean(r.Xtr.map(row => row[j]));
        const v = prev[e.feature] != null && Number.isFinite(Number(prev[e.feature])) ? Number(prev[e.feature]) : Number(mean.toPrecision(6));
        wrap.innerHTML = `<div class="slider-top"><span>${esc(e.feature)}</span><input type="number" step="any" aria-label="${esc(e.feature)} 數值"></div>` +
          `<input type="range" aria-label="${esc(e.feature)}"><div class="slider-scale"><span>${fmt(lo, 4)}</span><span>平均 ${fmt(mean, 5)}</span><span>${fmt(hi, 4)}</span></div>`;
        const num = wrap.querySelector('input[type="number"]'), rng = wrap.querySelector('input[type="range"]');
        rng.min = String(lo);
        rng.max = String(hi);
        rng.step = String(step);
        rng.value = String(v);
        num.value = String(v);
        num.dataset.feature = e.feature;
        rng.addEventListener('input', () => { num.value = String(Number(Number(rng.value).toPrecision(8))); updatePrediction(); });
        num.addEventListener('input', () => { if (Number.isFinite(Number(num.value))) rng.value = num.value; updatePrediction(); });
      } else {
        const cur = prev[e.feature] && e.levels.includes(prev[e.feature]) ? prev[e.feature] : e.baseline;
        wrap.innerHTML = `<div class="slider-top"><span>${esc(e.feature)}</span></div><div class="cat-seg" role="radiogroup" aria-label="${esc(e.feature)}"></div>`;
        const segEl = wrap.querySelector('.cat-seg');
        segEl.dataset.feature = e.feature;
        segEl.dataset.value = cur;
        for (const lv of e.levels) {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.textContent = lv;
          btn.setAttribute('aria-pressed', String(lv === cur));
          btn.addEventListener('click', () => {
            segEl.dataset.value = lv;
            segEl.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === btn)));
            updatePrediction();
          });
          segEl.append(btn);
        }
      }
      form.append(wrap);
    }
    updatePrediction();
  }

  function updatePrediction() {
    const r = state.result, cm = currentModel();
    const values = {};
    for (const e of r.design.encodings) values[e.feature] = e.type === 'numeric' ? MLR.mean(r.Xtr.map(row => row[r.design.names.indexOf(e.feature)])) : e.baseline;
    let invalid = false;
    document.querySelectorAll('#predictForm [data-feature]').forEach(el => {
      const v = el.dataset.value ?? el.value;
      if (el.tagName === 'INPUT' && (String(v).trim() === '' || !Number.isFinite(Number(v)))) invalid = true;
      values[el.dataset.feature] = v;
    });
    const out = $('#predictOut');
    if (invalid) { out.innerHTML = '<div class="lbl">請輸入有效的數字</div>'; return; }
    const row = MLR.encodeRow(values, r.design.encodings);
    let html;
    if (cm.kind === 'ols') {
      const pi = MLR.predictInterval(cm.model, row);
      html = `<div class="lbl">預測 ${esc(r.cfg.target)}（ESTIMATED）</div><div class="val">${fmt(pi.yhat, 6)}</div>` +
        `<div class="pi">95% 預測區間：${fmt(pi.lower, 6)} ～ ${fmt(pi.upper, 6)}</div>`;
    } else {
      html = `<div class="lbl">預測 ${esc(r.cfg.target)}（ESTIMATED）</div><div class="val">${fmt(cm.predict(row), 6)}</div>` +
        '<div class="pi">Lasso 收縮模型不提供 OLS 預測區間</div>';
    }
    out.innerHTML = html;
  }

  function coefTable(model, vifs, stdc) {
    const headers = ['變數', '係數', '標準誤', 't', 'p 值', '95% 信賴區間'];
    if (vifs) headers.push('VIF');
    if (stdc) headers.push('標準化係數');
    return htmlTable(headers, model.coefficients.map((c, k) => {
      const cells = [esc(c.name), fmt(c.estimate), fmt(c.se), Number.isFinite(c.t) ? c.t.toFixed(3) : '—',
        `${fmtP(c.p)} <span class="stars">${stars(c.p)}</span>`, `[${fmt(c.lower)}, ${fmt(c.upper)}]`];
      if (vifs) cells.push(k ? `<span class="${vifs[k - 1] > 10 ? 'warn' : ''}">${fmt(vifs[k - 1], 4)}</span>` : '—');
      if (stdc) cells.push(k ? stdc[k - 1].toFixed(3) : '—');
      return { cells, cls: k && c.p < 0.05 ? 'sig' : '' };
    }));
  }

  function renderCoef(cm) {
    const r = state.result;
    if (cm.kind === 'ols') {
      $('#coefTable').innerHTML = coefTable(cm.model, null, MLR.standardizedCoefficients(cm.model, r.Xtr, r.ytr)) +
        '<p class="note" style="padding:0 11px 8px">粗體列 = p &lt; 0.05；*** p&lt;0.001、** p&lt;0.01、* p&lt;0.05、. p&lt;0.1。</p>';
    } else {
      const s = sel();
      $('#coefTable').innerHTML = htmlTable(['變數', '係數（原始單位）', '標準化係數', '狀態'],
        [{ cells: ['(Intercept)', fmt(cm.l.interceptRaw), '—', '—'] },
          ...s.cols.map((j, a) => ({
            cells: [esc(r.design.names[j]), fmt(cm.l.coefRaw[a]), cm.l.coef[a].toFixed(4), cm.l.coef[a] === 0 ? '<span class="tag">收縮為 0</span>' : '<span class="tag x">保留</span>'],
          }))]);
    }
  }

  /* ------------------------------ 附錄 ------------------------------ */

  function renderAppendix(r) {
    const { table, columns } = state;
    const notes = PRESETS[state.key].notes || {};
    const used = new Set(r.cfg.features);
    $('#columnsTable').innerHTML = htmlTable(['欄位', '說明', '型態', '缺失', '唯一值', '角色'], columns.map(c => {
      const role = c.name === r.cfg.target ? '<span class="tag y">目標 y</span>'
        : used.has(c.name) ? '<span class="tag x">特徵 X</span>'
          : `<span class="tag">未使用${c.exclude ? '：' + esc(c.exclude) : ''}</span>`;
      return { cells: [esc(c.name), esc(notes[c.name] || ''), TYPE_LABEL[c.type], c.missing, c.unique, role] };
    }), [0, 1, 2, 5]);

    $('#preview').innerHTML = htmlTable(table.header.map(esc), table.rows.slice(0, 8).map(row => ({ cells: row.map(v => esc(v)) })));

    const numCols = columns.filter(c => c.type === 'numeric');
    $('#describe').innerHTML = htmlTable(['欄位', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], numCols.map(c => {
      const vals = table.rows.map(row => MLR.toNumber(row[c.index])).filter(Number.isFinite);
      const d = MLR.describe(vals);
      return { cells: [esc(c.name), d.count, fmt(d.mean), fmt(d.std), fmt(d.min), fmt(d.q1), fmt(d.median), fmt(d.q3), fmt(d.max)] };
    }));

    const colByName = name => columns.find(c => c.name === name);
    const vars = [...r.cfg.features.filter(f => colByName(f).type === 'numeric'), r.cfg.target];
    const series = vars.map(v => r.design.rows.map(row => MLR.toNumber(row[colByName(v).index])));
    const M = series.map(a => series.map(b => MLR.pearson(a, b)));
    const pos = getComputedStyle(document.documentElement).getPropertyValue('--heat-pos').trim();
    const neg = getComputedStyle(document.documentElement).getPropertyValue('--heat-neg').trim();
    const head = '<thead><tr><th></th>' + vars.map(v => `<th>${esc(v)}</th>`).join('') + '</tr></thead>';
    const body = vars.map((v, i) => '<tr><td><b>' + esc(v) + '</b></td>' + M[i].map(x => {
      const a = Number.isFinite(x) ? Math.min(1, Math.abs(x)) * 0.85 : 0;
      return `<td style="background:rgba(${x >= 0 ? pos : neg},${a.toFixed(3)});${a > 0.5 ? 'color:#fff;' : ''}">${Number.isFinite(x) ? x.toFixed(2) : '—'}</td>`;
    }).join('') + '</tr>').join('');
    $('#corr').innerHTML = `<table class="heat">${head}<tbody>${body}</tbody></table>`;

    const f = r.full;
    $('#fullStats').innerHTML = `R² = ${f4(f.r2)}、Adj R² = ${f4(f.adjR2)}、F = ${fmt(f.F, 4)}（p = ${fmtP(f.Fp)}）、殘差標準誤 σ = ${fmt(f.sigma, 4)}、訓練筆數 ${f.n}。VIF &gt; 10 代表共線性明顯。`;
    $('#fullCoef').innerHTML = coefTable(f, r.vif, MLR.standardizedCoefficients(f, r.Xtr, r.ytr));
  }

  /* ============================== 初始化 ============================== */

  function init() {
    const ps = $('#problem');
    for (const [k, p] of Object.entries(PRESETS)) ps.add(new Option(p.label, k));

    ps.addEventListener('change', () => selectProblem(ps.value, true));
    $('#load').addEventListener('click', loadUrl);
    $('#url').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); loadUrl(); } });
    $('#file').addEventListener('change', onFile);
    $('#target').addEventListener('change', () => { fillFeatures($('#target').value); state.step = 0; run(true); });
    $('#features').addEventListener('change', () => scheduleRun());
    ['#testRatio', '#seed'].forEach(s => $(s).addEventListener('input', () => scheduleRun()));

    $('#showAll').addEventListener('click', () => {
      if (!state.result) return;
      state.hidden.clear();
      renderMethodCards();
      renderPlots();
    });
    $('#methodCards').addEventListener('click', e => {
      const card = e.target.closest('.mcard');
      if (!card) return;
      const k = card.dataset.key;
      if (state.hidden.has(k)) state.hidden.delete(k);
      else state.hidden.add(k);
      renderMethodCards();
      renderPlots();
    });
    document.querySelectorAll('input[name="basis"]').forEach(i => i.addEventListener('change', () => {
      state.basis = i.value;
      renderControls();
      renderKButtons();
      renderPlots();
    }));
    $('#jitter').addEventListener('change', e => { state.jitter = e.target.checked; renderPlots(); });
    $('#showStar').addEventListener('change', e => { state.star = e.target.checked; renderPlots(); });
    $('#kButtons').addEventListener('click', e => {
      const b = e.target.closest('.kbtn');
      if (!b) return;
      focus(Number(b.dataset.k), null);
      renderView();
    });

    $('#stepTabs').addEventListener('click', e => {
      const b = e.target.closest('.step-tab');
      if (b) goStep(Number(b.dataset.step));
    });
    $('#stepPrev').addEventListener('click', () => goStep(state.step - 1));
    $('#stepNext').addEventListener('click', () => goStep(state.step + 1));
    $('#stepNext2').addEventListener('click', () => goStep(state.step + 1));
    $('#stepApply').addEventListener('click', applyStep);

    $('#tableTabs').addEventListener('click', e => {
      const b = e.target.closest('.tab');
      if (!b) return;
      state.tab = b.dataset.tab;
      if (sel().methods.some(m => m.key === state.tab)) state.model = { method: state.tab, k: state.focusK };
      renderTable();
      renderModel();
    });
    $('#tablePanel').addEventListener('click', e => {
      if (e.target.id === 'copyCode') {
        const code = pythonCode();
        const done = () => { e.target.textContent = '已複製 ✓'; setTimeout(() => { e.target.textContent = '複製'; }, 1600); };
        if (navigator.clipboard) navigator.clipboard.writeText(code).then(done, () => { e.target.textContent = '複製失敗'; });
        return;
      }
      const tr = e.target.closest('tr[data-method]');
      if (!tr) return;
      if (tr.dataset.method === 'lassocv') {
        state.model = { method: 'lassocv', k: sel().lassoCV.k };
        renderTable();
        renderModel();
        return;
      }
      focus(Number(tr.dataset.k), tr.dataset.method);
      renderView();
    });

    $('#modelMethod').addEventListener('change', e => {
      const v = e.target.value;
      if (v === 'lassocv') state.model = { method: 'lassocv', k: sel().lassoCV.k };
      else {
        state.model = { method: v, k: state.model.method === 'lassocv' ? state.focusK : state.model.k };
        if (state.tab !== 'compare' && state.tab !== 'rank' && state.tab !== 'code') state.tab = v;
      }
      renderTable();
      renderModel();
    });
    $('#modelK').addEventListener('change', e => {
      focus(Number(e.target.value), state.model.method);
      renderView();
    });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.result) renderAppendix(state.result); });

    const qs = new URLSearchParams(location.search);
    const url = qs.get('url'), problem = qs.get('problem');
    if (url) {
      state.key = 'custom';
      ps.value = 'custom';
      $('#url').value = url;
      state.pendingTarget = qs.get('target');
      loadUrl();
    } else {
      selectProblem(PRESETS[problem] ? problem : 'startups', true);
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
