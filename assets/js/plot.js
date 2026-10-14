/* plot.js — 輕量 SVG 圖表：特徵數 k 的多線折線圖（RMSE / R²）與實際 vs 預測散佈圖，無外部套件 */
(function (root) {
  'use strict';

  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = s => String(s).replace(/[&<>"']/g, c => ESC[c]);
  const r1 = v => Math.round(v * 10) / 10;
  // 估計文字寬度（12px 字）：全形字約 12px，其餘約 6.8px
  const textWidth = t => [...String(t)].reduce((w, ch) => w + (/[\u2E80-\uFFEF]/.test(ch) ? 12 : 6.8), 0);

  /* ------------------------------ 刻度 ------------------------------ */

  function niceStep(range, count) {
    const raw = range / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / mag;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * mag;
  }

  // 依資料範圍（含留白）回傳對齊整數刻度的值域
  function niceDomain(lo, hi, count, padRatio = 0.1) {
    if (!(hi > lo)) {
      const d = Math.abs(lo) * 0.02 || 1;
      lo -= d;
      hi += d;
    }
    const pad = (hi - lo) * padRatio;
    lo -= pad;
    hi += pad;
    const step = niceStep(hi - lo, count);
    const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step;
    const ticks = [];
    for (let i = 0; a + i * step <= b + step * 1e-9; i++) ticks.push(+(a + i * step).toPrecision(12));
    return { lo: a, hi: b, ticks, step };
  }

  // 刻度標籤：小數位數由刻度間距決定（500 → 7,500；0.005 → 0.915）
  function tickFmt(t, step) {
    const e = Math.floor(Math.log10(step));
    const mant = step / Math.pow(10, e);
    const d = Math.max(0, -e + (Math.abs(mant - 2.5) < 1e-9 ? 1 : 0));
    return t.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  /* ------------------------------ 標記 ------------------------------ */

  function polygon(pts) {
    return `<polygon points="${pts.map(p => `${r1(p[0])},${r1(p[1])}`).join(' ')}"/>`;
  }

  function regular(x, y, rad, n, rot = -90) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = ((rot + (360 / n) * i) * Math.PI) / 180;
      pts.push([x + rad * Math.cos(a), y + rad * Math.sin(a)]);
    }
    return pts;
  }

  function starPoints(x, y, outer, inner) {
    const pts = [];
    for (let i = 0; i < 10; i++) {
      const a = ((-90 + 36 * i) * Math.PI) / 180, rad = i % 2 ? inner : outer;
      pts.push([x + rad * Math.cos(a), y + rad * Math.sin(a)]);
    }
    return pts;
  }

  // r：標記半徑（直徑 ≥ 8px）
  function marker(shape, x, y, r) {
    switch (shape) {
      case 'square': {
        const s = r * 0.92;
        return `<rect x="${r1(x - s)}" y="${r1(y - s)}" width="${r1(2 * s)}" height="${r1(2 * s)}" rx="1.2"/>`;
      }
      case 'triangle': return polygon([[x, y - r * 1.3], [x + r * 1.15, y + r * 0.75], [x - r * 1.15, y + r * 0.75]]);
      case 'diamond': return polygon(regular(x, y, r * 1.3, 4));
      case 'pentagon': return polygon(regular(x, y, r * 1.15, 5));
      case 'star': return polygon(starPoints(x, y, r * 1.6, r * 0.7));
      default: return `<circle cx="${r1(x)}" cy="${r1(y)}" r="${r}"/>`;
    }
  }

  // 圖例 / 卡片用的小樣本：一段線 + 中間的標記
  function legendKey(s, w = 34, h = 14) {
    const y = h / 2;
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">` +
      `<line x1="1" y1="${y}" x2="${w - 1}" y2="${y}" style="stroke:${s.color};stroke-width:2;stroke-linecap:round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''}/>` +
      `<g style="fill:${s.hollow ? 'var(--surface)' : s.color};stroke:${s.color};stroke-width:1.6">${marker(s.shape, w / 2, y, 3.6)}</g></svg>`;
  }

  /* ------------------------------ 提示框 ------------------------------ */

  function tipEl() { return document.getElementById('tip'); }

  function showTip(html, cx, cy) {
    const tip = tipEl();
    if (!tip) return;
    tip.innerHTML = html;
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = cx + 16, y = cy + 14;
    if (x + w > innerWidth - 8) x = cx - w - 16;
    if (x < 8) x = 8;
    if (y + h > innerHeight - 8) y = Math.max(8, cy - h - 14);
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }

  function hideTip() {
    const tip = tipEl();
    if (tip) tip.hidden = true;
  }

  /* ------------------------------ 共用：尺寸與重繪 ------------------------------ */

  function mount(el, spec, draw) {
    el._spec = spec;
    el._draw = draw;
    if (!el._ro && typeof ResizeObserver !== 'undefined') {
      let last = 0;
      el._ro = new ResizeObserver(() => {
        const w = el.clientWidth;
        if (Math.abs(w - last) < 1) return;
        last = w;
        if (el._spec) el._draw(el, el._spec);
      });
      el._ro.observe(el);
    }
    draw(el, spec);
  }

  /* ========================= 多線折線圖（x = 特徵數 k） ========================= */

  // spec：{ K, series:[{key,label,color,dash,shape,hollow,values[1..K],notes[1..K]}], hidden:Set,
  //         jitter, focusK, best:{key,k}, star:{k,value,label}, yTitle, xTitle, fmt, onPick(k, key) }
  function drawLines(el, s) {
    const W = Math.max(280, el.clientWidth), H = Math.max(220, el.clientHeight);
    const shown = s.series.filter(x => !s.hidden.has(x.key));
    const vals = [];
    for (const x of shown) for (let k = 1; k <= s.K; k++) if (Number.isFinite(x.values[k])) vals.push(x.values[k]);
    if (s.star && Number.isFinite(s.star.value)) vals.push(s.star.value);
    if (!vals.length) {
      el.innerHTML = '<div class="empty">請至少開啟一種方法</div>';
      el.onpointermove = el.onpointerleave = el.onclick = null;
      return;
    }
    const dom = niceDomain(Math.min(...vals), Math.max(...vals), H < 300 ? 4 : 5);
    const labels = dom.ticks.map(t => tickFmt(t, dom.step));
    const m = { l: 26 + Math.max(...labels.map(textWidth)), r: 16, t: 34, b: 46 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const X = k => m.l + ((k - 0.5) / s.K) * pw;
    const Y = v => m.t + (1 - (v - dom.lo) / (dom.hi - dom.lo)) * ph;
    const order = s.series.map(x => x.key);
    // 防重疊水平偏移：以 k 為單位（0.06 = 相鄰 k 間距的 6%）換算成像素
    const off = key => (s.jitter ? (order.indexOf(key) - (order.length - 1) / 2) * s.jitter * (pw / s.K) : 0);
    const xs = k => X(k);

    let g = '';
    // 格線與 y 軸刻度
    dom.ticks.forEach((t, i) => {
      const y = r1(Y(t));
      g += `<line class="grid-line" x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/>`;
      g += `<text class="ax-text" x="${m.l - 8}" y="${y + 4}" text-anchor="end">${esc(labels[i])}</text>`;
    });
    // x 軸：只有整數 k
    const every = Math.ceil(s.K / Math.max(4, Math.floor(pw / 34)));
    g += `<line class="axis-line" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ph}" y2="${m.t + ph}"/>`;
    for (let k = 1; k <= s.K; k++) {
      const x = r1(X(k));
      g += `<line class="axis-line" x1="${x}" x2="${x}" y1="${m.t + ph}" y2="${m.t + ph + 5}"/>`;
      if ((k - 1) % every === 0 || k === s.K) g += `<text class="ax-text" x="${x}" y="${m.t + ph + 19}" text-anchor="middle">${k}</text>`;
    }
    g += `<text class="ax-title" x="${m.l + pw / 2}" y="${H - 6}" text-anchor="middle">${esc(s.xTitle)}</text>`;
    g += `<text class="ax-title" transform="translate(13 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">${esc(s.yTitle)}</text>`;

    // 聚焦的 k：虛線 + 標籤
    if (s.focusK) {
      const x = r1(X(s.focusK));
      g += `<line class="focus-line" x1="${x}" x2="${x}" y1="${m.t - 4}" y2="${m.t + ph}"/>`;
      const label = `k = ${s.focusK}`, w = textWidth(label) + 14;
      g += `<g class="focus-pill"><rect x="${r1(x - w / 2)}" y="${m.t - 26}" width="${w}" height="19" rx="6"/>` +
        `<text x="${x}" y="${m.t - 12.5}" text-anchor="middle">${label}</text></g>`;
    }
    g += `<line class="hover-line" x1="0" x2="0" y1="${m.t}" y2="${m.t + ph}" visibility="hidden"/>`;

    // 最佳點光暈（畫在線的下面）
    const bestSeries = s.best && shown.find(x => x.key === s.best.key);
    if (bestSeries && Number.isFinite(bestSeries.values[s.best.k])) {
      const bx = xs(s.best.k) + off(s.best.key), by = Y(bestSeries.values[s.best.k]);
      g += `<circle class="halo" cx="${r1(bx)}" cy="${r1(by)}" r="12"/>`;
      g += `<text class="best-label" x="${r1(bx)}" y="${r1(by + (by > m.t + ph - 30 ? -17 : 26))}" text-anchor="middle">Best</text>`;
    }

    // 折線
    for (const x of shown) {
      let d = '';
      for (let k = 1; k <= s.K; k++) {
        const v = x.values[k];
        if (!Number.isFinite(v)) continue;
        d += `${d ? 'L' : 'M'}${r1(xs(k) + off(x.key))},${r1(Y(v))}`;
      }
      g += `<g class="series" data-key="${x.key}"><path class="line" d="${d}" style="stroke:${x.color}" ${x.dash ? `stroke-dasharray="${x.dash}"` : ''}/></g>`;
    }
    // 標記（2px 底色外圈讓重疊的點仍可分辨）
    for (const x of shown) {
      let rings = '', marks = '';
      for (let k = 1; k <= s.K; k++) {
        const v = x.values[k];
        if (!Number.isFinite(v)) continue;
        const px = xs(k) + off(x.key), py = Y(v);
        rings += marker(x.shape, px, py, 6.2);
        marks += marker(x.shape, px, py, 4.4);
      }
      g += `<g class="series" data-key="${x.key}"><g class="ring">${rings}</g>` +
        `<g class="mk${x.hollow ? ' hollow' : ''}" style="stroke:${x.color};${x.hollow ? '' : `fill:${x.color}`}">${marks}</g></g>`;
    }

    // LassoCV ★
    if (s.star && Number.isFinite(s.star.value)) {
      const sx = X(s.star.k), sy = Y(s.star.value);
      const right = sx < m.l + pw * 0.7;
      g += `<g><polygon class="star-mark" points="${starPoints(sx, sy, 9.5, 4.2).map(p => `${r1(p[0])},${r1(p[1])}`).join(' ')}"/>` +
        `<text class="star-label" x="${r1(sx + (right ? 14 : -14))}" y="${r1(sy + 4)}" text-anchor="${right ? 'start' : 'end'}">${esc(s.star.label)}</text></g>`;
    }

    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(s.yTitle)} vs ${esc(s.xTitle)}">${g}</svg>`;

    // 互動：最近的 k → 十字線 + 提示框；點擊鎖定 k（點在標記上時一併選定方法）
    const svg = el.firstChild, hover = svg.querySelector('.hover-line');
    const locate = e => {
      const rect = svg.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W, py = ((e.clientY - rect.top) / rect.height) * H;
      if (px < m.l - 10 || px > W - m.r + 10 || py < m.t - 30 || py > m.t + ph + 10) return null;
      const k = Math.min(s.K, Math.max(1, Math.round(((px - m.l) / pw) * s.K + 0.5)));
      let near = null, nd = 14;
      for (const x of shown) {
        const v = x.values[k];
        if (!Number.isFinite(v)) continue;
        const d = Math.hypot(px - (xs(k) + off(x.key)), py - Y(v));
        if (d < nd) { nd = d; near = x.key; }
      }
      return { k, near };
    };
    el.onpointermove = e => {
      const hit = locate(e);
      if (!hit) { hover.setAttribute('visibility', 'hidden'); hideTip(); return; }
      const x = r1(X(hit.k));
      hover.setAttribute('x1', x);
      hover.setAttribute('x2', x);
      hover.setAttribute('visibility', 'visible');
      let html = `<div class="tip-title">k = ${hit.k}<span>${esc(s.yTitle)}</span></div>`;
      for (const x of shown) {
        const v = x.values[hit.k];
        html += `<div class="tip-row${hit.near === x.key ? ' on' : ''}">${legendKey(x, 24, 12)}<span class="nm">${esc(x.label)}</span><span class="val">${Number.isFinite(v) ? esc(s.fmt(v)) : '—'}</span>` +
          (x.notes && x.notes[hit.k] ? `<span class="tip-sub">${esc(x.notes[hit.k])}</span>` : '') + '</div>';
      }
      if (s.star && s.star.k === hit.k && Number.isFinite(s.star.value)) {
        html += `<div class="tip-row"><svg width="24" height="12" aria-hidden="true"><polygon class="star-mark" points="${starPoints(12, 6, 5.5, 2.4).map(p => p.map(r1).join(',')).join(' ')}"/></svg><span class="nm">${esc(s.star.label)}</span><span class="val">${esc(s.fmt(s.star.value))}</span></div>`;
      }
      showTip(html, e.clientX, e.clientY);
    };
    el.onpointerleave = () => { hover.setAttribute('visibility', 'hidden'); hideTip(); };
    el.onclick = e => {
      const hit = locate(e);
      if (hit && s.onPick) s.onPick(hit.k, hit.near);
    };
  }

  /* ========================= 散佈圖：實際 vs 預測 ========================= */

  // spec：{ groups:[{label, cls, points:[{x,y,note}] , r}], xTitle, yTitle, fmt }
  function drawScatter(el, s) {
    const W = Math.max(260, el.clientWidth), H = Math.max(220, el.clientHeight);
    const all = s.groups.flatMap(gr => gr.points).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (!all.length) { el.innerHTML = '<div class="empty">沒有資料</div>'; return; }
    const lo = Math.min(...all.map(p => Math.min(p.x, p.y))), hi = Math.max(...all.map(p => Math.max(p.x, p.y)));
    const dom = niceDomain(lo, hi, 5, 0.04);
    const labels = dom.ticks.map(t => tickFmt(t, dom.step));
    const m = { l: 26 + Math.max(...labels.map(textWidth)), r: 14, t: 30, b: 44 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const X = v => m.l + ((v - dom.lo) / (dom.hi - dom.lo)) * pw;
    const Y = v => m.t + (1 - (v - dom.lo) / (dom.hi - dom.lo)) * ph;
    let g = '';
    dom.ticks.forEach((t, i) => {
      const y = r1(Y(t)), x = r1(X(t));
      g += `<line class="grid-line" x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/>`;
      g += `<text class="ax-text" x="${m.l - 8}" y="${y + 4}" text-anchor="end">${esc(labels[i])}</text>`;
      if (i % (pw < 330 ? 2 : 1) === 0) g += `<text class="ax-text" x="${x}" y="${m.t + ph + 18}" text-anchor="middle">${esc(labels[i])}</text>`;
    });
    g += `<line class="axis-line" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ph}" y2="${m.t + ph}"/>`;
    g += `<line class="diag" x1="${r1(X(dom.lo))}" y1="${r1(Y(dom.lo))}" x2="${r1(X(dom.hi))}" y2="${r1(Y(dom.hi))}"/>`;
    g += `<text class="ax-title" x="${m.l + pw / 2}" y="${H - 6}" text-anchor="middle">${esc(s.xTitle)}</text>`;
    g += `<text class="ax-title" transform="translate(13 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">${esc(s.yTitle)}</text>`;
    // 圖例
    let lx = m.l;
    for (const gr of s.groups) {
      g += `<circle class="${gr.cls}" cx="${lx + 6}" cy="${m.t - 16}" r="${gr.r}"/><text class="legend-text" x="${lx + 15}" y="${m.t - 12}">${esc(gr.label)}</text>`;
      lx += 30 + textWidth(gr.label);
    }
    if (lx + 70 < W) g += `<line class="diag" x1="${lx + 4}" x2="${lx + 26}" y1="${m.t - 16}" y2="${m.t - 16}"/><text class="legend-text" x="${lx + 32}" y="${m.t - 12}">y = x</text>`;
    const pts = [];
    for (const gr of s.groups) {
      for (const p of gr.points) {
        const px = X(p.x), py = Y(p.y);
        pts.push({ px, py, p, gr });
        g += `<circle class="${gr.cls}" cx="${r1(px)}" cy="${r1(py)}" r="${gr.r}"/>`;
      }
    }
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(s.yTitle)} vs ${esc(s.xTitle)}">${g}</svg>`;
    const svg = el.firstChild;
    el.onpointermove = e => {
      const rect = svg.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * W, y = ((e.clientY - rect.top) / rect.height) * H;
      let best = null, bd = 14;
      for (const q of pts) {
        const d = Math.hypot(q.px - x, q.py - y);
        if (d < bd) { bd = d; best = q; }
      }
      if (!best) { hideTip(); return; }
      const res = best.p.x - best.p.y;
      showTip(`<div class="tip-title">${esc(best.gr.label)}<span>${esc(best.p.note || '')}</span></div>` +
        `<div class="tip-row"><span></span><span class="nm">實際值</span><span class="val">${esc(s.fmt(best.p.x))}</span></div>` +
        `<div class="tip-row"><span></span><span class="nm">預測值</span><span class="val">${esc(s.fmt(best.p.y))}</span></div>` +
        `<div class="tip-row"><span></span><span class="nm">殘差</span><span class="val">${res >= 0 ? '+' : ''}${esc(s.fmt(res))}</span></div>`, e.clientX, e.clientY);
    };
    el.onpointerleave = hideTip;
    el.onclick = null;
  }

  root.Plot = {
    lines: (el, spec) => mount(el, spec, drawLines),
    scatter: (el, spec) => mount(el, spec, drawScatter),
    legendKey,
    marker,
    starPoints,
    hideTip,
  };
})(typeof self !== 'undefined' ? self : this);
