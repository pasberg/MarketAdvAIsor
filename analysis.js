/* MarketAdvAIsor analysis rules — shared by the web page and the pick logger (Node).
 * Everything here is pure: candles in, levels / scores / trade outcomes out.
 * Candle: {time, open, high, low, close, vol}; time = Stockholm wall clock as UTC epoch seconds.
 */
(function (root) {
  'use strict';
  const HZ_ORDER = ['intradag', 'vecka', 'manad', 'kvartal', 'halvar', 'ar'];
  // weights for [TA, FA, Elliott, momentum, Fibonacci, sentiment] per horizon
  const WEIGHTS = {
    intradag: [.30, .02, .13, .30, .15, .10], vecka: [.28, .07, .18, .22, .15, .10], manad: [.24, .16, .20, .16, .14, .10],
    kvartal: [.18, .30, .18, .12, .12, .10], halvar: [.14, .38, .18, .10, .10, .10], ar: [.12, .42, .18, .08, .10, .10],
  };
  // chart series and window the live analysis uses per horizon
  const SERIES = { intradag: 'intra', vecka: 'hour', manad: 'day', kvartal: 'day', halvar: 'week', ar: 'week' };
  const WINDOW = { intradag: 420, vecka: 90, manad: 90, kvartal: 160, halvar: 80, ar: 130 };
  // backtest: ser = series; W = bars the analysis sees; K = max bars held; E = bars the entry order stays open
  const BT = {
    intradag: { ser: 'intra5', W: 24, E: 12, session: true }, vecka: { ser: 'hour', W: 90, K: 35, E: 9 },
    manad: { ser: 'day', W: 90, K: 21, E: 5 }, kvartal: { ser: 'day', W: 160, K: 63, E: 15 },
    halvar: { ser: 'week', W: 80, K: 26, E: 6 }, ar: { ser: 'week', W: 130, K: 52, E: 12 },
  };
  // follow-up of logged picks: same limits, measured on the finest series kept long enough
  const EVAL = {
    intradag: { ser: 'intra5', E: 12, session: true }, vecka: { ser: 'hour', K: 35, E: 9 },
    manad: { ser: 'day', K: 21, E: 5 }, kvartal: { ser: 'day', K: 63, E: 15 },
    halvar: { ser: 'day', K: 130, E: 30 }, ar: { ser: 'day', K: 260, E: 60 },
  };
  // rule variants compared in the backtest
  const VARIANTS = {
    std: { name: 'Standard', desc: 'Mål vid toppen och 127,2 / 161,8 %; fast stop loss', tp: 'swing' },
    rtp: { name: 'Närmare mål', desc: 'Mål vid 1R, 2R och 3R', tp: 'r' },
    trail: { name: 'Flyttad stop', desc: 'Stop till entry efter TP1, till TP1 efter TP2', tp: 'swing', trail: true },
    trend: { name: 'Trendfilter', desc: 'Bara affärer i 200-periodersmedelvärdets riktning', tp: 'swing', trend: true },
    combo: { name: 'Kombinerad', desc: 'Närmare mål + flyttad stop + trendfilter', tp: 'r', trail: true, trend: true },
  };
  // default trading cost per horizon, % of the position round trip (spread + courtage)
  const COST = { intradag: .08, vecka: .10, manad: .15, kvartal: .15, halvar: .15, ar: .15 };

  const dayOf = t => Math.floor(t / 86400);
  function ema(vals, p) { const k = 2 / (p + 1), o = []; vals.forEach((v, i) => o.push(i ? o[i - 1] + k * (v - o[i - 1]) : v)); return o; }
  function toCandles(ser, from, to) {
    const c = []; for (let i = from; i < to; i++) c.push({ time: ser.t[i], open: ser.o[i], high: ser.h[i], low: ser.l[i], close: ser.c[i], vol: ser.v[i] }); return c;
  }
  function indicators(c, N) {
    const closes = c.map(x => x.close);
    const e20 = ema(closes, 20), e50 = ema(closes, 50), e12 = ema(closes, 12), e26 = ema(closes, 26);
    const macd = e12.map((v, i) => v - e26[i]), sig = ema(macd, 9);
    let pv = 0, vv = 0; const vwap = c.map(x => { const tp = (x.high + x.low + x.close) / 3, v = x.vol || 1; pv += tp * v; vv += v; return pv / vv; });
    const rsi = []; let ag = 0, al = 0;
    for (let i = 1; i < N; i++) {
      const d = closes[i] - closes[i - 1], g = Math.max(d, 0), l = Math.max(-d, 0);
      if (i <= 14) { ag += g / 14; al += l / 14; rsi[i] = i === 14 ? 100 - 100 / (1 + ag / (al || 1e-9)) : null; }
      else { ag = (ag * 13 + g) / 14; al = (al * 13 + l) / 14; rsi[i] = 100 - 100 / (1 + ag / (al || 1e-9)); }
    }
    return { e20, e50, vwap, rsi, macd, sig };
  }

  // Levels and computed scores from a candle window (only data up to the last candle is used).
  // item: {s, dir 'L'|'S', sc:[6 case scores], kind, status}
  function analyzeCandles(c, item, hz) {
    const N = c.length, s = item.dir === 'L' ? 1 : -1, P = c[N - 1].close;
    // impulse in the pick's direction: extreme of the window and the opposite extreme before it
    const hiI = a => a.reduce((b, x, i) => x.high > a[b].high ? i : b, 0), loI = a => a.reduce((b, x, i) => x.low < a[b].low ? i : b, 0);
    let endI = s > 0 ? hiI(c) : loI(c); if (endI < 5) endI = s > 0 ? hiI(c.slice(0, N - 3)) : loI(c.slice(0, N - 3));
    const head = c.slice(0, endI + 1), startI = s > 0 ? loI(head) : hiI(head);
    const A = s > 0 ? c[startI].low : c[startI].high, B = s > 0 ? c[endI].high : c[endI].low, R = Math.abs(B - A) || P * .01;
    const val = u => A + s * u * R, curU = (P - A) / (s * R);
    const opts = [.382, .5, .618].map(r => ({ r, eU: 1 - r })).filter(o => o.eU <= curU + .05);
    const pick = opts.length ? opts.reduce((a, b) => Math.abs(b.eU - curU) < Math.abs(a.eU - curU) ? b : a) : { r: .618, eU: .382 };
    const eU = pick.eU, slU = eU - .24;
    const ind = indicators(c, N);
    const lv = { entry: val(eU), zLo: Math.min(val(eU - .03), val(eU + .03)), zHi: Math.max(val(eU - .03), val(eU + .03)), sl: val(slU), tp1: val(1), tp2: val(1.272), tp3: val(1.618), w1: val(0) };
    const risk = Math.abs(lv.entry - lv.sl), rr = [lv.tp1, lv.tp2, lv.tp3].map(v => Math.abs(v - lv.entry) / risk);
    // scores computed from the data: TA, momentum, Fibonacci; the rest stays from the case
    const e20 = ind.e20[N - 1], e50 = ind.e50[N - 1], slope = ind.e50[N - 1] - ind.e50[Math.max(0, N - 6)], rsiNow = Math.round(ind.rsi[N - 1] ?? 50);
    const macdUp = ind.macd[N - 1] > ind.sig[N - 1];
    const ta = 50 + (s * (P - e20) > 0 ? 12 : 0) + (s * (P - e50) > 0 ? 12 : 0) + (s * (e20 - e50) > 0 ? 14 : 0) + (s * slope > 0 ? 8 : 0);
    const rsiOk = s > 0 ? (rsiNow >= 38 && rsiNow <= 68) : (rsiNow >= 32 && rsiNow <= 62), rsiRise = s * ((ind.rsi[N - 1] ?? 50) - (ind.rsi[N - 4] ?? 50)) > 0;
    const mom = 48 + ((macdUp === (s > 0)) ? 18 : 0) + (rsiOk ? 16 : 0) + (rsiRise ? 10 : 0);
    const fib = Math.round(Math.max(40, Math.min(92, 90 - Math.abs(curU - eU) * 220)));
    const sc = [ta, item.sc[1], item.sc[2], mom, fib, item.sc[5]];
    const score = Math.round(sc.reduce((a, v, i) => a + v * WEIGHTS[hz][i], 0));
    const prob = Math.max(30, Math.min(78, Math.round(35 + (score - 50) * .9))), ev = (prob / 100) * rr[1] - (1 - prob / 100);
    const gap = curU - eU, status = item.kind !== 'watch' ? null : Math.abs(gap) < .06 ? 'near' : gap > .15 ? 'wait' : 'build';
    const it = Object.assign({}, item, { ret: pick.r });
    if (item.auto) Object.assign(it, describe({ s, P, A, B, curU, ret: pick.r, entry: lv.entry, e20, e50, rsiNow, macdUp, sym: item.s }));
    return {
      item: it, s, P, R, val, c, ...ind, lv, rr, score, prob, ev, risk, eU, hz, N, sc,
      curU, rsiNow, e20Now: e20, e50Now: e50, vwapNow: ind.vwap[N - 1], macdUp, status: status || item.status, A, B, startI, endI,
      valid: s * (P - lv.sl) > 0, // false: the price has already passed the stop loss
    };
  }

  // Setup, thesis and risk text for an automatic candidate (no hand-written case), from the numbers alone.
  const RET_TXT = { 0.382: '38,2 %', 0.5: '50 %', 0.618: '61,8 %' };
  function describe(x) {
    const fmt = v => v.toLocaleString('sv-SE', { minimumFractionDigits: 2, maximumFractionDigits: Math.abs(v) < 20 ? 4 : 2 });
    const L = x.s > 0, withTrend = (x.e20 > x.e50) === L, pct = RET_TXT[x.ret];
    const rel = v => (x.P > v ? 'över' : 'under');
    return {
      setup: `${pct}-rekyl ${withTrend ? (L ? 'i upptrend' : 'i nedtrend') : 'mot trenden'}`,
      text: `Automatiskt urval. Impulsen ${fmt(x.A)} → ${fmt(x.B)} har rekylerat ${Math.round((1 - x.curU) * 100)} %; entry vid ${pct}-nivån ${fmt(x.entry)}. ` +
        `Priset ligger ${rel(x.e20)} EMA 20 och ${rel(x.e50)} EMA 50, RSI ${x.rsiNow}, MACD ${x.macdUp ? 'över' : 'under'} signallinjen. ` +
        `Det finns inget manuellt case för ${x.sym}, så fundamental-, Elliott- och sentimentpoängen är neutrala (50).`,
      risk: 'Automatiskt förslag utan manuell granskning — kontrollera nyheter, rapportdatum och spread innan du handlar.',
    };
  }

  // Live analysis of one instrument from the data files ({symbols: {SYM: {series: {...}}}}).
  function liveAnalysis(symbols, item, hz) {
    const ser = symbols && symbols[item.s] && symbols[item.s].series[SERIES[hz]];
    if (!ser || ser.c.length < 30) return null;
    const n = Math.min(WINDOW[hz], ser.c.length);
    return analyzeCandles(toCandles(ser, ser.c.length - n, ser.c.length), item, hz);
  }
  // Case rows -> items. cases[hz] = {top: [[sym, dir, ret, setup, text, scores, risk]], watch: [[..., status]]}
  // With symbols (the price data), every other instrument with enough data for the horizon is added
  // as an automatic candidate, once long and once short, with neutral scores for what only a case can
  // judge (fundamentals, Elliott waves, sentiment).
  const NEUTRAL = [50, 50, 50, 50, 50, 50];
  function pool(cases, hz, symbols) {
    const d = cases[hz];
    const mk = (a, kind) => ({ s: a[0], dir: a[1], ret: a[2], setup: a[3], text: a[4], sc: a[5], risk: kind === 'top' ? a[6] : null, status: kind === 'watch' ? a[6] : null, kind });
    const top = d.top.map(a => mk(a, 'top')), watch = d.watch.map(a => mk(a, 'watch'));
    if (symbols) {
      const cased = new Set(top.map(x => x.s));
      for (const [s, x] of Object.entries(symbols)) {
        const ser = x.series && x.series[SERIES[hz]];
        if (cased.has(s) || !ser || ser.c.length < 30) continue;
        for (const dir of ['L', 'S']) top.push({ s, dir, ret: 0.5, setup: '', text: '', sc: NEUTRAL, risk: null, status: null, kind: 'top', auto: true });
      }
    }
    return { top, watch };
  }
  // one analysis per instrument: the direction with the higher score
  function bestPerSymbol(ms) {
    const by = new Map();
    for (const m of ms) { const b = by.get(m.item.s); if (!b || m.score > b.score) by.set(m.item.s, m); }
    return [...by.values()];
  }
  // The recommendations as the page shows them (all markets): best 5 top candidates by score.
  function topPicks(cases, symbols, hz, n = 5) {
    const ms = pool(cases, hz, symbols).top.map(it => liveAnalysis(symbols, it, hz)).filter(m => m && m.valid);
    return bestPerSymbol(ms).sort((a, b) => b.score - a.score).slice(0, n);
  }

  // Follow one trade plan from signal bar t. cfg: {E, K, session}; opts: {tp: 'swing'|'r', trail}.
  // state: 'missed' (no fill / invalidated), 'pending' (entry window not over yet),
  //        'open' (filled, limit not reached in the data), 'closed'.
  function simulateTrade(all, t, lv, s, cfg, opts) {
    opts = opts || {};
    const n = all.length;
    let last = cfg.session ? t : Math.min(n - 1, t + cfg.K), limitReached = cfg.session ? false : t + cfg.K <= n - 1;
    if (cfg.session) { while (last < n - 1 && dayOf(all[last + 1].time) === dayOf(all[t].time)) last++; limitReached = last < n - 1; }
    let j = -1, fill = 0;
    const eEnd = Math.min(t + cfg.E, last);
    for (let k = t + 1; k <= eEnd; k++) {
      const b = all[k];
      if (s * (b.open - lv.sl) <= 0) return { state: 'missed', endI: k }; // opened beyond the stop: setup no longer valid
      if (s > 0 ? b.low <= lv.entry : b.high >= lv.entry) { j = k; fill = s > 0 ? Math.min(lv.entry, b.open) : Math.max(lv.entry, b.open); break; }
    }
    if (j < 0) return { state: (eEnd < t + cfg.E && !limitReached) ? 'pending' : 'missed', endI: eEnd };
    const risk = Math.abs(lv.entry - lv.sl);
    const tps = opts.tp === 'r' ? [1, 2, 3].map(k => lv.entry + s * k * risk) : [lv.tp1, lv.tp2, lv.tp3];
    const R = []; let hit = 0, stop = lv.sl, exitI = -1, res = 'tid';
    for (let k = j; k <= last && R.length < 3; k++) {
      const b = all[k];
      if (s > 0 ? b.low <= stop : b.high >= stop) { // stop first when both are touched in one bar (conservative)
        const px = s > 0 ? Math.min(stop, b.open) : Math.max(stop, b.open);
        while (R.length < 3) R.push(s * (px - fill) / risk); exitI = k; res = hit ? 'tp' : 'sl'; break;
      }
      while (hit < 3 && (s > 0 ? b.high >= tps[hit] : b.low <= tps[hit])) { R.push(s * (tps[hit] - fill) / risk); hit++; exitI = k; res = 'tp'; }
      if (opts.trail) stop = hit >= 2 ? tps[0] : hit >= 1 ? fill : lv.sl;
    }
    if (R.length < 3) {
      if (!limitReached) return { state: 'open', fillI: j, fill, hit, risk, endI: n - 1 };
      const px = all[last].close; while (R.length < 3) R.push(s * (px - fill) / risk); exitI = last; if (!hit) res = 'tid';
    }
    return { state: 'closed', fillI: j, exitI, fill, risk, hit, res, r: (R[0] + R[1] + R[2]) / 3, bars: exitI - j + 1 };
  }

  // Replay the rules over a whole series. opts: {minScore, tp, trail, trend}. memo caches analyses per bar.
  function backtest(ser, item, hz, opts, memo) {
    const cfg = BT[hz];
    if (!ser || ser.c.length < cfg.W + 5) return null;
    const all = toCandles(ser, 0, ser.c.length), n = all.length, trades = [];
    const cum = [0]; for (const x of all) cum.push(cum[cum.length - 1] + x.close);
    const sma = (t, p) => { const a = Math.max(0, t - p + 1); return (cum[t + 1] - cum[a]) / (t + 1 - a); };
    let missed = 0, t = cfg.W - 1;
    while (t < n - 1) {
      let a = memo && memo[t];
      if (a === undefined) {
        let from = t - cfg.W + 1;
        if (cfg.session) { from = t; while (from > 0 && dayOf(all[from - 1].time) === dayOf(all[t].time)) from--; }
        if (t - from + 1 < cfg.W) a = null;
        else { const m = analyzeCandles(all.slice(from, t + 1), item, hz); a = { score: m.score, s: m.s, P: m.P, lv: m.lv }; }
        if (memo) memo[t] = a;
      }
      if (!a || a.score < opts.minScore || a.s * (a.P - a.lv.sl) <= 0 || (opts.trend && a.s * (all[t].close - sma(t, 200)) <= 0)) { t++; continue; }
      const r = simulateTrade(all, t, a.lv, a.s, cfg, opts);
      if (r.state === 'missed') { missed++; t = Math.max(t + 1, r.endI); continue; }
      if (r.state !== 'closed') break; // the data ends before the trade is over
      trades.push({ s: item.s, dir: item.dir, t0: all[t].time, t1: all[r.exitI].time, entry: r.fill, sl: a.lv.sl, risk: r.risk, r: r.r, hit: r.hit, res: r.res, bars: r.bars, score: a.score });
      t = r.exitI + 1;
    }
    return { trades, missed, from: all[cfg.W - 1].time, to: all[n - 1].time, firstClose: all[cfg.W - 1].close, lastClose: all[n - 1].close };
  }
  // cost in R for a trade: % of position round trip, relative to the planned risk
  const costR = (trade, pct) => pct / 100 * trade.entry / trade.risk;

  const api = { HZ_ORDER, WEIGHTS, SERIES, WINDOW, BT, EVAL, VARIANTS, COST, dayOf, ema, toCandles, indicators, analyzeCandles, describe, liveAnalysis, pool, bestPerSymbol, topPicks, simulateTrade, backtest, costR };
  root.MAA = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
