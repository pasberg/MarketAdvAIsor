#!/usr/bin/env node
/* Measure how the site's own rules did, per horizon and score band, so the page can show a
 * measured hit rate instead of a rule of thumb.
 *
 * For every instrument (long and short, neutral case scores, as the automatic candidates) the
 * rules are replayed over the history (mockup/analysis.js backtest, standard variant). Intraday
 * uses the 5-minute archive (data/archive_bars.json, written by intraday_archive.py) when present.
 * Per score band: filled trades, share reaching TP1, share with a gain after costs, mean R after costs.
 *
 * Usage: node pipeline/calibrate.js [--data data]   -> data/calib.json
 */
'use strict';
const fs = require('fs');
const path = require('path');
const MAA = require(path.join(__dirname, '..', 'mockup', 'analysis.js'));
const { mergeSymbols } = require('./log_picks.js');

const BANDS = [[0, 60], [60, 65], [65, 70], [70, 75], [75, 80], [80, 101]];
const readJSON = f => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { return null; } };

function band(trades, lo, hi) {
  const a = trades.filter(t => t.score >= lo && t.score < hi), n = a.length;
  if (!n) return { lo, hi, n: 0 };
  const avg = a.reduce((x, t) => x + t.rn, 0) / n;
  return { lo, hi, n, tp1: a.filter(t => t.hit >= 1).length / n, win: a.filter(t => t.rn > 0).length / n, avgR: avg,
    avgRgross: a.reduce((x, t) => x + t.r, 0) / n };
}

function calibrate(symbols, hz) {
  const cfg = MAA.BT[hz], trades = [];
  let from = Infinity, to = 0, missed = 0;
  for (const [s, x] of Object.entries(symbols)) {
    const ser = x.series && x.series[cfg.ser];
    if (!ser || !ser.t || ser.t.length < cfg.W + 5) continue;
    for (const dir of ['L', 'S']) {
      const r = MAA.backtest(ser, { s, dir, sc: [50, 50, 50, 50, 50, 50], auto: true, kind: 'top' }, hz, { minScore: 0, tp: 'swing' }, {});
      if (!r) continue;
      from = Math.min(from, r.from); to = Math.max(to, r.to); missed += r.missed;
      for (const t of r.trades) trades.push({ score: t.score, hit: t.hit, r: t.r, rn: t.r - MAA.costR(t, MAA.COST[hz]) });
    }
  }
  return { n: trades.length, missed, from: isFinite(from) ? from : null, to: to || null, cost: MAA.COST[hz],
    bands: BANDS.map(([lo, hi]) => band(trades, lo, hi)), all: band(trades, 0, 101) };
}

function main() {
  const i = process.argv.indexOf('--data'), dir = i > 0 ? process.argv[i + 1] : 'data';
  const symbols = mergeSymbols(['intra', 'hour', 'daily'].map(p => readJSON(path.join(dir, p + '.json'))));
  const arch = readJSON(path.join(dir, 'archive_bars.json'));
  if (arch) for (const [s, ser] of Object.entries(arch.symbols || {})) (symbols[s] || (symbols[s] = { series: {} })).series.intra5 = ser;
  const out = { part: 'calib', generated: new Date().toISOString(), source: arch ? 'archive' : 'intra5', horizons: {} };
  for (const hz of MAA.HZ_ORDER) {
    const t0 = Date.now();
    out.horizons[hz] = calibrate(symbols, hz);
    const a = out.horizons[hz].all;
    console.log(`Kalibrering ${hz}: ${a.n} affärer, TP1 ${a.n ? Math.round(a.tp1 * 100) : '–'} %, snitt ${a.n ? a.avgR.toFixed(2) : '–'}R efter kostnad (${Math.round((Date.now() - t0) / 1000)} s)`);
  }
  fs.writeFileSync(path.join(dir, 'calib.json'), JSON.stringify(out));
}

if (require.main === module) main();
module.exports = { calibrate, band, BANDS };
