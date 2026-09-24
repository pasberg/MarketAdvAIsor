#!/usr/bin/env node
/* Log the site's top-5 recommendations at fixed times and follow them up against later prices.
 * Uses the same rules as the page (mockup/analysis.js) and the case list embedded in mockup/index.html.
 *
 * When picks are logged (Stockholm time, weekdays):
 *   intradag  09:45 (markets open then: Nordic, commodities) and 16:05 (after the US open)
 *   vecka     09:45 every weekday
 *   longer    once a week, first run after 18:00 when the last log is 6+ days old
 * An instrument is not logged again for a horizon while an earlier pick is still pending or open.
 *
 * Usage: node pipeline/log_picks.js [--data data] [--page mockup/index.html] [--now 2026-09-23T08:00:00Z]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const MAA = require(path.join(__dirname, '..', 'mockup', 'analysis.js'));

const arg = (name, def) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : def; };
const dataDir = arg('data', 'data');
const pagePath = arg('page', path.join('mockup', 'index.html'));
const now = new Date(arg('now', new Date().toISOString()));

function readCases(html) {
  const m = html.match(/<script type="application\/json" id="cases">([\s\S]*?)<\/script>/);
  if (!m) throw new Error('Hittar inte case-blocket i ' + pagePath);
  return JSON.parse(m[1].replace(/<\\\//g, '</'));
}
function readJSON(f) { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { return null; } }
function mergeSymbols(parts) {
  const out = {};
  for (const d of parts) {
    if (!d || !d.symbols) continue;
    for (const [sym, x] of Object.entries(d.symbols)) {
      const t = out[sym] || (out[sym] = { series: {} });
      Object.assign(t.series, x.series || {});
    }
  }
  return out;
}
// Stockholm wall clock for "now", same encoding as the candle times
function stockholm(d) {
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-GB', { timeZone: 'Europe/Stockholm', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', weekday: 'short', hour12: false })
    .formatToParts(d).map(x => [x.type, x.value]));
  const hm = +p.hour % 24 * 60 + +p.minute;
  return { date: `${p.year}-${p.month}-${p.day}`, hm, weekday: !['Sat', 'Sun'].includes(p.weekday), epochDay: Date.UTC(+p.year, +p.month - 1, +p.day) / 86400000 };
}
const HM = s => { const [h, m] = s.split(':').map(Number); return h * 60 + m; };

function duePlans(log, local) {
  const plans = [];
  if (!local.weekday) return plans;
  const has = (hz, slot) => log.entries.some(e => e.hz === hz && e.slot === slot);
  for (const slot of ['09:45', '16:05']) {
    const tag = `${local.date} ${slot}`;
    if (local.hm >= HM(slot) && local.hm < HM(slot) + 180 && !has('intradag', tag)) plans.push({ hz: 'intradag', slot: tag, todayOnly: true });
  }
  if (local.hm >= HM('09:45') && !has('vecka', `${local.date} 09:45`)) plans.push({ hz: 'vecka', slot: `${local.date} 09:45` });
  for (const hz of ['manad', 'kvartal', 'halvar', 'ar']) {
    const last = log.entries.filter(e => e.hz === hz).reduce((a, e) => Math.max(a, Date.parse(e.loggedAt)), 0);
    if (local.hm >= HM('18:00') && now - last > 6 * 86400000) plans.push({ hz, slot: `${local.date} 18:00` });
  }
  return plans;
}

function logPicks(cases, symbols, log, plan, local) {
  const busy = new Set(log.entries.filter(e => e.hz === plan.hz && (e.status === 'pending' || e.status === 'open')).map(e => e.sym));
  let cands = MAA.pool(cases, plan.hz, symbols).top.map(it => MAA.liveAnalysis(symbols, it, plan.hz)).filter(m => m && m.valid);
  if (plan.todayOnly) cands = cands.filter(m => MAA.dayOf(m.c[m.N - 1].time) === local.epochDay); // market open today
  const top = MAA.bestPerSymbol(cands).sort((a, b) => b.score - a.score).slice(0, 5);
  let n = 0;
  top.forEach((m, i) => {
    if (busy.has(m.item.s)) return;
    const t0 = m.c[m.N - 1].time, lv = m.lv;
    log.entries.push({
      id: `${plan.hz}|${m.item.s}|${plan.slot}`, hz: plan.hz, slot: plan.slot, sym: m.item.s, dir: m.item.dir, rank: i + 1, score: m.score,
      loggedAt: now.toISOString(), t0, price: m.P, entry: lv.entry, zLo: lv.zLo, zHi: lv.zHi, sl: lv.sl, tp: [lv.tp1, lv.tp2, lv.tp3], status: 'pending',
    });
    n++;
  });
  return n;
}

function evaluate(symbols, log) {
  let changed = 0;
  for (const e of log.entries) {
    if (e.status !== 'pending' && e.status !== 'open') continue;
    const cfg = MAA.EVAL[e.hz], ser = symbols[e.sym] && symbols[e.sym].series[cfg.ser];
    if (!ser || !ser.t.length) continue;
    const all = MAA.toCandles(ser, 0, ser.t.length);
    let t = -1; for (let i = 0; i < all.length && all[i].time <= e.t0; i++) t = i;
    if (t < 0) continue; // the data no longer reaches back to the signal
    const lv = { entry: e.entry, sl: e.sl, tp1: e.tp[0], tp2: e.tp[1], tp3: e.tp[2] };
    const r = MAA.simulateTrade(all, t, lv, e.dir === 'L' ? 1 : -1, cfg, { tp: 'swing' });
    const prev = e.status;
    e.status = r.state;
    if (r.fillI != null) { e.fill = r.fill; e.fillT = all[r.fillI].time; e.risk = r.risk; e.hit = r.hit; }
    if (r.state === 'closed') { e.exitT = all[r.exitI].time; e.res = r.res; e.r = r.r; e.bars = r.bars; }
    if (e.status !== prev || r.state === 'open') changed++;
  }
  return changed;
}

function main() {
  const cases = readCases(fs.readFileSync(pagePath, 'utf8'));
  const symbols = mergeSymbols(['intra', 'hour', 'daily'].map(p => readJSON(path.join(dataDir, p + '.json'))));
  const file = path.join(dataDir, 'log.json');
  const log = readJSON(file) || { part: 'log', started: now.toISOString(), entries: [] };
  const local = stockholm(now);
  let added = 0;
  for (const plan of duePlans(log, local)) added += logPicks(cases, symbols, log, plan, local);
  const updated = evaluate(symbols, log);
  log.generated = now.toISOString();
  fs.writeFileSync(file, JSON.stringify(log));
  const open = log.entries.filter(e => e.status === 'pending' || e.status === 'open').length;
  console.log(`Logg: ${added} nya förslag, ${updated} uppdaterade, ${open} väntar/öppna, ${log.entries.length} totalt -> ${file}`);
}

if (require.main === module) main();
module.exports = { readCases, mergeSymbols, stockholm, duePlans, logPicks, evaluate };
