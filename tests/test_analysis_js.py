"""Tests for the shared rules (mockup/analysis.js) and the pick logger (pipeline/log_picks.js), run in Node."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def node(js: str):
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, cwd=ROOT)
    if out.returncode:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout)


C = "const c=(o,h,l,cl,t)=>({time:t,open:o,high:h,low:l,close:cl,vol:1});"


@unittest.skipUnless(shutil.which("node"), "node saknas")
class SimulateTrade(unittest.TestCase):
    def run_trade(self, opts, bars):
        return node(f"""const M=require('./mockup/analysis.js');{C}
const all=[c(102,103,101,102,0),...{bars}.map((b,i)=>c(...b,(i+1)*86400))];
const lv={{entry:100,sl:96,tp1:105,tp2:108,tp3:112}};
console.log(JSON.stringify(M.simulateTrade(all,0,lv,1,{{E:3,K:10}},{json.dumps(opts)})));""")

    BARS = "[[101,101,100,100.5],[101,106,100.5,105.5],[104,104,95,95]]"

    def test_open_when_data_ends_before_limit(self):
        r = self.run_trade({}, "[[101,101,100,100.5],[101,103,100.5,102]]")
        self.assertEqual((r["state"], r["hit"], r["fill"]), ("open", 0, 100))

    def test_rules(self):
        fixed = self.run_trade({}, self.BARS)
        trail = self.run_trade({"trail": True}, self.BARS)
        self.assertEqual(fixed["state"], "closed")
        self.assertEqual(fixed["res"], "tp")
        self.assertAlmostEqual(fixed["r"], (1.25 - 1 - 1) / 3)
        # trailing: after TP1 the stop moves to the fill price (100) -> the rest exits at break-even
        self.assertAlmostEqual(trail["r"], 1.25 / 3)
        r_targets = self.run_trade({"tp": "r"}, self.BARS)
        # 1R target = 104 -> hit on bar 2; 2R = 108 not hit; stop on bar 3
        self.assertAlmostEqual(r_targets["r"], (1 - 1 - 1) / 3)

    def test_missed_and_pending(self):
        far = "[[103,104,102,103],[103,104,102,103],[103,104,102,103],[103,104,102,103]]"
        self.assertEqual(self.run_trade({}, far)["state"], "missed")
        self.assertEqual(self.run_trade({}, "[[103,104,102,103]]")["state"], "pending")
        gap = "[[95,96,94,95]]"  # opens below the stop: setup invalid
        self.assertEqual(self.run_trade({}, gap)["state"], "missed")


@unittest.skipUnless(shutil.which("node"), "node saknas")
class AutoCandidates(unittest.TestCase):
    def test_every_instrument_with_data_is_a_candidate(self):
        r = node("""const M=require('./mockup/analysis.js');
const ser=p=>({t:p.map((_,i)=>i*86400),o:p,h:p.map(x=>x+1),l:p.map(x=>x-1),c:p,v:p.map(()=>1)});
const up=[],down=[];for(let i=0;i<60;i++){up.push(100+i);down.push(200-i)}for(let i=0;i<8;i++){up.push(159-i*2);down.push(141+i*2)}
const symbols={AAA:{series:{day:ser(up)}},BBB:{series:{day:ser(down)}},CCC:{series:{day:ser([1,2,3])}}};
const cases={manad:{top:[['AAA','L',.5,'s','t',[80,80,80,80,80,80],'r']],watch:[]}};
const p=M.pool(cases,'manad',symbols);
const picks=M.topPicks(cases,symbols,'manad');
console.log(JSON.stringify({pool:p.top.map(x=>x.s+x.dir+(x.auto?'*':'')),picks:picks.map(m=>[m.item.s,m.item.dir,m.item.auto||false,m.item.setup,m.item.text.slice(0,20)])}));""")
        # the case row stays as it is; BBB (no case) is tried both ways; CCC has too little data
        self.assertEqual(r["pool"], ["AAAL", "BBBL*", "BBBS*"])
        by = {p[0]: p for p in r["picks"]}
        self.assertEqual(len(r["picks"]), 2)  # one analysis per instrument
        self.assertEqual(by["BBB"][1], "S")  # falling market: the short scores higher
        self.assertTrue(by["BBB"][2])
        self.assertIn("i nedtrend", by["BBB"][3])
        self.assertTrue(by["BBB"][4].startswith("Automatiskt urval"))


@unittest.skipUnless(shutil.which("node"), "node saknas")
class PickLogger(unittest.TestCase):
    def test_due_plans(self):
        r = node("""const L=require('./pipeline/log_picks.js');
const log={entries:[]},plans=d=>L.duePlans(log,L.stockholm(new Date(d))).map(p=>p.hz+' '+p.slot);
console.log(JSON.stringify({morning:plans('2026-09-23T07:50:00Z'),evening:plans('2026-09-23T16:30:00Z'),sunday:plans('2026-09-27T10:00:00Z')}));""")
        self.assertEqual(r["morning"], ["intradag 2026-09-23 09:45", "vecka 2026-09-23 09:45"])
        self.assertEqual(r["evening"], ["intradag 2026-09-23 16:05", "vecka 2026-09-23 09:45",
                                        "manad 2026-09-23 18:00", "kvartal 2026-09-23 18:00",
                                        "halvar 2026-09-23 18:00", "ar 2026-09-23 18:00"])
        self.assertEqual(r["sunday"], [])

    def test_log_and_follow_up(self):
        r = node("""const L=require('./pipeline/log_picks.js');
// a daily uptrend with a pullback, then a rally to new highs
const px=[];for(let i=0;i<60;i++)px.push(100+i);for(let i=0;i<12;i++)px.push(159-i*2);
const ser=p=>({t:p.map((_,i)=>i*86400),o:p,h:p.map(x=>x+1),l:p.map(x=>x-1),c:p,v:p.map(()=>1)});
const symbols={AAA:{series:{day:ser(px)}}};
const cases={manad:{top:[['AAA','L',.5,'s','t',[80,80,80,80,80,80],'r']],watch:[]}};
const log={entries:[]},local={epochDay:0};
L.logPicks(cases,symbols,log,{hz:'manad',slot:'x'},local);
const e=log.entries[0];
const after=px.concat([e.entry-1,e.entry+5,e.tp[0]+1,e.tp[1]+1,e.tp[2]+1]);
L.evaluate({AAA:{series:{day:ser(after)}}},log);
console.log(JSON.stringify(log.entries));""")
        self.assertEqual(len(r), 1)
        e = r[0]
        self.assertEqual((e["sym"], e["hz"], e["rank"]), ("AAA", "manad", 1))
        self.assertEqual(e["status"], "closed")
        self.assertEqual(e["hit"], 3)
        self.assertGreater(e["r"], 0)


if __name__ == "__main__":
    unittest.main()
