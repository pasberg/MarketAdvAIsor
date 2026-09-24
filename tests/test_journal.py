"""The trading journal: the browser's encryption (WebCrypto) must match the pipeline's export,
and the Avanza import must pair buys and sells into trades."""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from pipeline.encrypt_data import user_id
from pipeline.journal_export import export

ROOT = Path(__file__).resolve().parents[1]
ITER = 1000


def page_functions(*names: str) -> str:
    """Pull top-level function/const definitions out of the page script, for Node."""
    src = (ROOT / "mockup" / "index.html").read_text(encoding="utf-8")
    out = []
    for n in names:
        m = re.search(rf"^const {n}=.*$", src, re.M) or re.search(rf"^function {n}\(.*?\n\}}$", src, re.M | re.S)
        if not m:
            raise AssertionError(f"{n} saknas i sidan")
        out.append(m.group(0))
    return "\n".join(out)


@unittest.skipUnless(shutil.which("node"), "node saknas")
class Journal(unittest.TestCase):
    def node(self, js: str):
        r = subprocess.run(["node", "-e", js], capture_output=True, text=True, cwd=ROOT)
        if r.returncode:
            raise AssertionError(r.stderr)
        return json.loads(r.stdout)

    def test_browser_encrypted_journal_is_exported(self):
        box = self.node(r"""
const {subtle}=require('crypto').webcrypto,te=new TextEncoder();
const b64=u=>Buffer.from(u).toString('base64');
(async()=>{
  const user='anna',h=async s=>new Uint8Array(await subtle.digest('SHA-256',te.encode(s)));
  const salt=(await h('marketadvaisor-v1:'+user)).slice(0,16);
  const base=await subtle.importKey('raw',te.encode('pw1'),'PBKDF2',false,['deriveBits']);
  const kek=new Uint8Array(await subtle.deriveBits({name:'PBKDF2',hash:'SHA-256',salt,iterations:1000},base,256));
  const pre=te.encode('marketadvaisor-journal-v1:'),buf=new Uint8Array(pre.length+kek.length);buf.set(pre);buf.set(kek,pre.length);
  const key=await subtle.importKey('raw',await subtle.digest('SHA-256',buf),'AES-GCM',false,['encrypt']);
  const iv=crypto.getRandomValues(new Uint8Array(12));
  const ct=await subtle.encrypt({name:'AES-GCM',iv},key,te.encode(JSON.stringify({v:1,trades:[{id:'m1',name:'VOLV-B'}],fills:[],notes:{}})));
  console.log(JSON.stringify({v:1,iv:b64(iv),ct:b64(new Uint8Array(ct))}));
})();""")
        res = export({user_id("anna"): json.dumps(box).encode(), "okänd": b"{}"}, {"anna": "pw1"}, ITER)
        self.assertEqual(res["journals"]["anna"]["trades"][0]["name"], "VOLV-B")
        self.assertIn("okänd"[:8], res["errors"])
        wrong = export({user_id("anna"): json.dumps(box).encode()}, {"anna": "fel"}, ITER)
        self.assertEqual(wrong["journals"], {})

    def test_avanza_import_pairs_trades(self):
        fns = page_functions("jrNum", "guessProduct", "guessView", "parseAvanza", "roundTrips")
        csv = "\n".join([
            "Datum;Konto;Typ av transaktion;Värdepapper/beskrivning;Antal;Kurs;Belopp;Transaktionsvaluta;Courtage;Valutakurs;Instrumentvaluta;ISIN;Resultat",
            "2026-09-01;ISK;Köp;Volvo B;10;280,00;-2 801,00;SEK;1,00;;SEK;SE0000115446;",
            "2026-09-02;ISK;Köp;Volvo B;10;290,00;-2 901,00;SEK;1,00;;SEK;SE0000115446;",
            "2026-09-05;ISK;Sälj;Volvo B;20;300,00;5 998,00;SEK;2,00;;SEK;SE0000115446;",
            "2026-09-08;ISK;Köp;BEAR OMX X10 AVA 3;100;5,00;-501,00;SEK;1,00;;SEK;;",
            "2026-09-08;ISK;Sälj;BEAR OMX X10 AVA 3;100;5,50;549,00;SEK;1,00;;SEK;;",
            "2026-09-09;ISK;Köp;Ericsson B;5;80,00;-400,00;SEK;0,00;;SEK;;",
            "2026-09-10;ISK;Utdelning;Volvo B;20;7,00;140,00;SEK;;;SEK;;",
        ])
        r = self.node(f"""const hash=s=>{{let h=2166136261;for(const c of s){{h^=c.charCodeAt(0);h=Math.imul(h,16777619)}}return h>>>0}};
const FX={{SEK:1}};{fns}
const fills=parseAvanza({json.dumps(csv)}),again=parseAvanza({json.dumps(csv)});
console.log(JSON.stringify({{n:fills.length,sameKeys:fills.map(f=>f.k).join()===again.map(f=>f.k).join(),trips:roundTrips(fills)}}));""")
        self.assertEqual(r["n"], 6)  # the dividend is not a trade
        self.assertTrue(r["sameKeys"])  # re-importing the same file adds nothing
        by = {t["name"]: t for t in r["trips"]}
        volvo = by["Volvo B"]
        self.assertEqual((volvo["open"], volvo["close"], volvo["qty"]), ("2026-09-01", "2026-09-05", 20))
        self.assertAlmostEqual(volvo["entry"], 285)
        self.assertAlmostEqual(volvo["pnlSEK"], 5998 - 2801 - 2901)
        bear = by["BEAR OMX X10 AVA 3"]
        self.assertEqual((bear["product"], bear["dir"], bear["hz"]), ("certifikat", "S", "intradag"))
        self.assertAlmostEqual(bear["pnlSEK"], 48)
        self.assertIsNone(by["Ericsson B"].get("close"))  # still open


if __name__ == "__main__":
    unittest.main()
