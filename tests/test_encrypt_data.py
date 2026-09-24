import base64
import gzip
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pipeline.encrypt_data import derive_kek, encrypt_dir, parse_users, user_id
from pipeline.restore_log import restore

ITER = 1000  # fast for tests; production uses ITERATIONS


def unseal(key, box):
    return AESGCM(key).decrypt(base64.b64decode(box["iv"]), base64.b64decode(box["ct"]), None)


class EncryptData(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "src").mkdir()
        (self.tmp / "src" / "intra.json").write_text('{"part":"intra","generated":"2026-09-23T14:00:00+00:00","x":"åäö"}', encoding="utf-8")
        (self.tmp / "src" / "daily.json").write_text('{"part":"daily"}', encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_parse_users(self):
        self.assertEqual(parse_users(" Anna:hemligt:med:kolon \n\nbad line\nper:pw2;Lisa:x"),
                         {"anna": "hemligt:med:kolon", "per": "pw2", "lisa": "x"})

    def test_roundtrip_per_user(self):
        written = encrypt_dir(self.tmp / "src", self.tmp / "dst", {"anna": "pw1", "per": "pw2"}, ITER)
        self.assertEqual(written, ["intra.enc.json", "daily.enc.json"])
        manifest = json.loads((self.tmp / "dst" / "manifest.json").read_text())
        self.assertEqual(manifest, {"intra": "2026-09-23T14:00:00+00:00", "daily": None})
        auth = json.loads((self.tmp / "dst" / "auth.json").read_text())
        self.assertNotIn("anna", json.dumps(auth))  # user names are hashed
        box = auth["users"][user_id("Anna")]
        dek = unseal(derive_kek("ANNA ", "pw1", ITER), box)
        enc = json.loads((self.tmp / "dst" / "intra.enc.json").read_text())
        self.assertEqual(enc["z"], "gzip")
        self.assertEqual(json.loads(gzip.decompress(unseal(dek, enc)))["x"], "åäö")
        self.assertEqual(json.loads(restore(auth, enc, {"anna": "pw1"}))["x"], "åäö")
        with self.assertRaises(Exception):
            unseal(derive_kek("anna", "fel", ITER), box)

    def test_no_users_gives_locked_site(self):
        encrypt_dir(self.tmp / "src", self.tmp / "dst", {}, ITER)
        self.assertEqual(json.loads((self.tmp / "dst" / "auth.json").read_text())["users"], {})

    @unittest.skipUnless(shutil.which("node"), "node saknas")
    def test_browser_decrypts_with_webcrypto(self):
        """The same steps the page runs, in Node's WebCrypto."""
        encrypt_dir(self.tmp / "src", self.tmp / "dst", {"anna": "pw1"}, ITER)
        js = r"""
const fs=require('fs'),{subtle}=require('crypto').webcrypto,te=new TextEncoder();
const b=s=>Uint8Array.from(Buffer.from(s,'base64'));
(async()=>{
  const dir=process.argv[process.argv.length-1],auth=JSON.parse(fs.readFileSync(dir+'/auth.json'));
  const user='anna',h=async s=>new Uint8Array(await subtle.digest('SHA-256',te.encode(s)));
  const uid=Buffer.from(await h(user)).toString('hex');
  const salt=(await h(auth.saltPrefix+user)).slice(0,16);
  const base=await subtle.importKey('raw',te.encode('pw1'),'PBKDF2',false,['deriveBits']);
  const kekRaw=await subtle.deriveBits({name:'PBKDF2',hash:'SHA-256',salt,iterations:auth.iterations},base,256);
  const kek=await subtle.importKey('raw',kekRaw,'AES-GCM',false,['decrypt']);
  const u=auth.users[uid];
  const dek=await subtle.importKey('raw',await subtle.decrypt({name:'AES-GCM',iv:b(u.iv)},kek,b(u.ct)),'AES-GCM',false,['decrypt']);
  const e=JSON.parse(fs.readFileSync(dir+'/intra.enc.json'));
  const plain=await subtle.decrypt({name:'AES-GCM',iv:b(e.iv)},dek,b(e.ct));
  // same unpacking as the page: DecompressionStream for gzip boxes
  const raw=e.z==='gzip'?await new Response(new Blob([plain]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():plain;
  console.log(new TextDecoder().decode(raw));
})().catch(e=>{console.error(e);process.exit(1)});
"""
        out = subprocess.run(["node", "-e", js, str(self.tmp / "dst")], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(out.stdout)["x"], "åäö")


if __name__ == "__main__":
    unittest.main()
