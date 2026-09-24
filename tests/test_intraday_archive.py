import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pipeline import intraday_archive as ar

DAY = 86400


def bars(day0: int, times: list[int], base: float = 100.0) -> dict:
    t = [day0 * DAY + m * 60 for m in times]
    return {"t": t, "o": [base] * len(t), "h": [base + 1] * len(t), "l": [base - 1] * len(t), "c": [base + i for i in range(len(t))], "v": [1] * len(t)}


class Bars(unittest.TestCase):
    def test_split_by_stockholm_date_and_merge(self):
        d0 = 20720  # 2026-09-24 as days since epoch
        s = bars(d0, [540, 545])
        two = {k: s[k] + bars(d0 + 1, [540])[k] for k in ar.FIELDS}
        days = ar.by_day({"X": two})
        self.assertEqual(sorted(days), ["2026-09-24", "2026-09-25"])
        self.assertEqual(len(days["2026-09-24"]["X"]["t"]), 2)
        old = bars(d0, [540, 545], 100)
        new = bars(d0, [545, 550], 200)  # 545 overlaps: the newer value wins
        m = ar.merge_bars(old, new)
        self.assertEqual(m["t"], [d0 * DAY + 540 * 60, d0 * DAY + 545 * 60, d0 * DAY + 550 * 60])
        self.assertEqual(m["o"], [100, 200, 200])


@unittest.skipUnless(shutil.which("git"), "git saknas")
class Branch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.remote = self.tmp / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", str(self.remote)], check=True)
        subprocess.run(["git", "config", "uploadpack.allowFilter", "true"], cwd=self.remote, check=True)
        self.users = {"anna": "pw1"}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_once(self, series, now):
        work = self.tmp / f"w{now.timestamp():.0f}"
        b = ar.Branch(f"file://{self.remote}", work)
        res = ar.update(b, self.users, series, now)
        b.commit_and_push("test")
        return res

    def test_days_are_written_encrypted_and_finished_days_kept(self):
        d0 = 20720  # 2026-09-24
        # a run during the day (12:00 Stockholm = 10:00 UTC): the day stays open
        noon = datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)
        r1 = self.run_once({"X": bars(d0, [540, 545])}, noon)
        self.assertEqual(r1["written"], ["2026-09-24"])
        self.assertFalse(r1["index"]["days"]["2026-09-24"]["complete"])
        # the evening run adds the afternoon and closes the day
        night = datetime(2026, 9, 24, 20, 40, tzinfo=timezone.utc)
        r2 = self.run_once({"X": bars(d0, [545, 1000]), "Y": bars(d0, [600])}, night)
        day = r2["index"]["days"]["2026-09-24"]
        self.assertTrue(day["complete"])
        self.assertEqual((day["symbols"], day["bars"]), (2, 4))
        # a later run with the same bars does not touch the finished day
        r3 = self.run_once({"X": bars(d0, [540])}, datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc))
        self.assertEqual(r3["written"], [])
        # the file on the branch is encrypted and readable with the account's key
        check = self.tmp / "check"
        subprocess.run(["git", "clone", "-q", "-b", "archive", f"file://{self.remote}", str(check)], check=True)
        raw = (check / "days/2026/09/2026-09-24.enc.json").read_bytes()
        self.assertNotIn(b'"symbols"', raw)
        b = ar.Branch(f"file://{self.remote}", self.tmp / "reader")
        key = ar.archive_key(b, self.users)
        self.assertEqual(sorted(ar.unpack(key, raw)["symbols"]), ["X", "Y"])
        self.assertEqual(ar.status(r3["index"], datetime.now(timezone.utc))["days"], 1)

    def test_an_unknown_account_cannot_open_the_archive(self):
        self.run_once({"X": bars(20720, [540])}, datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc))
        b = ar.Branch(f"file://{self.remote}", self.tmp / "other")
        with self.assertRaises(RuntimeError):
            ar.archive_key(b, {"per": "fel"})


if __name__ == "__main__":
    unittest.main()
