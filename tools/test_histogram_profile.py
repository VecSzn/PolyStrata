#!/usr/bin/env python3
"""tools/histogram_profile.py on a small made-up histogram: ranking, fill, phases, the file format, coverage.

    python tools/test_histogram_profile.py      (or: python -m unittest tools/test_histogram_profile.py)
"""
import importlib.util
import pathlib
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "histogram_profile.py"
spec = importlib.util.spec_from_file_location("histogram_profile", TOOL)
hp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp)


def make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE models (model_id TEXT, arch TEXT, n_layers INT, n_experts INT, top_k INT, "
                "hidden_size INT, notes TEXT)")
    con.execute("CREATE TABLE histogram_counts (model_id TEXT, class_id TEXT, phase TEXT, layer INT, "
                "expert_id INT, count INT)")
    con.execute("INSERT INTO models VALUES ('toy', 'toy', 2, 4, 2, 64, '')")
    rows = [("live", "decode", 0, 3, 50), ("live", "decode", 1, 0, 40), ("live", "decode", 0, 1, 10),
            ("live", "prefill", 1, 2, 400), ("other", "decode", 1, 3, 30)]
    con.executemany("INSERT INTO histogram_counts VALUES ('toy', ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()


def test_rank_decode_then_fill():
    with tempfile.TemporaryDirectory() as d:
        db = pathlib.Path(d) / "h.sqlite"
        make_db(db)
        nl, ne, freq = hp.read_db(str(db), "toy")
        assert (nl, ne) == (2, 4)
        ranked, n_counted = hp.rank_pairs(freq, nl, ne)
        assert ranked[:4] == [(0, 3), (1, 0), (1, 3), (0, 1)]       # decode only, most used first, both classes
        assert n_counted == 4 and len(ranked) == 8 and len(set(ranked)) == 8   # every pair ranked once
        assert ranked[4:] == [(0, 0), (1, 1), (0, 2), (1, 2)]       # uncounted pairs interleaved by layer


def test_class_and_phase_filters():
    with tempfile.TemporaryDirectory() as d:
        db = pathlib.Path(d) / "h.sqlite"
        make_db(db)
        _, _, live = hp.read_db(str(db), "toy", classes=["live"])
        assert (1, 3) not in live
        _, _, pre = hp.read_db(str(db), "toy", phase="prefill")
        assert pre == {(1, 2): 400.0}                               # no weight unless prefill is mixed in
        _, _, mix = hp.read_db(str(db), "toy", phase="all", prefill_weight=0.25)
        assert mix[(1, 2)] == 100.0 and mix[(0, 3)] == 50.0


def test_file_format_and_coverage():
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "p.bin"
        freq = {(0, 3): 50.0, (1, 0): 40.0, (1, 3): 30.0, (0, 1): 10.0}
        ranked, _ = hp.rank_pairs(freq, 2, 4)
        hp.write_profile(str(out), ranked, 2, 4)
        blob = out.read_bytes()
        assert blob[:4] == b"STRP" and struct.unpack_from("<5I", blob, 4) == (1, 2, 4, 8, 8)
        assert len(blob) == 24 + 4 * 8 + 4 * 8                      # header, u16 pairs, i32 slot table
        table = struct.unpack_from("<8i", blob, 24 + 4 * 8)
        assert table[0 * 4 + 3] == 0 and table[1 * 4 + 0] == 1      # (layer, expert) -> its slot
        assert hp.read_profile(str(out)) == (2, 4, ranked)
        assert hp.coverage(ranked, freq, 2) == 90 / 130


def test_counts_format_for_strata_poly():
    with tempfile.TemporaryDirectory() as d:
        db = pathlib.Path(d) / "h.sqlite"
        make_db(db)
        out = pathlib.Path(d) / "p.txt"
        r = subprocess.run([sys.executable, str(TOOL), "--db", str(db), "--model-id", "toy", "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        # what src/poly/engine.cpp read_profile scans: "layer expert count", counted pairs only
        assert out.read_text() == "0 1 10\n0 3 50\n1 0 40\n1 3 30\n"


def test_cli_reports_missing_phase():
    with tempfile.TemporaryDirectory() as d:
        db = pathlib.Path(d) / "h.sqlite"
        make_db(db)
        con = sqlite3.connect(db)
        con.execute("INSERT INTO models VALUES ('pre', 'toy', 2, 4, 2, 64, '')")
        con.execute("INSERT INTO histogram_counts VALUES ('pre', 'scan', 'prefill', 0, 0, 5)")
        con.commit()
        con.close()
        r = subprocess.run([sys.executable, str(TOOL), "--db", str(db), "--model-id", "pre",
                            "--out", str(pathlib.Path(d) / "p.bin")], capture_output=True, text=True)
        assert r.returncode != 0 and "no decode counts" in r.stderr and "scan/prefill" in r.stderr


class HistogramProfile(unittest.TestCase):
    """The test_ functions above, for the tools suite (python -m unittest tools/test_*.py)."""


for _name, _fn in list(globals().items()):
    if _name.startswith("test_"):
        setattr(HistogramProfile, _name, lambda self, fn=_fn: fn())
del _name, _fn   # left bound to the last global, the class itself, unittest would collect it twice


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
