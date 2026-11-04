"""Tests for setup.py's Kolibri 1 (docs/BACKENDS.md): its three pinned files, the estimates for an 8 GB card, and
the install by GLM's steps with no draft model and no image encoder.  Mocked like tools/test_setup_glm.py, whose
harness this uses.

    python -m unittest tools.test_setup_kolibri
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["kolibri"]
# setup's order: the size taken with the RAM for it, the one with the least RAM, then the others
SIZES = [("Q4_K_M", 47454113472), ("Q2_K", 28634017760), ("Q3_K_M", 37523996640)]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("kolibri")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/Hob-forge/Kolibri-1-GGUF/resolve/[0-9a-f]{40}/$")
        self.assertEqual(list(FAM["sha256"]), [f"Kolibri-1-{m}.gguf" for m, _ in SIZES])
        for size, n in SIZES:
            file = f"Kolibri-1-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
            self.assertNotIn("hf", d)
        # the engine's prompt batch, not a figure of this family's
        self.assertFalse(setup.poly_batch(FAM, FAM["models"]["Q4_K_M"], 31.8, 32768))

    def test_the_about_line_says_the_first_size(self):
        n = setup.poly_ram_need(FAM["models"]["Q4_K_M"], 8.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card", FAM["about"])
        self.assertIn("run on 32 GB", FAM["about"])
        # it has run on Windows, on a 12 GB card: no "not tried" question there
        self.assertIs(FAM.get("windows"), True)
        self.assertEqual(FAM.get("ran_on_gb"), 12)


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="kolibri", model=size, ram=64.0, vram=8188 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-kolibri1-{size.lower()}.json").read_text())
                file = f"Kolibri-1-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["hf"] + file])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                tag = "kolibri1-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                for flag in ("--draft-model", "--tail-skip", "--prompt-batch"):
                    self.assertNotIn(flag, args)
                self.assertEqual(cfg["model_name"], f"kolibri-1-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"Kolibri 1 {size}")
                self.assertNotIn("vision", cfg)
                self.assertIs(cfg["tool_call_recovery"], True)   # JSON inside the template's wrapper

    def test_the_size_taken_without_a_name(self):
        for ram, size in ((64.0, "Q4_K_M"), (40.0, "Q2_K")):
            with self.subTest(ram=ram):
                code, out, _ = self.main(family="kolibri", ram=ram, vram=8.0)
                self.assertEqual(code, 0, out)
                self.assertTrue((self.t / f"strata-kolibri1-{size.lower()}.json").is_file(), out)

    def test_windows_is_not_asked(self):
        code, out, _ = self.main(win=True, family="kolibri", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)
        self.assertNotIn("has not been tried on Windows", out)

    def test_vision_is_not_used(self):
        code, out, cfg = self.main(["--vision", "yes"], family="kolibri", ram=64.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertIn("not used with Kolibri 1", out)
        self.assertNotIn("vision", cfg)


if __name__ == "__main__":
    unittest.main()
