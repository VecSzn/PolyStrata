"""Tests for setup.py's Flint 40B-A2.5B: its four pinned files, the estimates for an 8 GB card, and the install by
GLM's steps with no draft model and no image encoder.  Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_flint
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

FAM = setup.FAMILIES["flint"]
# setup's order: the size taken with the RAM for it, the one with the least RAM, then the others
SIZES = [("UD-Q4_K_XL", 25054688704), ("UD-Q3_K_XL", 18494977472), ("UD-Q6_K_XL", 36319999424),
         ("Q8_0", 42301076928)]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("flint")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/unsloth/flint-GGUF/resolve/[0-9a-f]{40}/$")
        self.assertEqual(list(FAM["sha256"]), [f"flint-{m}.gguf" for m, _ in SIZES])
        for size, n in SIZES:
            file = f"flint-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
        self.assertEqual(setup.POLY_GGUF_ARCHS["flint"], ("flint",))

    def test_the_about_line_says_the_first_size(self):
        n = setup.poly_ram_need(FAM["models"]["UD-Q4_K_XL"], 8.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card", FAM["about"])


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="flint", model=size, ram=64.0, vram=8188 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-flint-{size.lower()}.json").read_text())   # one config a size
                file = f"flint-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["hf"] + file])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                tag = "flint-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                for flag in ("--draft-model", "--tail-skip"):
                    self.assertNotIn(flag, args)
                self.assertEqual(args[args.index("--prompt-batch") + 1], "2048")   # on the 8 GB card too
                self.assertEqual(cfg["model_name"], f"flint-40b-a2.5b-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"Flint 40B-A2.5B {size}")
                self.assertNotIn("vision", cfg)

    def test_the_size_taken_without_a_name(self):
        # UD-Q4_K_XL wants 15 GB past its estimate: a 32 GB PC with an 8 GB card gets UD-Q3_K_XL, one whose card holds
        # most of the experts does not
        for ram, vram, size in ((48.0, 8.0, "UD-Q4_K_XL"), (31.8, 8.0, "UD-Q3_K_XL"), (24.0, 8.0, "UD-Q3_K_XL"),
                                (31.8, 32.0, "UD-Q4_K_XL")):
            with self.subTest(ram=ram, vram=vram):
                self.tearDown()   # a folder of its own: each run leaves its size's config
                self.setUp()
                code, out, _ = self.main(family="flint", ram=ram, vram=vram)
                self.assertEqual(code, 0, out)
                self.assertTrue((self.t / f"strata-flint-{size.lower()}.json").is_file(), out)

    def test_the_contexts_show_its_own_laptop_speed(self):
        # UD-Q3_K_XL on the 8 GB laptop after prompts of 29K, 63K and 126K tokens
        code, out, _ = self.main(family="flint", model="UD-Q3_K_XL", ram=64.0, vram=8.0)
        self.assertEqual(code, 0, out)
        for tps in ("49", "37-38", "23-25"):
            self.assertIn(f"8 GB card: {tps} tokens/s after a full prompt", out)
        self.assertNotIn("52-58", out)
        self.assertNotIn("no long prompt run", out)

    def test_windows_is_not_asked(self):
        code, out, cfg = self.main(win=True, family="flint", ram=32.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertNotIn("Windows", out)

    def test_vision_is_not_used(self):
        code, out, cfg = self.main(["--vision", "yes"], family="flint", ram=32.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertIn("not used with Flint 40B-A2.5B", out)
        self.assertNotIn("vision", cfg)


if __name__ == "__main__":
    unittest.main()
