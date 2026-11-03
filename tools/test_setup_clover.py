"""Tests for setup.py's Clover-35B-A3B-Preview (docs/BACKENDS.md): its three pinned files in two repositories, the
chat template a file without one gets from DiffuSpace's repository, the estimates for an 8 GB card, and the install
by GLM's steps with no draft model and no image encoder.  Mocked like tools/test_setup_glm.py, whose harness this
uses.

    python -m unittest tools.test_setup_clover
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

FAM = setup.FAMILIES["clover"]
# setup's order: the size taken with the RAM for it, the one with the least RAM, then the other
SIZES = [("Q3_K_M", "Diffuspace-Clover-35B-A3B-Preview-Q3_K_M.gguf", 16767725024),
         ("APEX-I-NANOPLUS", "Clover-35B-A3B-Preview.APEX-I-NanoPlus-V2.1.gguf", 12557370304),
         ("Q4_K_M", "Diffuspace-Clover-35B-A3B-Preview-Q4_K_M.gguf", 21169718752)]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("clover")), [m for m, _, _ in SIZES])
        self.assertRegex(FAM["hf"],
                         r"^https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF/resolve/[0-9a-f]{40}/$")
        self.assertEqual(list(FAM["sha256"]), [f for _, f, _ in SIZES])
        for size, file, n in SIZES:
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
        # IsValorum's file is in its own repository, maternion's two in theirs
        self.assertRegex(FAM["models"]["APEX-I-NANOPLUS"]["hf"],
                         r"^https://huggingface.co/IsValorum/Clover-35B-A3B-Preview-APEX-I-NanoPlus-V2.1-GGUF/"
                         r"resolve/[0-9a-f]{40}/$")
        self.assertEqual([m for m, d in FAM["models"].items() if "hf" in d], ["APEX-I-NANOPLUS"])

    def test_the_template_is_pinned_in_the_model_s_repository(self):
        self.assertRegex(FAM["template_hf"],
                         r"^https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/resolve/[0-9a-f]{40}/$")
        name, size, sha = FAM["template"]
        self.assertEqual(name, "chat_template.jinja")
        self.assertGreater(size, 0)
        self.assertRegex(sha, r"^[0-9a-f]{64}$")

    def test_the_about_line_says_the_first_size(self):
        n = setup.poly_ram_need(FAM["models"]["Q3_K_M"], 8.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card", FAM["about"])

    def test_its_files_are_clover_s_architecture(self):
        self.assertEqual(setup.POLY_GGUF_ARCHS["clover"], ("clover_moe",))


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        name, size_t, sha_t = FAM["template"]
        for size, file, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="clover", model=size, ram=64.0, vram=8188 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-clover-{size.lower()}.json").read_text())   # one config a size
                tag = "clover-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                # the mocked tokenizer writes no template, so each size takes the model's own
                self.assertEqual(self.downloads, [FAM["models"][size].get("hf", FAM["hf"]) + file,
                                                  FAM["template_hf"] + name])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file]), (name, size_t, sha_t)])
                self.assertTrue((pack / "tokenizer" / name).is_file())
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                for flag in ("--draft-model", "--tail-skip"):
                    self.assertNotIn(flag, args)
                self.assertEqual(cfg["model_name"], f"clover-35b-a3b-preview-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"Clover-35B-A3B-Preview {size}")
                self.assertNotIn("vision", cfg)

    def test_a_template_the_file_carries_is_kept(self):
        # what the tokenizer leaves when the GGUF has a template, as IsValorum's has
        tpl = self.t / "data" / "packs" / "clover-apex-i-nanoplus" / "tokenizer" / "chat_template.jinja"
        tpl.parent.mkdir(parents=True)
        tpl.write_text("{{ messages }}")
        code, out, _ = self.main(family="clover", model="APEX-I-NANOPLUS", ram=64.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [FAM["models"]["APEX-I-NANOPLUS"]["hf"] + SIZES[1][1]])
        self.assertEqual(tpl.read_text(), "{{ messages }}")

    def test_the_size_taken_without_a_name(self):
        for ram, size in ((32.0, "Q3_K_M"), (14.0, "APEX-I-NANOPLUS")):
            with self.subTest(ram=ram):
                code, out, _ = self.main(family="clover", ram=ram, vram=8.0)
                self.assertEqual(code, 0, out)
                self.assertTrue((self.t / f"strata-clover-{size.lower()}.json").is_file(), out)

    def test_vision_is_not_used(self):
        code, out, cfg = self.main(["--vision", "yes"], family="clover", ram=32.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertIn("not used with Clover-35B-A3B-Preview", out)
        self.assertNotIn("vision", cfg)


if __name__ == "__main__":
    unittest.main()
