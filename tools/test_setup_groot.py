"""Groot's pinned GGUF and shared strata-poly setup flow; no GPU or downloads.

    python -m unittest tools.test_setup_groot
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup
from tools import test_setup_glm as glm_tests

FILE = "GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf"
PROJECTOR = "mmproj-GrootTeam_Groot-19B-A3B-261023-bf16.gguf"
REV = "0025df0d03fb385344394f4e5afd73ce47da0680"
QUANTS = ("Q4_K_M", "Q6_K", "Q8_0")


class Pins(unittest.TestCase):
    def test_pinned_files(self):
        fam = setup.FAMILIES["groot"]
        self.assertEqual(fam["hf"], f"https://huggingface.co/bartowski/GrootTeam_Groot-19B-A3B-261023-GGUF/resolve/{REV}/")
        self.assertEqual(fam["mmproj_hf"], fam["hf"])
        self.assertEqual(setup.model_file(fam, "Q4_K_M", 1), FILE)
        self.assertEqual(setup.model_shards(fam, "Q4_K_M"), 1)
        self.assertEqual(fam["mmproj"], PROJECTOR)
        self.assertEqual(fam["sha256"][FILE],
                         (12484339584, "c941e12a231b40b8fb9f6ad7f6601878d72db39c2c62aefb185fbe5c7d0c2e29"))
        self.assertEqual(fam["sha256"][PROJECTOR],
                         (902822208, "0140cef0473d63595a030c68b75b2c8d110bb623bb12ae5993c0cfdfb2d2b0a7"))
        self.assertEqual(tuple(setup.family_models("groot")), QUANTS)
        files = [setup.model_file(fam, quant, 1) for quant in QUANTS]
        self.assertEqual(list(fam["sha256"]), [*files, PROJECTOR])
        for quant, file in zip(QUANTS, files):
            with self.subTest(quant=quant):
                d = fam["models"][quant]
                size, sha = fam["sha256"][file]
                self.assertEqual(setup.model_shards(fam, quant), 1)
                self.assertRegex(sha, r"^[0-9a-f]{64}$")
                self.assertAlmostEqual(d["download_gb"], size / 1e9, delta=0.05)
                self.assertEqual(d["draft"], "own")
                self.assertTrue(d["memory_estimate"])
                self.assertEqual(round(setup.poly_ram_need(d, 8.0)), d["ram_gb"])
        self.assertEqual(setup.POLY_GGUF_ARCHS.get("groot", ("qwen35moe",)), ("qwen35moe",))
        self.assertTrue(fam["unverified"])
        self.assertIn("Q4_K_M text", fam["validation"])
        self.assertNotIn("ran_on_gb", fam)
        self.assertNotIn("tail_skip", fam)
        self.assertNotIn("no_draft", fam)
        self.assertNotIn("contexts", fam)
        self.assertEqual(fam["row_bytes"], 11 * 2 * 2 * 256 * 2)
        for hip in (False, True):
            self.assertIn("groot", setup.menu_families(hip))

    def test_estimates_fit_the_target_but_are_not_measurements(self):
        d = setup.family_models("groot")["Q4_K_M"]
        self.assertEqual(d["arena_gb"], 9937354752 / 1e9)
        self.assertTrue(d["memory_estimate"])
        self.assertEqual(round(setup.poly_ram_need(d, 8.0)), d["ram_gb"])
        self.assertTrue(setup.poly_fits(d, 32.0, 8.0, False))
        self.assertGreater(setup.poly_ram_need(d, 8.0 - 1.6), setup.poly_ram_need(d, 8.0))
        self.assertFalse(setup.poly_fits(d, 32.0, 4.0, False))

    def test_default_and_low_ram_selection(self):
        sizes = setup.family_models("groot")
        self.assertEqual(setup.poly_default(sizes, 32.0, 8.0), "Q4_K_M")
        self.assertEqual(setup.poly_default(sizes, 24.0, 8.0), "Q4_K_M")
        self.assertEqual(setup.poly_default(sizes, 4.0, 8.0), "Q4_K_M")   # nothing fits: the smallest
        self.assertLess(setup.poly_ram_need(sizes["Q4_K_M"], 8.0),
                        setup.poly_ram_need(sizes["Q8_0"], 8.0))

    def test_choices_round_trip(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "strata-groot-q4_k_m.json"
            p.write_text(json.dumps({"exe": setup.POLY_EXE, "args": ["--max-context", "65536"]}))
            choices = setup.choices_from_config(p)
            self.assertEqual((choices["family"], choices["model"], choices["context"]), ("groot", "Q4_K_M", 65536))


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_windows_text_setup_names_what_is_untested(self):
        code, out, cfg = self.main(family="groot", ram=32.0, vram=8.0, win=True)
        self.assertEqual(code, 0, out)
        fam = setup.FAMILIES["groot"]
        self.assertEqual(self.downloads, [fam["hf"] + FILE])
        self.assertEqual(self.verified, [(FILE, *fam["sha256"][FILE])])
        self.assertEqual(self.encoder, "none")
        self.assertNotIn("vision", cfg)
        self.assertEqual(cfg["model_name"], "groot-19b-a3b-261023-q4_k_m")
        self.assertEqual(cfg["display_name"], "Groot 19B-A3B 261023 Q4_K_M")
        args = cfg["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "32768")
        self.assertNotIn("--no-draft", args)
        self.assertNotIn("--draft-model", args)
        self.assertNotIn("--tail-skip", args)
        self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
        self.assertIn("images, Q6_K, Q8_0 and other hardware are untested", out)
        self.assertNotIn("has been run on Linux only", out)
        self.assertNotIn("has run on no card", out)
        self.assertNotIn("tokens/s after a full prompt", out)

    def test_startup_menu_selects_downloads_and_configures_groot(self):
        original_main = setup.main

        def startup():
            argv = list(sys.argv)
            i = argv.index("--family")
            del argv[i:i + 2]
            with mock.patch.object(sys, "argv", argv):
                return original_main()

        choice = str(setup.menu_families(False).index("groot") + 1)
        with mock.patch.object(setup, "main", startup):
            code, out, cfg = self.main(family="groot", ram=32.0, vram=8.0, win=True,
                                       answers={"Which model?": choice, "Context?": "4", "Do you want images?": "y"})
        self.assertEqual(code, 0, out)
        for ctx in setup.POLY_CONTEXTS:
            self.assertIn(f"{ctx // 1024}K tokens", out)
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "131072")
        fam = setup.FAMILIES["groot"]
        self.assertEqual(self.downloads, [fam["hf"] + FILE, fam["mmproj_hf"] + PROJECTOR])
        self.assertEqual(self.encoder, "cpu")
        self.assertNotIn("--no-draft", cfg["args"])

    def test_interactive_size_choice_controls_download_and_model_id(self):
        sizes = list(setup.family_models("groot"))
        choice = str(sizes.index("Q6_K") + 1)
        code, out, cfg = self.main(family="groot", ram=32.0, vram=8.0, win=True,
                                   answers={"Size?": choice})
        self.assertEqual(code, 0, out)
        self.assertTrue(any("Size?" in prompt for prompt in self.asked))
        self.assertEqual(self.downloads, [setup.FAMILIES["groot"]["hf"] +
                                         "GrootTeam_Groot-19B-A3B-261023-Q6_K.gguf"])
        self.assertEqual(cfg["model_name"], "groot-19b-a3b-261023-q6_k")
        self.assertNotIn("--no-draft", cfg["args"])

    def test_each_offered_quantization_uses_its_own_download_and_pins(self):
        fam = setup.FAMILIES["groot"]
        for quant in QUANTS:
            with self.subTest(quant=quant):
                self.downloads, self.verified, self.runs = [], [], []
                code, out, _ = self.main(family="groot", model=quant, ram=64.0, vram=24.0, win=True)
                self.assertEqual(code, 0, out)
                file = setup.model_file(fam, quant, 1)
                self.assertEqual(self.downloads, [fam["hf"] + file])
                self.assertEqual(self.verified, [(file, *fam["sha256"][file])])
                cfg = json.loads((self.t / f"strata-groot-{quant.lower()}.json").read_text())
                self.assertEqual(cfg["model_name"], f"groot-19b-a3b-261023-{quant.lower()}")
                self.assertEqual(Path(cfg["args"][cfg["args"].index("--model") + 1]).name, file)
                self.assertNotIn("--draft-model", cfg["args"])
                self.assertNotIn("--no-draft", cfg["args"])

    def test_repeated_setup_reuses_downloads(self):
        argv = ["--vision", "yes"]
        code, out, cfg = self.main(argv, family="groot", ram=32.0, vram=8.0, win=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.downloads), 2)
        self.downloads = []
        code, out, again = self.main(argv, family="groot", ram=32.0, vram=8.0, win=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [])
        self.assertEqual(again["args"], cfg["args"])
        self.assertEqual(again["vision"], cfg["vision"])

    def test_source_change_does_not_reuse_the_original_tokenizer(self):
        old = self.t / "data" / "packs" / "groot-q4_k_m" / "tokenizer"
        old.mkdir(parents=True)
        (old / "vocab.json").write_text("{}")
        code, out, cfg = self.main(family="groot", ram=32.0, vram=8.0, win=True)
        self.assertEqual(code, 0, out)
        new_pack = self.t / "data" / "packs" / "groot-bartowski-q4_k_m"
        self.assertEqual(Path(cfg["tokenizer"]), new_pack / "tokenizer")
        self.assertEqual(self.runs[0][self.runs[0].index("--out") + 1], str(new_pack))
        self.assertTrue((old / "vocab.json").exists())

    def test_hardware_range_keeps_shared_defaults(self):
        for ram, vram, win in ((24.0, 8.0, True), (32.0, 12.0, False), (64.0, 24.0, True),
                                (128.0, 48.0, False)):
            with self.subTest(ram=ram, vram=vram, win=win):
                code, out, cfg = self.main(family="groot", ram=ram, vram=vram, win=win)
                self.assertEqual(code, 0, out)
                args = cfg["args"]
                self.assertEqual(args[args.index("--max-context") + 1], "32768")
                self.assertNotIn("--no-draft", args)

    def test_context_override_parallel_and_multi_gpu(self):
        code, out, cfg = self.main(["--gpus", "all", "--parallel", "2", "--context", "65536"],
                                   family="groot", ram=64.0, vram=12.0, win=True, n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["layer_split"], "auto")
        self.assertEqual(cfg["parallel"], 2)
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "65536")

    def test_owned_files_do_not_download(self):
        mine = self.t / "my-models"
        mine.mkdir()
        for name in (FILE, PROJECTOR):
            (mine / name).write_bytes(b"")
        code, out, cfg = self.main(["--gguf-dir", str(mine), "--vision", "cpu"],
                                   family="groot", ram=32.0, vram=8.0, win=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [])
        self.assertEqual([v[0] for v in self.verified], [FILE, PROJECTOR])
        self.assertEqual(Path(cfg["vision"]["mmproj"]), mine / PROJECTOR)

    def test_windows_vision_uses_cpu_on_8_gb(self):
        code, out, cfg = self.main(["--vision", "yes", "--context", "8192"],
                                   family="groot", ram=32.0, vram=8.0, win=True)
        self.assertEqual(code, 0, out)
        fam = setup.FAMILIES["groot"]
        self.assertEqual(self.downloads, [fam["hf"] + FILE, fam["mmproj_hf"] + PROJECTOR])
        self.assertEqual(self.verified, [(name, *fam["sha256"][name]) for name in (FILE, PROJECTOR)])
        self.assertEqual(self.encoder, "cpu")
        self.assertFalse(cfg["vision"]["gpu"])
        self.assertEqual(Path(cfg["vision"]["mmproj"]).name, PROJECTOR)
        self.assertIn("threads", cfg["vision"])
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "8192")

    def test_explicit_gpu_vision_is_preserved(self):
        code, out, cfg = self.main(["--vision", "gpu"], family="groot", ram=32.0, vram=12.0, win=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.encoder, "gpu")
        self.assertTrue(cfg["vision"]["gpu"])

    def test_context_override_is_validated(self):
        code, out, cfg = self.main(["--context", "0"], family="groot", ram=32.0, vram=8.0, win=True)
        self.assertNotEqual(code, 0)
        self.assertIn("--context 0", out)
        self.assertIsNone(cfg)


if __name__ == "__main__":
    unittest.main()
