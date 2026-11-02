"""Tests for setup.py's Qwen3.6-35B-A3B, Ornith 1.5, Gemma 4 26B-A4B and Tiel Coder (docs/BACKENDS.md): the pinned
files, the estimates for a small card, and that each is installed by GLM's steps with nothing of GLM's own (no shipped
expert profile, no draft block to fetch).  Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_gguf_models
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402   (the module: its test classes are not this file's)

# family: (size, file, repository, bytes) of the size setup takes when the PC has the RAM for it
NEW = {
    "qwen36": ("UD-Q3_K_M", "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf", "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", 17104402720),
    "ornith": ("APEX-I-COMPACT", "Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf",
               "mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF", 17437861408),
    "gemma4": ("UD-Q3_K_M", "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf", "unsloth/gemma-4-26B-A4B-it-GGUF", 12728497888),
    "tiel": ("UD-Q3_K_XL", "Tiel-Coder-35B-A3B-MTP-UD-Q3_K_XL.gguf", "peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP",
             17234935648),
}
# family: its other sizes in setup's order, the smaller and then the larger: (size, file, bytes)
MORE = {
    "qwen36": [("UD-Q2_K_XL", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf", 12574128416),
               ("UD-Q4_K_M", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", 22663387424)],
    "ornith": [("APEX-I-MINI", "Ornith-1.5-35B-A3B-APEX-MTP-I-Mini.gguf", 14366220832),
               ("APEX-I-QUALITY", "Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf", 23718411808)],
    "gemma4": [("UD-Q2_K_XL", "gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf", 10546934240),
               ("UD-Q4_K_M", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 16947541728),
               ("QAT-UD-Q4_K_XL", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", 14249047104)],
    "tiel": [("UD-IQ3_XXS", "Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf", 13600579424),
             ("UD-Q4_K_XL", "Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf", 22749880160)],
}


class Pins(unittest.TestCase):
    def test_the_files(self):
        for family, (size0, file0, repo, n0) in NEW.items():
            sizes = [(size0, file0, n0), *MORE[family]]
            fam = setup.FAMILIES[family]
            self.assertEqual(fam["engine"], "poly", family)
            self.assertRegex(fam["hf"], rf"^https://huggingface.co/{repo}/resolve/[0-9a-f]{{40}}/$")
            self.assertEqual(list(setup.family_models(family)), [size for size, _, _ in sizes])
            more = []                                                    # a draft model the sizes share
            for size, file, n in sizes:
                self.assertEqual((setup.model_file(fam, size, 1), setup.model_shards(fam, size)), (file, 1))
                d = fam["models"][size]
                more = [Path(d["draft_model"]).name] if d.get("draft_model") else []
                sha = {**fam["sha256"], **d.get("sha256", {})}             # a size of another repository's own files
                self.assertEqual(fam["sha256"][file][0], n)
                self.assertAlmostEqual(sum(sha[f][0] for f in [file, *more]) / 1e9, d["download_gb"], delta=0.05)
                if "hf" in d:                                                # its draft and image encoder are its own
                    self.assertEqual(sorted(d["sha256"]), sorted([*more, fam["mmproj"]]), size)
                    self.assertIn(d["hf"].split("/resolve/")[0], d["mmproj_hf"])
                self.assertLess(d["arena_gb"], d["download_gb"])        # the experts are part of the file
                self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
            # the image encoder is pinned too; it is downloaded when images are wanted, so it has its own size
            self.assertEqual(list(fam["sha256"]), [*(file for _, file, _ in sizes), *more, fam["mmproj"]])
            for pinned in fam["sha256"].values():
                self.assertRegex(pinned[1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(fam["sha256"][fam["mmproj"]][0] / 1e9, fam["vision_gb"], delta=0.05)

    def test_two_families_have_a_size_of_one_name(self):
        q, g = setup.FAMILIES["qwen36"]["models"]["UD-Q3_K_M"], setup.FAMILIES["gemma4"]["models"]["UD-Q3_K_M"]
        self.assertNotEqual(q["download_gb"], g["download_gb"])
        self.assertNotIn("UD-Q3_K_M", setup.MODELS)                      # the other families' sizes are not theirs
        for f in ("qwen", "swift", "coder", "unsloth", "glm"):
            self.assertNotIn("UD-Q3_K_M", setup.family_models(f), f)

    def test_tags_tell_the_families_apart(self):
        tags = [d["tag"] for d in setup.FAMILIES.values() if d["tag"]]
        for a in tags:
            self.assertEqual([b for b in tags if b.startswith(a)], [a])


class Estimates(unittest.TestCase):
    def test_a_32_gb_card_holds_every_expert(self):
        for family in NEW:
            for d in setup.family_models(family).values():
                self.assertEqual(setup.poly_held_gb(d, 31.84), d["arena_gb"], family)
                self.assertEqual(setup.poly_ram_need(d, 31.84), setup.POLY_RAM_BESIDE_GB)

    def test_an_8_gb_card(self):
        # what the families' menu lines say of an 8 GB card
        need = {f: setup.poly_ram_need(next(iter(setup.family_models(f).values())), 8.0) for f in NEW}
        self.assertAlmostEqual(need["qwen36"], 21, delta=0.5)
        self.assertAlmostEqual(need["ornith"], 22, delta=0.5)
        self.assertAlmostEqual(need["gemma4"], 18, delta=0.5)
        self.assertAlmostEqual(need["tiel"], 21, delta=0.5)
        for f, n in need.items():
            self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card", setup.FAMILIES[f]["about"])
            self.assertEqual(next(iter(setup.family_models(f).values()))["ram_gb"], round(n))

    def test_check_lines(self):
        d = setup.FAMILIES["qwen36"]["models"]["UD-Q3_K_M"]
        self.assertIn("fits by setup's estimate (~21 GB of RAM", setup.poly_verdict(d, 32, 8.0, False))
        self.assertIn("read from the SSD", setup.poly_verdict(d, 12, 8.0, False))
        self.assertIn("does not fit", setup.poly_verdict(d, 64, 6.0, False))
        with mock.patch.object(setup, "WIN", False):   # #11: AMD on Linux runs it
            self.assertNotIn("does not run", setup.poly_verdict(d, 64, 24.0, True))
        with mock.patch.object(setup, "WIN", True):     # a family that has run on Windows is not said to be untried
            self.assertIn("AMD", setup.poly_verdict(d, 64, 24.0, True))           # only AMD on Windows is left out
            self.assertIn("not tried on Windows", setup.poly_verdict(d, 32, 8.0, False))
            self.assertNotIn("not tried on Windows", setup.poly_verdict(d, 32, 8.0, False, True))


class Config(unittest.TestCase):
    def test_choices_from_config(self):
        with tempfile.TemporaryDirectory() as t:
            for family, (size, *_rest) in NEW.items():
                p = Path(t) / f"strata-{setup.FAMILIES[family]['tag']}{size.lower()}.json"
                p.write_text(json.dumps({"exe": "/x/engine-poly/" + setup.POLY_EXE, "args": ["--max-context", "8192"]}))
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"], ch["context"]), (family, size, 8192))


class Main(unittest.TestCase):
    """setup.main() for the four families, every outside effect mocked."""
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install(self):
        for family, (size, file, _repo, _n) in NEW.items():
            with self.subTest(family):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, cfg = self.main(family=family, ram=32.0, vram=8188 / 1024)   # what an 8 GB card reports
                self.assertEqual(code, 0, out)
                fam = setup.FAMILIES[family]
                guesser = fam["models"][size].get("draft_model")      # Gemma 4's: a second file, beside the model's
                more = [Path(guesser).name] if guesser else []
                self.assertEqual(self.downloads, [fam["hf"] + file] + ([fam["hf"] + guesser] if guesser else []))
                self.assertEqual(self.verified, [(f, *fam["sha256"][f]) for f in [file, *more]])
                tag = fam["tag"] + size
                pack = self.t / "data" / "packs" / tag.lower()
                # the tokenizer alone: no draft block is fetched for these
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                self.assertEqual(self.runs[0][self.runs[0].index("--out") + 1], str(pack))
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                self.assertFalse((pack / "expert-profile.txt").exists())   # none is shipped: the engine starts it
                if guesser:
                    self.assertEqual(Path(args[args.index("--draft-model") + 1]),
                                     self.t / "models" / tag / more[0])
                else:
                    self.assertNotIn("--draft-model", args)        # theirs is in the model's file
                if family == "qwen36":                            # the others' answers move more (docs/BACKENDS.md)
                    self.assertEqual(args[args.index("--tail-skip") + 1], "6")
                else:
                    self.assertNotIn("--tail-skip", args)
                self.assertEqual(cfg["model_name"], f"{fam['name']}-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"{fam['title']} {size}")
                self.assertEqual(Path(cfg["exe"]), self.t / setup.ENGINE_POLY_DIR / setup.POLY_EXE)
                self.assertIn("experts fit in VRAM (100% on a 32 GB card)", out)
                self.assertNotIn("has run on no card", out)   # each has run on one of 8 GB
                self.assertEqual((self.encoder, cfg.get("vision")), ("none", None))   # --yes: no images

    def test_a_hybrid_cpu_gets_a_thread_count(self):
        """The engine's own count is every logical CPU, which is the slowest on a hybrid one."""
        code, out, cfg = self.main(family="gemma4", ram=32.0, vram=8.0, cores=(8, 12))
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--threads") + 1], "13")
        self.assertIn("hybrid CPU (8 performance + 12 efficiency cores): 13 CPU threads", out)
        for cores in (None, (8, 8)):
            code, out, cfg = self.main(family="gemma4", ram=32.0, vram=8.0, cores=cores)
            self.assertEqual(code, 0, out)
            self.assertNotIn("--threads", cfg["args"])

    def test_images(self):
        for family, (size, file, _repo, _n) in NEW.items():
            fam = setup.FAMILIES[family]
            folder = self.t / "models" / (fam["tag"] + size)
            # (the card's VRAM, --vision) -> where the encoder runs: a small card keeps its VRAM for the model
            for vram, asked, where in ((31.84, "yes", "gpu"), (8.0, "yes", "cpu"), (12.0, "gpu", "gpu")):
                with self.subTest(family=family, vram=vram, asked=asked):
                    self.downloads, self.verified = [], []
                    code, out, cfg = self.main(["--vision", asked], family=family, ram=64.0, vram=vram)
                    self.assertEqual(code, 0, out)
                    self.assertEqual(self.encoder, where)
                    if vram > 30:                      # the first run: the encoder is the last file fetched
                        self.assertEqual(self.downloads[-1], fam["mmproj_hf"] + fam["mmproj"])
                    self.assertEqual(self.verified[-1], (fam["mmproj"], *fam["sha256"][fam["mmproj"]]))
                    v = cfg["vision"]
                    self.assertEqual((Path(v["exe"]), Path(v["mmproj"]), Path(v["model"]), v["gpu"]),
                                     (self.t / setup.ENGINE_POLY_DIR / setup.VEXE, folder / fam["mmproj"],
                                      folder / file, where == "gpu"))
                    self.assertEqual(v["max_tokens"], setup.VISION[where]["max_tokens"])
                    self.assertEqual("threads" in v, where == "cpu")
                    self.assertIn("images: on" + (" (encoder on the CPU)" if where == "cpu" else ""), out)
            # the encoder was downloaded once, beside the model's file, and found there afterwards
            self.assertTrue((folder / fam["mmproj"]).exists())
            self.assertEqual(self.downloads, [])
        # the encoder on the GPU leaves less VRAM to the experts, and the estimate says so
        need = {}
        for asked in ("no", "gpu"):
            code, out, cfg = self.main(["--vision", asked], family="gemma4", ram=64.0, vram=16.0,
                                       answers={})
            need[asked] = out
        self.assertIn("about 10 of", need["no"])
        self.assertIn("about 8 of", need["gpu"])
        # and on a card it leaves too little of, setup stops and says how to keep it off the card
        code, out, cfg = self.main(["--vision", "gpu"], family="gemma4", ram=64.0, vram=8.0)
        self.assertNotEqual(code, 0)
        self.assertIn("and its image encoder 2.4 GB", out)
        self.assertIn("--vision cpu keeps the image encoder off the card", out)

    def test_a_size_from_another_repository(self):
        """Gemma 4's QAT file: its repository's draft model and image encoder, checked against that size's hashes."""
        fam, size = setup.FAMILIES["gemma4"], "QAT-UD-Q4_K_XL"
        d = fam["models"][size]
        draft = Path(d["draft_model"]).name
        self.downloads, self.verified = [], []
        code, out, cfg = self.main(["--model", size, "--vision", "yes"], family="gemma4", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [d["hf"] + d["file"], d["hf"] + d["draft_model"], d["mmproj_hf"] + fam["mmproj"]])
        self.assertEqual(self.verified, [(d["file"], *fam["sha256"][d["file"]]), (draft, *d["sha256"][draft]),
                                         (fam["mmproj"], *d["sha256"][fam["mmproj"]])])
        self.assertNotEqual(d["sha256"][draft], fam["sha256"][draft])      # same names, other files
        folder = self.t / "models" / (fam["tag"] + size)
        args = cfg["args"]
        self.assertEqual(Path(args[args.index("--model") + 1]), folder / d["file"])
        self.assertEqual(Path(args[args.index("--draft-model") + 1]), folder / draft)
        self.assertEqual(Path(cfg["vision"]["mmproj"]), folder / fam["mmproj"])

    def test_images_are_asked_about(self):
        code, out, cfg = self.main(family="qwen36", ram=64.0, vram=24.0, answers={"Do you want images": "y"})
        self.assertEqual(code, 0, out)
        self.assertIn("a 0.9 GB", out)
        self.assertEqual((self.encoder, cfg["vision"]["gpu"]), ("gpu", True))
        # GLM 5.3 Flash reads no pictures: it is not asked, and --vision is said to be unused
        code, out, cfg = self.main(["--vision", "yes"], family="glm")
        self.assertEqual(code, 0, out)
        self.assertIn("not used with GLM 5.3 Flash", out)
        self.assertNotIn("vision", cfg)

    def test_the_draft_model_beside_files_of_ones_own(self):
        size, file, _repo, _n = NEW["gemma4"]
        fam = setup.FAMILIES["gemma4"]
        guesser = fam["models"][size]["draft_model"]
        mine = self.t / "mine"
        mine.mkdir()
        (mine / file).write_bytes(b"")
        # --gguf-dir has the model alone: its draft model is downloaded, to where setup keeps its own files
        code, out, cfg = self.main(["--gguf-dir", str(mine)], family="gemma4", ram=32.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [fam["hf"] + guesser])
        args = cfg["args"]
        self.assertEqual(Path(args[args.index("--model") + 1]), mine / file)
        self.assertEqual(Path(args[args.index("--draft-model") + 1]),
                         self.t / "models" / (fam["tag"] + size) / Path(guesser).name)
        # with both files there, nothing is downloaded and both are used where they are
        (mine / Path(guesser).name).write_bytes(b"")
        self.downloads, self.verified = [], []
        code, out, cfg = self.main(["--gguf-dir", str(mine)], family="gemma4", ram=32.0, vram=8.0)
        self.assertEqual((code, self.downloads), (0, []), out)
        self.assertEqual([v[0] for v in self.verified], [file, Path(guesser).name])
        self.assertEqual(Path(cfg["args"][cfg["args"].index("--draft-model") + 1]), mine / Path(guesser).name)

    def test_a_pc_below_the_qwen_models_ram_is_not_stopped(self):
        # 24 GB is under what the smallest Qwen3.8 size needs, and enough for these by their own estimate
        code, out, cfg = self.main(family="gemma4", ram=24.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertNotIn("the Coder", out)
        self.assertIn("RAM: 24 GB", out)
        self.assertIsNotNone(cfg)

    def test_too_little_ram_for_the_model_stops(self):
        code, out, cfg = self.main(family="qwen36", ram=12.0, vram=8.0)   # too little for its smallest size too
        self.assertNotEqual(code, 0)
        self.assertIn("wants about 17 GB", out)
        self.assertIsNone(cfg)
        code, out, cfg = self.main(family="qwen36", ram=12.0, vram=8.0, model="UD-Q3_K_M")   # named: a risk taken
        self.assertEqual(code, 0, out)
        self.assertIn("as you chose", out)

    def test_the_size_taken_without_a_name(self):
        """The first size where the PC has the RAM for it, the smaller one where not; the larger one only by name."""
        for ram, size in ((64.0, "UD-Q3_K_M"), (20.0, "UD-Q3_K_M"), (16.0, "UD-Q2_K_XL")):
            code, out, cfg = self.main(family="qwen36", ram=ram, vram=8.0)
            self.assertEqual(code, 0, out)
            self.assertIn(f"size: {size}", out, ram)
        code, out, cfg = self.main(family="qwen36", ram=64.0, vram=8.0, model="UD-Q4_K_M")
        self.assertEqual(code, 0, out)
        self.assertIn("size: UD-Q4_K_M", out)

    def test_a_size_of_another_family_is_refused_by_name(self):
        code, out, cfg = self.main(family="gemma4", model="APEX-I-COMPACT")
        self.assertNotEqual(code, 0)
        self.assertIn("Gemma 4 26B-A4B has no APEX-I-COMPACT model file", out)
        self.assertIn("UD-Q3_K_M", out)
        code, out, cfg = self.main(family="qwen", model="UD-Q3_K_M")     # and the other way: who has it
        self.assertNotEqual(code, 0)
        self.assertIn("Qwen3.8-Flash-Next has no UD-Q3_K_M model file", out)
        self.assertIn("--family qwen36 --model UD-Q3_K_M, --family gemma4 --model UD-Q3_K_M", out)

    def test_the_context_note_is_the_familys(self):
        code, out, cfg = self.main(family="gemma4", ram=32.0, vram=8.0, answers={"Context?": "3"})
        self.assertEqual(code, 0, out)
        self.assertIn("0.9 GB more VRAM", out)                           # 65536: 32768 tokens of 28672 bytes more
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "65536")


if __name__ == "__main__":
    unittest.main()
