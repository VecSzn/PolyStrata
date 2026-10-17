"""Tests for setup.py's GLM 5.3 Flash choice (docs/GLM.md): the pinned file with its size and SHA-256, the steps of
its own setup (no pack, no draft layer, no image encoder), the run config, the conversation cache kept on a second
run, and what setup says about RAM, VRAM and the options of the Qwen engine.  Mocked - no GPU, no compile, no
downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_glm
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

M = "3.0BIT"
FILE = "GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf"
REV = "7feedfeb5ee924bfbe0064c63b124728b443ddc0"
HALF = "REAP50-Q3_K_M"                                  # half the routed experts, from another repository
HALF_FILE = "GLM-5.3-Flash-REAP50-Q3_K_M.gguf"
SPLIT = "UD-IQ2_XXS"                                    # the numbered files of a split GGUF, in a folder of a third
SPLIT_FILES = [f"GLM-5.3-Flash-UD-IQ2_XXS-0000{i}-of-00004.gguf" for i in range(1, 5)]


def quiet(fn, *args):
    """-> (exit code or None, printed text)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class Pins(unittest.TestCase):
    def test_the_file(self):
        fam = setup.FAMILIES["glm"]
        self.assertEqual(fam["hf"], f"https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF/"
                                    f"resolve/{REV}/")
        self.assertEqual(list(fam["sha256"]), [FILE, HALF_FILE, *SPLIT_FILES])
        size, sha = fam["sha256"][FILE]
        self.assertEqual(size, 113585695168)
        self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertEqual(setup.model_file(fam, M, 1), FILE)
        self.assertEqual(setup.model_shards(fam, M), 1)

    def test_a_size_from_another_repository(self):
        fam, d = setup.FAMILIES["glm"], setup.MODELS[HALF]
        self.assertRegex(d["hf"], r"^https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF/resolve/"
                                  r"[0-9a-f]{40}/$")
        self.assertEqual(setup.model_file(fam, HALF, 1), HALF_FILE)
        size, sha = fam["sha256"][HALF_FILE]
        self.assertEqual(size, 78776759712)
        self.assertRegex(sha, r"^[0-9a-f]{64}$")
        rows = [line.split() for line in (ROOT / "data" / d["profile"]).read_text().splitlines()]
        self.assertEqual(sorted({int(r[0]) for r in rows}), list(range(3, 46)))     # 45: its draft block
        self.assertEqual(max(int(r[1]) for r in rows), 143)             # 144 experts a layer

    def test_not_in_the_other_families(self):
        for f in setup.FAMILIES:
            names = [m for m in setup.MODELS if f in setup.MODELS[m].get("families", ("qwen", "swift"))]
            self.assertEqual(M in names, f == "glm", f)

    def test_the_shipped_profile_names_every_routed_layer(self):
        rows = [line.split() for line in (ROOT / "data" / setup.FAMILIES["glm"]["profile"]).read_text().splitlines()]
        self.assertTrue(all(len(r) == 3 and int(r[2]) > 0 for r in rows))
        self.assertEqual(sorted({int(r[0]) for r in rows}), list(range(3, 46)))     # 45: the draft block
        self.assertEqual(max(int(r[1]) for r in rows), 287)


class Estimates(unittest.TestCase):
    def test_the_measured_card(self):
        # an RTX 5090 lists 31.8 GiB; its cache held 19.9 GiB of experts at 32K
        self.assertAlmostEqual(setup.poly_held_gb(setup.MODELS[M], 31.84) / 1.073741824, 19.9, delta=0.2)
        self.assertAlmostEqual(setup.poly_ram_need(setup.MODELS[M], 31.84), 96, delta=1)

    def test_the_half_expert_size(self):
        # the same card held 20.8 GiB of its experts
        self.assertAlmostEqual(setup.poly_held_gb(setup.MODELS[HALF], 31.84) / 1.073741824, 20.8, delta=0.2)
        self.assertAlmostEqual(setup.poly_ram_need(setup.MODELS[HALF], 31.84), 60, delta=1)

    def test_the_size_taken_when_none_is_named(self):
        sizes = setup.family_models("glm")
        self.assertEqual(list(sizes), [M, SPLIT, HALF])                  # the whole model before the half of it
        self.assertEqual(setup.poly_default(sizes, 128, 31.84), M)
        self.assertEqual(setup.poly_default(sizes, 88, 31.84), SPLIT)     # 3.0BIT wants 96 GB there
        self.assertEqual(setup.poly_default(sizes, 64, 31.84), HALF)
        self.assertEqual(setup.poly_default(sizes, 32, 31.84), HALF)      # none fits: the one that wants least

    def test_the_split_size(self):
        fam, d = setup.FAMILIES["glm"], setup.MODELS[SPLIT]
        self.assertEqual([setup.model_file(fam, SPLIT, i) for i in range(1, 5)], SPLIT_FILES)
        self.assertEqual(setup.model_shards(fam, SPLIT), 4)
        self.assertRegex(d["hf"], r"^https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/[0-9a-f]{40}/UD-IQ2_XXS/$")
        self.assertAlmostEqual(sum(fam["sha256"][f][0] for f in SPLIT_FILES) / 1e9, d["download_gb"], delta=0.1)
        # the same card held 19.3 GiB of its experts
        self.assertAlmostEqual(setup.poly_held_gb(setup.MODELS[SPLIT], 31.84) / 1.073741824, 19.3, delta=0.1)
        self.assertAlmostEqual(setup.poly_ram_need(setup.MODELS[SPLIT], 31.84), d["ram_gb"], delta=1)

    def test_a_card_that_holds_none(self):
        self.assertEqual(setup.poly_held_gb(setup.MODELS[M], 8.0), 0.0)
        self.assertAlmostEqual(setup.poly_ram_need(setup.MODELS[M], 8.0), 117.2, delta=0.1)

    def test_check_lines(self):
        with mock.patch.object(setup, "WIN", False):   # #11: AMD on Linux runs it
            self.assertNotIn("does not run", setup.poly_verdict(setup.MODELS[M], 256, 24.0, True))
        with mock.patch.object(setup, "WIN", True):
            self.assertIn("AMD", setup.poly_verdict(setup.MODELS[M], 256, 24.0, True))   # only AMD on Windows is left out
        self.assertIn("does not fit", setup.poly_verdict(setup.MODELS[M], 256, 12.0, False))
        self.assertIn("fits by setup's estimate", setup.poly_verdict(setup.MODELS[M], 128, 31.84, False))
        self.assertIn("read from the SSD", setup.poly_verdict(setup.MODELS[M], 64, 31.84, False))


class Config(unittest.TestCase):
    def test_choices_from_config(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "strata-glm-3.0bit.json"
            p.write_text(json.dumps({"exe": "/x/engine-poly/" + setup.POLY_EXE, "args": ["--max-context", "65536"]}))
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"], ch["context"], ch["vision"]), ("glm", M, 65536, "none"))
            self.assertTrue(setup.poly_config(json.loads(p.read_text())))

    def test_an_install_from_before_the_engine_was_renamed(self):
        # its config has strata-glm in engine-glm/: it is this engine's, and an update moves it to the new one
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            p = root / "strata-glm-3.0bit.json"
            cfg = {"exe": str(root / setup.OLD_ENGINE_POLY_DIR / setup.OLD_POLY_EXE), "args": ["--max-context", "32768"]}
            p.write_text(json.dumps(cfg))
            self.assertTrue(setup.poly_config(cfg))
            self.assertEqual(setup.choices_from_config(p)["family"], "glm")

            def build(gpu, yes, llama, vision="none", works=True):
                if not works:
                    raise RuntimeError("no compiler")
                (root / setup.ENGINE_POLY_DIR).mkdir(exist_ok=True)
                (root / setup.ENGINE_POLY_DIR / setup.POLY_EXE).write_text("")
                (root / setup.ENGINE_POLY_DIR / "BUILD.json").write_text(json.dumps({"cuda_dirs": ["/cuda/bin"]}))
                return root / setup.ENGINE_POLY_DIR

            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "get_llama_cpp", lambda: root), \
                    mock.patch.object(setup, "gpu_info", lambda i=None: {"arch": "89"}):
                # a compile that fails leaves the install as it was: the old engine still starts
                with mock.patch.object(setup, "build_engine_poly", lambda *a: build(*a, works=False)):
                    _code, out = quiet(setup.update_poly_engine, dict(cfg), False, p)
                self.assertIn("could not update strata-poly", out)
                self.assertEqual(json.loads(p.read_text())["exe"], cfg["exe"])
                with mock.patch.object(setup, "build_engine_poly", build):
                    _code, out = quiet(setup.update_poly_engine, cfg, False, p)
            new = root / setup.ENGINE_POLY_DIR / setup.POLY_EXE
            self.assertEqual((Path(cfg["exe"]), Path(json.loads(p.read_text())["exe"])), (new, new))
            self.assertEqual(json.loads(p.read_text())["lib_dirs"], ["/cuda/bin"])
            self.assertIn(setup.OLD_ENGINE_POLY_DIR + "/ can be deleted", out)
        self.assertFalse(setup.poly_config({"exe": "/x/engine/" + setup.EXE}))

    def test_the_compile_takes_the_cmake_pip_installed(self):
        # a cmake on PATH can be older than CMakeLists.txt takes (Ubuntu 22.04's); the one beside Python is setup's
        with tempfile.TemporaryDirectory() as t:
            mine, name = Path(t) / "bin", "cmake" + (".exe" if setup.WIN else "")
            mine.mkdir()
            with mock.patch.object(sys, "executable", str(mine / "python")), \
                    mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n), \
                    mock.patch.object(setup.Path, "home", lambda: Path(t) / "home"):
                self.assertEqual(setup.find_tool("cmake"), "/usr/bin/cmake")       # none of its own: PATH's
                (mine / name).write_text("")
                self.assertEqual(setup.find_tool("cmake"), str(mine / name))
                with mock.patch.object(setup.shutil, "which", lambda n: None):
                    self.assertIsNone(setup.find_tool("ninja"))

    def test_saved_conversation_cache(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "c.json"
            self.assertIsNone(setup.poly_saved_conv([p, None]))
            p.write_text(json.dumps({"args": ["--model", "m.gguf"]}))
            self.assertEqual(setup.poly_saved_conv([None, p]), 0)
            p.write_text(json.dumps({"args": ["--conversation-cache-mib", "4096", "--model", "m.gguf"]}))
            self.assertEqual(setup.poly_saved_conv([p]), 4096)


class Main(unittest.TestCase):
    """setup.main() for --family glm, every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data").mkdir()
        (self.t / "data" / "expert-profile-glm.txt").write_text("3 0 7\n")
        (self.t / "data" / "expert-profile-glm-reap50.txt").write_text("3 1 5\n")
        self.downloads, self.runs, self.verified, self.built = [], [], [], []

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv=(), ram=128.0, vram=31.84, model=False, answers=None, win=False, n_gpus=1, family="glm",
             cores=None):
        """answers: None = --yes; else {words of a question: its answer} (Enter for the others); the questions asked
        are kept in self.asked.  model: False, True (this file's M) or a size's name.  cores: (performance,
        efficiency) of a hybrid CPU."""
        self.asked = []

        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r}")
            self.asked.append(prompt)
            return next((v for k, v in answers.items() if k in prompt), "")

        found = [{"index": i, "name": "NVIDIA GeForce RTX 5090", "vram_gb": vram, "arch": "120", "driver": "580.97"}
                 for i in range(n_gpus)]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_build(gpu, yes, llama, vision="none", toolkit=13):
            self.built.append(gpu["archs"])
            self.encoder, self.toolkit = vision, toolkit
            eng = self.t / setup.ENGINE_POLY_DIR
            eng.mkdir(exist_ok=True)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "cuda_dirs": ["/cuda/lib64"]}))
            return eng

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "WIN", win),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a: []),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "cpu_cores", lambda: cores),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", mock.Mock(side_effect=AssertionError("the Qwen engine"))),
            mock.patch.object(setup, "build_engine_poly", fake_build),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "whole_shard", lambda s: False),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", lambda cmd, *a, **k: self.runs.append([str(x) for x in cmd])),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.sh"),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.object(sys, "argv", ["setup.py", "--family", family,
                                            *(["--model", M if model is True else model] if model else []),
                                            *(["--yes"] if answers is None else []), "--no-start",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfg = next(iter(sorted(self.t.glob(f"strata-{setup.FAMILIES[family]['tag']}*.json"))), None)
        return code, out, (json.loads(cfg.read_text()) if cfg else None)

    def test_install(self):
        code, out, cfg = self.main()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [setup.FAMILIES["glm"]["hf"] + FILE])
        self.assertEqual(self.verified, [(FILE, *setup.FAMILIES["glm"]["sha256"][FILE])])
        self.assertEqual(self.built, [["120"]])
        self.assertEqual(len(self.runs), 2, self.runs)                 # the tokenizer and the draft block: no pack
        self.assertTrue(self.runs[0][1].endswith("strata_tokenizer.py"))
        pack = self.t / "data" / "packs" / "glm-3.0bit"
        self.assertEqual(self.runs[0][self.runs[0].index("--out") + 1], str(pack))
        # this size's file has no draft block: the model's is fetched, and the engine is given it
        block = self.t / "data" / "mtp" / "glm-5.3-flash" / "draft-block.gguf"
        self.assertTrue(self.runs[1][1].endswith("glm_draft_fetch.py"))
        self.assertEqual(self.runs[1][2:], ["fetch", "--out", str(block)])
        self.assertEqual((pack / "expert-profile.txt").read_text(), "3 0 7\n")
        args = cfg["args"]
        self.assertEqual(Path(cfg["exe"]), self.t / setup.ENGINE_POLY_DIR / setup.POLY_EXE)
        self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / "glm-3.0BIT" / FILE)
        self.assertEqual(args[args.index("--max-context") + 1], "32768")
        self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
        self.assertEqual(Path(args[args.index("--draft-model") + 1]), block)
        self.assertEqual(args[args.index("--conversation-cache-mib") + 1], "8192")
        for flag in ("--pack", "--native", "--mtp", "--kv", "--vision", "--pool-workers", "--prefill"):
            self.assertNotIn(flag, args)
        self.assertEqual(cfg["model_name"], "glm-5.3-flash-3.0bit")
        self.assertEqual(Path(cfg["tokenizer"]), pack / "tokenizer")
        self.assertEqual(cfg["lib_dirs"], ["/cuda/lib64"])
        self.assertNotIn("vision", cfg)
        self.assertEqual(cfg["gpu"], 0)                                # as main() does for every model

    def test_a_draft_block_that_is_there_is_not_fetched_again(self):
        block = self.t / "data" / "mtp" / "glm-5.3-flash" / "draft-block.gguf"
        block.parent.mkdir(parents=True)
        block.write_bytes(b"GGUF")
        code, out, cfg = self.main()
        self.assertEqual(code, 0, out)
        self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
        self.assertEqual(Path(cfg["args"][cfg["args"].index("--draft-model") + 1]), block)

    def test_the_engine_keeps_its_own_profile(self):
        pack = self.t / "data" / "packs" / "glm-3.0bit"
        pack.mkdir(parents=True)
        (pack / "expert-profile.txt").write_text("4 1 9\n")
        code, out, cfg = self.main()
        self.assertEqual(code, 0, out)
        self.assertEqual((pack / "expert-profile.txt").read_text(), "4 1 9\n")

    def test_the_questions(self):
        code, out, cfg = self.main(answers={"Context": "3", "Keep several": "n"})
        self.assertEqual(code, 0, out)
        self.assertEqual([q.split("?")[0] for q in self.asked],
                         ["Size", "Context", "Keep several conversations in RAM"])
        self.assertEqual(cfg["model_name"], "glm-5.3-flash-3.0bit")      # Enter: the whole model, with 128 GB
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "65536")
        self.assertNotIn("--conversation-cache-mib", cfg["args"])
        # a second run does not ask for the cache again and keeps it off
        code, out, cfg = self.main(answers={})
        self.assertEqual(code, 0, out)
        self.assertEqual([q.split("?")[0] for q in self.asked], ["Size", "Context"])
        self.assertNotIn("--conversation-cache-mib", cfg["args"])

    def test_the_cache_is_not_recommended_without_room(self):
        code, out, cfg = self.main(ram=100.0)                          # 96 GB wanted: no 8 GB beside it
        self.assertEqual(code, 0, out)
        self.assertNotIn("--conversation-cache-mib", cfg["args"])

    def test_less_ram_gets_the_half_expert_size(self):
        code, out, cfg = self.main(ram=63.7)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["model_name"], "glm-5.3-flash-reap50-q3_k_m")
        self.assertEqual(self.downloads, [setup.MODELS[HALF]["hf"] + HALF_FILE])
        self.assertEqual(self.verified, [(HALF_FILE, *setup.FAMILIES["glm"]["sha256"][HALF_FILE])])
        pack = self.t / "data" / "packs" / "glm-reap50-q3_k_m"
        self.assertEqual((pack / "expert-profile.txt").read_text(), "3 1 5\n")     # its own profile
        args = cfg["args"]
        self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / "glm-REAP50-Q3_K_M" / HALF_FILE)
        self.assertNotIn("--draft-model", args)                          # its file has its own, pruned with it
        self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])

    def test_a_size_in_several_files(self):
        code, out, cfg = self.main(ram=88.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["model_name"], "glm-5.3-flash-ud-iq2_xxs")
        self.assertEqual(self.downloads, [setup.MODELS[SPLIT]["hf"] + f for f in SPLIT_FILES])
        self.assertEqual(self.verified, [(f, *setup.FAMILIES["glm"]["sha256"][f]) for f in SPLIT_FILES])
        args = cfg["args"]                                               # the engine is given the first file
        self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / "glm-UD-IQ2_XXS" / SPLIT_FILES[0])
        pack = self.t / "data" / "packs" / "glm-ud-iq2_xxs"
        self.assertEqual((pack / "expert-profile.txt").read_text(), "3 0 7\n")     # the whole model's profile
        self.assertNotIn("--draft-model", args)                          # the draft block is in its last file
        self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])

    def test_deepseek(self):
        """DeepSeek V4 Flash: GLM's steps for Unsloth's three files, no draft block and no shipped profile."""
        fam = setup.FAMILIES["deepseek"]
        d = fam["models"]["UD-IQ2_XXS"]
        files = [f"DeepSeek-V4-Flash-0731-UD-IQ2_XXS-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
        self.assertRegex(d["hf"], r"^https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/[0-9a-f]{40}/"
                                  r"UD-IQ2_XXS/$")
        self.assertAlmostEqual(sum(fam["sha256"][f][0] for f in files) / 1e9, d["download_gb"], delta=0.05)
        self.assertAlmostEqual(setup.poly_ram_need(d, 31.84), d["ram_gb"], delta=0.5)
        code, out, cfg = self.main(family="deepseek")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [d["hf"] + f for f in files])
        self.assertEqual(self.verified, [(f, *fam["sha256"][f]) for f in files])
        self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
        args = cfg["args"]
        self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / "dsv4-UD-IQ2_XXS" / files[0])
        self.assertNotIn("--draft-model", args)
        self.assertFalse((self.t / "data" / "packs" / "dsv4-ud-iq2_xxs" / "expert-profile.txt").exists())
        self.assertEqual(cfg["model_name"], "deepseek-v4-flash-ud-iq2_xxs")

    def test_too_little_ram(self):
        code, out, cfg = self.main(ram=40.0)
        self.assertNotEqual(code, 0)
        self.assertIn("wants about 60 GB of RAM; this PC has 40", out)
        self.assertIsNone(cfg)
        self.assertEqual(self.built, [])
        code, out, cfg = self.main(ram=63.7, model=True)               # --model with --yes is the consent
        self.assertEqual(code, 0, out)
        self.assertIn("going on with 64 GB of RAM, as you chose", out)

    def test_a_smaller_card(self):
        code, out, cfg = self.main(ram=256.0, vram=23.9)
        self.assertEqual(code, 0, out)
        self.assertIn("has run on no card under 32 GB", out)
        code, out, cfg = self.main(ram=256.0, vram=11.9)
        self.assertNotEqual(code, 0)
        self.assertIn("needs more VRAM than 12 GB", out)

    def test_qwen_options_are_named(self):
        code, out, cfg = self.main(["--kv", "q4_0", "--vision", "yes", "--parallel", "2"])
        self.assertEqual(code, 0, out)
        line = next(ln for ln in out.splitlines() if "not used with GLM 5.3 Flash" in ln)
        for flag in ("--kv", "--vision"):
            self.assertIn(flag, line)
        self.assertNotIn("--parallel", line)   # strata-poly answers several requests at once too
        self.assertEqual(cfg["parallel"], 2)

    def test_two_cards(self):
        code, out, cfg = self.main(["--gpu", "1"], n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertEqual((cfg["gpu"], cfg["gpus_asked"]), (1, True))

    def test_layer_split(self):
        code, out, cfg = self.main(["--gpus", "0,1"], n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertEqual((cfg["gpu"], cfg["gpus_asked"], cfg["layer_split"]), ([0, 1], True, "auto"))
        self.assertNotIn("runs on one GPU", out)
        code, out, cfg = self.main(["--gpus", "0,1", "--layer-split", "30"], n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["layer_split"], "30")
        self.assertNotIn("--layer-split", next((ln for ln in out.splitlines() if "not used with" in ln), ""))

    def test_windows_is_asked(self):
        code, out, cfg = self.main(win=True, answers={})
        self.assertNotEqual(code, 0)
        self.assertIn("has not been tried on Windows", out)
        self.assertEqual(self.built, [])

    def test_windows_is_not_asked_for_a_family_that_ran_there(self):
        for family in ("qwen36", "ornith", "gemma4", "tiel"):
            with self.subTest(family):
                self.built = []
                code, out, cfg = self.main(win=True, family=family, ram=32.0, vram=8.0)
                self.assertEqual(code, 0, out)
                self.assertNotIn("Windows", out)
                self.assertEqual(len(self.built), 1)

    def test_cuda_12_compiles_the_engine_with_it(self):
        code, out, cfg = self.main(["--cuda", "12"])
        self.assertEqual(code, 0, out)
        self.assertEqual((self.built, self.toolkit), ([["120"]], 12))
        self.assertIn("with CUDA 12 is compiled here", out)
        self.assertEqual(cfg["cuda"], 12)
        # a copy set up from this install keeps it; the same one set up again chooses again
        self.assertEqual(setup.choices_from_config(next(self.t.glob("strata-glm*.json")))["cuda"], 12)
        code, out, cfg = self.main()
        self.assertEqual((cfg.get("cuda"), self.toolkit), (None, 13))


if __name__ == "__main__":
    unittest.main()
