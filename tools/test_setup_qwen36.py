"""Tests for setup.py's Qwen3.6-35B-A3B install (a family with its own engine path: OWN_PATH_FAMILIES): the pinned and
hashed files, the menu entry, steps 2-7 of install_own_path (size, context, the engine's --families check, the
download, the tokenizer, the config) and reading such a config back.  Mocked engine, downloads and processes -
nothing is built, fetched or started.

    python -m unittest tools.test_setup_qwen36
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/")
REPO = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"


def args(**kw):
    base = dict(family="qwen36-strata", model=None, check=False, yes=True, vision=None, models_dir=None, gguf_dir=None,
                context=None, rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram="auto",
                vram_reserve_mib=None, host=None, api_key=None, browser=None, no_start=True, data_dir=None, gpu=None)
    base.update(kw)
    return argparse.Namespace(**base)


def run(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            return fn(*a, **kw), out.getvalue()
        except SystemExit as e:
            return e.code, out.getvalue()


class Qwen36Entries(unittest.TestCase):
    def test_pinned_and_hashed(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36-strata"]
        self.assertIs(setup.PREVIEW_FAMILIES, setup.OWN_PATH_FAMILIES)   # the earlier name still works
        self.assertEqual(setup.HF_REVISIONS[REPO], "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d")
        for key in ("hf", "mmproj_hf"):
            self.assertRegex(fam[key], SHA)
            self.assertIn(REPO, fam[key])
        self.assertEqual(fam["arch"], "qwen35moe")
        self.assertEqual(list(fam["models"]), ["UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0"])
        for m in fam["models"]:
            size, sha = fam["sha256"][fam["file"].format(q=m)]
            self.assertGreater(size, 20e9)
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertEqual(fam["sha256"]["Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"][0], 22853663008)

    def test_not_a_pack_family(self):
        # FAMILIES are Flash-Next files with packs; Qwen3.6 is installed by install_own_path
        self.assertNotIn("qwen36-strata", setup.FAMILIES)
        for m, d in setup.MODELS.items():
            self.assertNotIn("qwen36-strata", d.get("families", ()))

    def test_size_recommendation(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36-strata"]
        self.assertEqual(setup.own_path_size(fam, 24.0), "UD-Q4_K_XL")   # none fits: the smallest
        self.assertEqual(setup.own_path_size(fam, 32.0), "UD-Q4_K_XL")
        self.assertEqual(setup.own_path_size(fam, 46.0), "UD-Q6_K_XL")
        self.assertEqual(setup.own_path_size(fam, 64.0), "Q8_0")

    def test_context_recommendation(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36-strata"]
        self.assertEqual(setup.own_path_context(fam, 12.0), 32768)
        self.assertEqual(setup.own_path_context(fam, 16.0), 32768)
        self.assertEqual(setup.own_path_context(fam, 24.0), 65536)
        self.assertEqual(setup.own_path_context(fam, 32.0), 131072)


class InstallOwnPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.families = {"qwen4exp", "qwen35moe"}
        eng = self.root / "engine"
        self.cfg = {}

        def engine_step(a, gpu, hip, amd, cuda_tk, vision):
            self.calls.append(("engine", vision))
            if vision != "none":                                   # the image encoder beside the engine
                eng.mkdir(parents=True, exist_ok=True)
                (eng / setup.VEXE).write_text("x")
            return self.root / "llama", eng, {"version": "0.1.44"}, ["/libs"], gpu, vision

        def write_cfg(path, cfg, source=None):
            self.cfg = cfg
            self.calls.append(("config", path.name))

        patches = [mock.patch.object(setup, "download", lambda url, dst, what=None: self.calls.append(("get", url, dst))),
                   mock.patch.object(setup, "verify_sha256", lambda s, size, sha, own=False: self.calls.append(("sha", s.name))),
                   mock.patch.object(setup, "free_gb", lambda p: 500.0),
                   mock.patch.object(setup, "pip_install", lambda *a, **k: None),
                   mock.patch.object(setup, "engine_step", engine_step),
                   mock.patch.object(setup, "engine_families", lambda e: self.families),
                   mock.patch.object(setup, "run", lambda cmd, **k: self.calls.append(("run", cmd))),
                   mock.patch.object(setup, "data_folder", lambda d: (self.root / "data", [])),
                   mock.patch.object(setup, "write_setup_config", write_cfg),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "recommend_pool_workers", lambda a: a),
                   mock.patch.object(setup, "ROOT", self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def install(self, ram=64.0, vram=16.0, hip=False, **kw):
        gpu = {"index": 0, "count": 1, "vram_gb": vram, "name": "Test GPU"}
        return run(setup.install_own_path, args(models_dir=str(self.root / "models"), **kw), "qwen36-strata", gpu, hip, [],
                   13, [], 8080, ram, None)

    def test_installs_the_pinned_file_and_writes_the_config(self):
        rc, out = self.install(model="UD-Q4_K_XL", vision="no")
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        self.assertEqual(len(get), 1)
        gguf = self.root / "models" / "qwen36s-UD-Q4_K_XL" / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
        self.assertEqual(get[0][1], setup.hf(REPO) + gguf.name)
        self.assertEqual(get[0][2], gguf)
        self.assertIn(("sha", gguf.name), self.calls)
        self.assertIn(("engine", "none"), self.calls)            # --vision no: no image encoder
        self.assertNotIn("--vision", self.cfg["args"])
        self.assertNotIn("vision", self.cfg)
        tok = [c for c in self.calls if c[0] == "run"]
        self.assertTrue(tok and "strata_tokenizer.py" in str(tok[0][1]))
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--native") + 1], str(gguf))
        self.assertEqual(a[a.index("--max-context") + 1], "32768")   # 16 GB card
        self.assertEqual(a[a.index("--mtp") + 1], "2")
        self.assertNotIn("--pack", a)
        self.assertNotIn("--expert-cache", a)
        self.assertEqual(self.cfg["arch"], "qwen35moe")
        self.assertEqual(self.cfg["model_name"], "qwen3.6-35b-a3b-ud-q4_k_xl")
        self.assertEqual(Path(self.cfg["exe"]).parent, self.root / "engine")
        self.assertIn(("config", "strata-qwen36s-ud-q4_k_xl.json"), self.calls)

    def test_default_size_follows_the_ram(self):
        rc, out = self.install(ram=64.0)
        self.assertEqual(rc, 0, out)
        self.assertIn("Recommended for 64 GB of RAM: Q8_0", out)
        self.assertTrue(self.cfg["args"][self.cfg["args"].index("--native") + 1].endswith("Qwen3.6-35B-A3B-Q8_0.gguf"))
        rc, out = self.install(ram=46.0)
        self.assertTrue(self.cfg["args"][self.cfg["args"].index("--native") + 1].endswith("UD-Q6_K_XL.gguf"))

    def test_a_bigger_size_than_the_ram_is_asked_not_refused(self):
        with mock.patch.object(setup, "confirm_risk", lambda *a, **k: None):
            rc, out = self.install(ram=32.0, model="Q8_0")
        self.assertEqual(rc, 0, out)
        self.assertIn("installing Q8_0 with 32 GB of RAM, as you chose", out)

    def test_an_engine_without_the_family_is_refused_before_the_download(self):
        self.families = {"qwen4exp"}
        rc, out = self.install(model="UD-Q4_K_XL")
        self.assertEqual(rc, 1)
        self.assertIn("does not run Qwen3.6-35B-A3B yet", out)
        self.assertEqual([c for c in self.calls if c[0] == "get"], [])

    def test_a_flash_next_size_is_refused(self):
        rc, out = self.install(model="IQ3_XXS")
        self.assertEqual(rc, 1)
        self.assertIn("has no IQ3_XXS", out)
        self.assertEqual(self.calls, [])

    def test_too_little_disk(self):
        with mock.patch.object(setup, "free_gb", lambda p: 10.0):
            rc, out = self.install(model="Q8_0")
        self.assertEqual(rc, 1)
        self.assertIn("not enough free disk space", out)

    def test_bigger_card_bigger_context(self):
        rc, out = self.install(model="UD-Q4_K_XL", vram=32.0, vision="no")
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "131072")

    def test_images_by_default_on_a_16gb_nvidia_card_use_the_gpu_encoder(self):
        rc, out = self.install(model="UD-Q4_K_XL", vram=16.0)               # --yes: the question's default answer
        self.assertEqual(rc, 0, out)
        self.assertIn(("engine", "gpu"), self.calls)
        mm = self.root / "models" / "qwen36s-UD-Q4_K_XL" / "mmproj-F16.gguf"
        self.assertIn(("get", setup.hf(REPO) + "mmproj-F16.gguf", mm), self.calls)
        self.assertIn(("sha", "mmproj-F16.gguf"), self.calls)               # pinned: size + SHA-256 (899283584 bytes)
        self.assertEqual(setup.OWN_PATH_FAMILIES["qwen36-strata"]["sha256"]["mmproj-F16.gguf"][0], 899283584)
        a = self.cfg["args"]
        self.assertIn("--vision", a)
        v = self.cfg["vision"]
        self.assertTrue(v["gpu"])
        self.assertEqual(v["mmproj"], str(mm))
        self.assertTrue(v["model"].endswith("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"))   # the text model, opened vocab-only
        self.assertEqual(Path(v["exe"]).name, setup.VEXE)
        self.assertEqual(v["max_tokens"], 1024)
        self.assertIn("images: on", out)

    def test_a_12gb_card_is_recommended_the_cpu_encoder_never_forced(self):
        rc, out = self.install(model="UD-Q4_K_XL", vram=12.0)               # the default answer is no
        self.assertEqual(rc, 0, out)
        self.assertNotIn("vision", self.cfg)
        rc, out = self.install(model="UD-Q4_K_XL", vram=12.0, vision="yes")  # yes: the CPU encoder, said why
        self.assertEqual(rc, 0, out)
        self.assertFalse(self.cfg["vision"]["gpu"])
        self.assertEqual(self.cfg["vision"]["max_tokens"], 300)
        self.assertIn("threads", self.cfg["vision"])
        rc, out = self.install(model="UD-Q4_K_XL", vram=12.0, vision="gpu")  # gpu asked for: kept, with a note
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.cfg["vision"]["gpu"])
        self.assertIn("kept as you chose", out)

    def test_amd_reads_pictures_on_the_cpu(self):
        with mock.patch.object(setup, "WIN", False):
            rc, out = self.install(model="UD-Q4_K_XL", vram=32.0, hip=True, vision="yes")
        self.assertEqual(rc, 0, out)
        self.assertFalse(self.cfg["vision"]["gpu"])
        self.assertIn("--vision", self.cfg["args"])
        with mock.patch.object(setup, "WIN", True):
            rc, out = self.install(model="UD-Q4_K_XL", vram=32.0, hip=True, vision="cpu")
        self.assertNotIn("vision", self.cfg)
        self.assertIn("Linux-only", out)

    def test_parallel_is_opt_in_and_recommended_where_the_experts_fit(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36-strata"]
        # nothing written without --parallel; the recommendation is said
        rc, out = self.install(model="UD-Q4_K_XL", vram=32.0, vision="no", context=32768)
        self.assertEqual(rc, 0, out)
        self.assertNotIn("parallel", self.cfg)
        self.assertIn("--parallel 3", out)
        # a 12 GB card runs the hybrid mode: parallel is offered there too, with its price said (never forced)
        # (measured: 12 GB at 8K, 4 slots: a request alone -4%; 16 GB at 32K, 2 slots -9 to -11%, 4 slots no gain together)
        self.assertEqual(setup.own_path_parallel_recommend(fam, "UD-Q4_K_XL", 12.0, 8192), 4)
        self.assertEqual(setup.own_path_parallel_recommend(fam, "UD-Q4_K_XL", 12.0, 32768), 0)
        self.assertEqual(setup.own_path_parallel_recommend(fam, "UD-Q4_K_XL", 16.0, 32768), 2)
        self.assertEqual(setup.own_path_parallel_recommend(fam, "UD-Q4_K_XL", 12.0, 131072), 0)
        self.assertEqual(setup.own_path_parallel_recommend(fam, "UD-Q4_K_XL", 8.0, 32768), 0)
        # a card that holds every expert for ONE session only is not offered more: 4 slots on a 32 GB R9700 at 128K left
        # 15 of 256 experts per layer in VRAM and decode ran 81 tok/s against ~180
        self.assertEqual(setup.own_path_parallel_recommend(fam, "UD-Q4_K_XL", 31.9, 131072), 0)
        rc, out = self.install(model="UD-Q4_K_XL", vram=12.0, vision="no", context=8192)
        self.assertNotIn("parallel", self.cfg)
        self.assertIn("--parallel 4", out)
        self.assertIn("does not hold every expert", out)
        self.assertIn("a request alone runs about 4-11% slower", out)
        self.assertIn("WITH the draft layer", out)
        rc, out = self.install(model="UD-Q4_K_XL", vram=12.0, vision="no", context=8192, parallel=6)
        self.assertEqual(self.cfg["parallel"], 6)
        self.assertIn("recommended for this card: 4", out)
        self.assertIn("kept as you chose", out)
        rc, out = self.install(model="UD-Q4_K_XL", vram=8.0, vision="no", parallel=4)
        self.assertEqual(self.cfg["parallel"], 4)
        self.assertIn("recommended for this card: one at a time", out)
        self.assertIn("kept as you chose", out)
        rc, out = self.install(model="UD-Q4_K_XL", vram=12.0, vision="no", context=8192, parallel=1)
        self.assertEqual(self.cfg["parallel"], 1)
        rc, out = self.install(model="UD-Q4_K_XL", vram=32.0, vision="no", parallel=12)
        self.assertIn("at most 8", out)
        self.assertEqual(self.cfg["parallel"], 12)             # written as asked; the server and the engine cap it
        self.assertGreater(setup.own_path_parallel_slot_gb(fam, 32768), 1.4)

    def test_parallel_is_asked_when_attended_and_enter_keeps_one_at_a_time(self):
        asked = []

        def ask(question, choices, default, yes):
            if "requests at once" not in question:
                return default                                   # size, context, images: the recommended answers
            asked.append((question, default))
            return answers.pop(0)
        for answers, want in ((["1"], None), (["3"], 3)):
            self.cfg = {}
            with mock.patch.object(setup, "ask", ask):
                rc, out = self.install(model="UD-Q4_K_XL", vram=12.0, vision="no", context=8192, yes=False)
            self.assertEqual(rc, 0, out)
            self.assertEqual(self.cfg.get("parallel"), want)
            self.assertIn("--parallel 4", out)                  # the offer and its price are said before the question
            self.assertEqual(out.count("Several requests at once (opt-in)"), 1)
        self.assertTrue(all(d == "1" for _, d in asked))        # Enter = one at a time
        # a card with no room for a second session is not asked
        asked.clear()
        with mock.patch.object(setup, "ask", ask):
            rc, out = self.install(model="UD-Q4_K_XL", vram=8.0, vision="no", yes=False)
        self.assertEqual(asked, [])
        self.assertNotIn("parallel", self.cfg)

    def test_config_read_back(self):
        self.install(model="UD-Q6_K_XL", context=65536)
        p = self.root / "strata-qwen36s-ud-q6_k_xl.json"
        p.write_text(json.dumps(self.cfg), encoding="utf-8")
        ch = setup.choices_from_config(p)
        self.assertEqual((ch["family"], ch["model"], ch["context"]), ("qwen36-strata", "UD-Q6_K_XL", 65536))
        self.assertEqual(ch["vision"], "gpu")                  # 16 GB NVIDIA, the default answer
        self.assertTrue(setup.own_path_config(self.cfg))
        self.assertEqual(setup.calibration_targets([p], None, None), [])   # the tuning is Flash-Next's


if __name__ == "__main__":
    unittest.main()
