"""Tests for setup.py's Ornith-1.5-122B support (ornith122 family): pinned revisions, manifest,
byte counts, SHA-256 digests, VRAM/RAM hardware calculations, --check verdicts, subdirectory
and flat --gguf-dir resolution, and non-interactive setup flow with mocked hardware.

    python -m unittest tools.test_setup_ornith122
"""
from __future__ import annotations

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
FAMILY = "ornith122"
GGUF_REPO = "unsloth/Ornith-1.5-122B-GGUF"
BASE_REPO = "ornith-ai/Ornith-1.5-122B"
EXPECTED_GGUF_SHA = "3842d003341507c216295c6f28dfcd30349925c6"
EXPECTED_BASE_SHA = "dfa07bab90e90746ffbfc0b0cba3126eb9265de2"


def quiet(fn, *args, **kw):
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args, **kw)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class PinsAndManifest(unittest.TestCase):
    def test_revisions_pinned(self):
        self.assertIn(GGUF_REPO, setup.HF_REVISIONS)
        self.assertEqual(setup.HF_REVISIONS[GGUF_REPO], EXPECTED_GGUF_SHA)
        self.assertIn(BASE_REPO, setup.HF_REVISIONS)
        self.assertEqual(setup.HF_REVISIONS[BASE_REPO], EXPECTED_BASE_SHA)

    def test_family_metadata(self):
        self.assertIn(FAMILY, setup.FAMILIES)
        fam = setup.FAMILIES[FAMILY]
        self.assertEqual(fam["engine"], "poly")
        self.assertEqual(fam["tag"], "ornith122-")
        self.assertEqual(fam["name"], "ornith-1.5-122b")
        self.assertEqual(fam["title"], "Ornith 1.5 122B")
        self.assertTrue(fam["vision"])
        self.assertTrue(fam["windows"])
        self.assertTrue(fam["experimental"])
        self.assertEqual(fam["ran_on_gb"], 16)
        self.assertEqual(fam["row_bytes"], 26624)
        self.assertEqual(fam["mmproj"], "mmproj-F16.gguf")
        self.assertEqual(fam["vision_gb"], 0.9)
        self.assertEqual(fam["vision_vram_gb"], 1.6)
        self.assertRegex(fam["hf"], SHA)
        self.assertRegex(fam["mmproj_hf"], SHA)
        self.assertNotIn("/resolve/main/", fam["hf"])
        self.assertNotIn("/resolve/main/", fam["mmproj_hf"])

    def test_tag_isolation(self):
        fam = setup.FAMILIES[FAMILY]
        tags = [d["tag"] for d in setup.FAMILIES.values() if d["tag"]]
        for a in tags:
            self.assertEqual([b for b in tags if b.startswith(a)], [a])
        self.assertFalse(fam["tag"].startswith(setup.FAMILIES["ornith"]["tag"]))

    def test_models_manifest(self):
        fam = setup.FAMILIES[FAMILY]
        expected_models = ["UD-Q3_K_XL", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q4_K_XL"]
        self.assertEqual(list(setup.family_models(FAMILY)), expected_models)

        # Shard counts and model file paths
        self.assertEqual(setup.model_shards(fam, "UD-Q3_K_XL"), 3)
        self.assertEqual(setup.model_shards(fam, "UD-Q2_K_XL"), 1)
        self.assertEqual(setup.model_shards(fam, "UD-IQ3_XXS"), 1)
        self.assertEqual(setup.model_shards(fam, "UD-Q4_K_XL"), 3)

        self.assertEqual(setup.model_file(fam, "UD-Q3_K_XL", 1),
                         "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf")
        self.assertEqual(setup.model_file(fam, "UD-Q3_K_XL", 2),
                         "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00002-of-00003.gguf")
        self.assertEqual(setup.model_file(fam, "UD-Q3_K_XL", 3),
                         "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00003-of-00003.gguf")

        self.assertEqual(setup.model_file(fam, "UD-Q2_K_XL", 1),
                         "Ornith-1.5-122B-UD-Q2_K_XL.gguf")
        self.assertEqual(setup.model_file(fam, "UD-IQ3_XXS", 1),
                         "Ornith-1.5-122B-UD-IQ3_XXS.gguf")

        self.assertEqual(setup.model_file(fam, "UD-Q4_K_XL", 1),
                         "UD-Q4_K_XL/Ornith-1.5-122B-UD-Q4_K_XL-00001-of-00003.gguf")

        for m, d in fam["models"].items():
            self.assertEqual(d["draft"], "own")
            self.assertEqual(d["vram_beside_gb"], 8.5)
            self.assertLess(d["arena_gb"], d["download_gb"])

    def test_sha256_checksums_and_sizes(self):
        fam = setup.FAMILIES[FAMILY]
        expected_files = {
            "Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf":
                (10942929, "d8a23c487741fdac7026e10013fa8e320502cf01048a2333b2db8ba2a9f74fe5"),
            "Ornith-1.5-122B-UD-Q3_K_XL-00002-of-00003.gguf":
                (49791670080, "95f76e67ffc49e2b0b4a4dfe5c42269b4b275ecf9e3696f599f3eeac92180323"),
            "Ornith-1.5-122B-UD-Q3_K_XL-00003-of-00003.gguf":
                (9958641536, "1c4b52bbb1d45a21b5b7927429a5c80f264c688f2d14ed50ba0b6da0260d4028"),
            "Ornith-1.5-122B-UD-Q2_K_XL.gguf":
                (44533952768, "93fb32cfb1c31b34df426b4a41d961bdaffcddbbd217d850262136cf3689ba66"),
            "Ornith-1.5-122B-UD-IQ3_XXS.gguf":
                (49515667712, "027a9a480e8bcc6079fcf834e368467a169953561fb4b98dc9528d6185c8a438"),
            "Ornith-1.5-122B-UD-Q4_K_XL-00001-of-00003.gguf":
                (10942929, "7159d31a6a6ce5528500b490880383844c3c9b21fcdaad94e66c9e65233ff359"),
            "Ornith-1.5-122B-UD-Q4_K_XL-00002-of-00003.gguf":
                (49971099680, "e66952128394bb4c75554c7e10a12a8b63dd1c7f049d876a02885a946e30c1c3"),
            "Ornith-1.5-122B-UD-Q4_K_XL-00003-of-00003.gguf":
                (29731729056, "21325ec18a9e3210eeffada0fc327a25a79171918f1d04f8f0a06b61866ac9ba"),
            "mmproj-F16.gguf":
                (908724544, "39db176176a904b101f5173a4f03e785990f9279f94ab373c4fd629ae7011817"),
        }
        self.assertEqual(fam["sha256"], expected_files)
        for fname, (size, digest) in fam["sha256"].items():
            self.assertGreater(size, 0)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

        # Check total download_gb matches byte counts
        q3_bytes = sum(fam["sha256"][Path(setup.model_file(fam, "UD-Q3_K_XL", i)).name][0] for i in range(1, 4))
        self.assertAlmostEqual(q3_bytes / 1e9, fam["models"]["UD-Q3_K_XL"]["download_gb"], delta=0.05)

        q2_bytes = fam["sha256"][setup.model_file(fam, "UD-Q2_K_XL", 1)][0]
        self.assertAlmostEqual(q2_bytes / 1e9, fam["models"]["UD-Q2_K_XL"]["download_gb"], delta=0.05)

        iq3_bytes = fam["sha256"][setup.model_file(fam, "UD-IQ3_XXS", 1)][0]
        self.assertAlmostEqual(iq3_bytes / 1e9, fam["models"]["UD-IQ3_XXS"]["download_gb"], delta=0.05)

        q4_bytes = sum(fam["sha256"][Path(setup.model_file(fam, "UD-Q4_K_XL", i)).name][0] for i in range(1, 4))
        self.assertAlmostEqual(q4_bytes / 1e9, fam["models"]["UD-Q4_K_XL"]["download_gb"], delta=0.05)

        mmproj_bytes = fam["sha256"][fam["mmproj"]][0]
        self.assertAlmostEqual(mmproj_bytes / 1e9, fam["vision_gb"], delta=0.05)


class MemoryEstimates(unittest.TestCase):
    def setUp(self):
        self.fam = setup.FAMILIES[FAMILY]

    def test_poly_beside_gb(self):
        for d in self.fam["models"].values():
            self.assertEqual(setup.poly_beside_gb(d), 8.5)

    def test_minimum_vram_floor(self):
        # Requires vram >= beside + 2 = 10.5 GB. 8 GB card fails regardless of RAM.
        for d in self.fam["models"].values():
            self.assertFalse(setup.poly_fits(d, ram=128, vram_gb=8.0, hip=False))
            self.assertFalse(setup.poly_fits(d, ram=128, vram_gb=10.0, hip=False))
            self.assertTrue(setup.poly_fits(d, ram=128, vram_gb=16.0, hip=False))

    def test_held_and_ram_need_calculations(self):
        # On a 24 GB card: (24.0 - 8.5) * 1.073741824 = 16.643 GB
        held_24 = (24.0 - 8.5) * 1.073741824
        for m, d in self.fam["models"].items():
            self.assertAlmostEqual(setup.poly_held_gb(d, 24.0), held_24, delta=0.01)
            calculated_ram = d["arena_gb"] - held_24 + setup.POLY_RAM_BESIDE_GB
            self.assertEqual(round(calculated_ram), d["ram_gb"], f"{m} ram_gb mismatch")

    def test_poly_default_selection(self):
        sizes = self.fam["models"]
        # 24 GB VRAM, 64 GB RAM: chooses UD-Q3_K_XL (daily choice, first size)
        self.assertEqual(setup.poly_default(sizes, ram=64, vram_gb=24.0), "UD-Q3_K_XL")

        # 16 GB VRAM, 48 GB RAM: UD-Q3_K_XL needs ~55 GB RAM (> 48), picks UD-Q2_K_XL (needs ~40.5 GB)
        self.assertEqual(setup.poly_default(sizes, ram=48, vram_gb=16.0), "UD-Q2_K_XL")

        # 16 GB VRAM, 32 GB RAM: neither meets RAM requirement, picks UD-Q2_K_XL (lowest RAM need)
        self.assertEqual(setup.poly_default(sizes, ram=32, vram_gb=16.0), "UD-Q2_K_XL")


class CheckVerdicts(unittest.TestCase):
    def setUp(self):
        self.fam = setup.FAMILIES[FAMILY]
        self.q3 = self.fam["models"]["UD-Q3_K_XL"]

    def test_small_gpu_verdict(self):
        v = setup.poly_verdict(self.q3, ram=64, vram_gb=8.0, hip=False)
        self.assertIn("does not fit: it takes about 8.5 GB of VRAM before any expert", v)

    def test_sufficient_hardware_verdict(self):
        v = setup.poly_verdict(self.q3, ram=64, vram_gb=24.0, hip=False, windows=bool(self.fam.get("windows")))
        self.assertIn("fits by setup's estimate (~47 GB of RAM on this card)", v)

    def test_ssd_fallback_verdict(self):
        v = setup.poly_verdict(self.q3, ram=32, vram_gb=16.0, hip=False, windows=bool(self.fam.get("windows")))
        self.assertIn("read from the SSD", v)


class SubdirectoryDownloadResolution(unittest.TestCase):
    def test_relative_path_for_sharded_and_flat_models(self):
        fam = setup.FAMILIES[FAMILY]
        models_dir = Path("models/ornith122-ud-q3_k_xl")

        # Multi-shard model with subdirectory in file pattern
        s1 = models_dir / setup.model_file(fam, "UD-Q3_K_XL", 1)
        rel1 = s1.relative_to(models_dir).as_posix() if s1.is_relative_to(models_dir) else s1.name
        self.assertEqual(rel1, "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf")
        url1 = fam["hf"] + rel1
        self.assertTrue(url1.endswith("/UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf"))
        self.assertNotIn("UD-Q3_K_XL/UD-Q3_K_XL", url1)

        # Single-shard model with flat root file pattern
        s_flat = models_dir / setup.model_file(fam, "UD-Q2_K_XL", 1)
        rel_flat = s_flat.relative_to(models_dir).as_posix() if s_flat.is_relative_to(models_dir) else s_flat.name
        self.assertEqual(rel_flat, "Ornith-1.5-122B-UD-Q2_K_XL.gguf")
        url_flat = fam["hf"] + rel_flat
        self.assertTrue(url_flat.endswith("/Ornith-1.5-122B-UD-Q2_K_XL.gguf"))


class GgufDirResolution(unittest.TestCase):
    def test_gguf_dir_subdirectory_layout(self):
        fam = setup.FAMILIES[FAMILY]
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "UD-Q3_K_XL"
            sub.mkdir()
            for i in range(1, 4):
                (sub / f"Ornith-1.5-122B-UD-Q3_K_XL-0000{i}-of-00003.gguf").write_bytes(b"")
            shards = setup.gguf_dir_shards(Path(d), fam, "UD-Q3_K_XL")
            self.assertEqual(len(shards), 3)
            self.assertEqual([s.name for s in shards],
                             [f"Ornith-1.5-122B-UD-Q3_K_XL-0000{i}-of-00003.gguf" for i in range(1, 4)])
            self.assertEqual(shards[0].parent, sub)

    def test_gguf_dir_flat_layout(self):
        fam = setup.FAMILIES[FAMILY]
        with tempfile.TemporaryDirectory() as d:
            for i in range(1, 4):
                (Path(d) / f"Ornith-1.5-122B-UD-Q3_K_XL-0000{i}-of-00003.gguf").write_bytes(b"")
            shards = setup.gguf_dir_shards(Path(d), fam, "UD-Q3_K_XL")
            self.assertEqual(len(shards), 3)
            self.assertEqual([s.name for s in shards],
                             [f"Ornith-1.5-122B-UD-Q3_K_XL-0000{i}-of-00003.gguf" for i in range(1, 4)])
            self.assertEqual(shards[0].parent, Path(d))

    def test_gguf_dir_single_shard(self):
        fam = setup.FAMILIES[FAMILY]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "Ornith-1.5-122B-UD-Q2_K_XL.gguf"
            f.write_bytes(b"")
            shards = setup.gguf_dir_shards(Path(d), fam, "UD-Q2_K_XL")
            self.assertEqual(shards, [f])


class NonInteractiveInstallFlow(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.t = Path(self.td.name)
        self.downloads = []
        self.verified_downloads = []
        self.verified_own = []
        self.runs = []

    def tearDown(self):
        self.td.cleanup()

    def run_setup(self, argv: list[str]):
        found_gpus = [{"index": 0, "name": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0, "arch": "89",
                       "archs": ["89"], "driver": "580.97"}]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"GGUF")
            setup.mark(dst)

        def fake_build(gpu, yes, llama, vision="none", toolkit=13):
            eng = self.t / setup.ENGINE_POLY_DIR
            eng.mkdir(parents=True, exist_ok=True)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "cuda_dirs": ["/cuda/lib64"]}))
            return eng

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found_gpus),
            mock.patch.object(setup, "amd_gpus", lambda *a: []),
            mock.patch.object(setup, "ram_gb", lambda: 64.0),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "cpu_cores", lambda: (8, 0)),
            mock.patch.object(setup, "page_file_gb", lambda: 32.0),
            mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "build_engine_poly", fake_build),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "whole_shard", lambda s: True),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256",
                              lambda s, size, sha, own=False: self.verified_downloads.append((s.name, size, sha))),
            mock.patch.object(setup, "verify_poly_download",
                              lambda s, size, sha, fam: self.verified_downloads.append((s.name, size, sha))),
            mock.patch.object(setup, "verify_poly_own",
                              lambda s, size, sha, fam: self.verified_own.append((s.name, size, sha))),
            mock.patch.object(setup, "run", lambda cmd, *a, **k: self.runs.append([str(x) for x in cmd])),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.bat"),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.object(sys, "argv", ["setup.py", *argv]),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)

        cfg_path = self.t / "strata-ornith122-ud-q3_k_xl.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
        return code, out, cfg

    def test_non_interactive_setup_download_flow(self):
        code, out, cfg = self.run_setup([
            "--family", "ornith122",
            "--model", "UD-Q3_K_XL",
            "--vision", "yes",
            "--yes", "--no-start",
            "--models-dir", str(self.t / "models"),
        ])
        self.assertEqual(code, 0, out)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["model_name"], "ornith-1.5-122b-ud-q3_k_xl")
        self.assertEqual(cfg["display_name"], "Ornith 1.5 122B UD-Q3_K_XL")
        self.assertIn("--model", cfg["args"])
        model_idx = cfg["args"].index("--model")
        self.assertTrue(cfg["args"][model_idx + 1].endswith("Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf"))
        self.assertNotIn("--draft-model", cfg["args"])

        # Shards and mmproj downloaded with relative URLs
        fam = setup.FAMILIES[FAMILY]
        expected_urls = [
            fam["hf"] + "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf",
            fam["hf"] + "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00002-of-00003.gguf",
            fam["hf"] + "UD-Q3_K_XL/Ornith-1.5-122B-UD-Q3_K_XL-00003-of-00003.gguf",
            fam["mmproj_hf"] + "mmproj-F16.gguf",
        ]
        self.assertEqual(self.downloads, expected_urls)

    def test_non_interactive_setup_gguf_dir_flow(self):
        gguf_dir = self.t / "local_gguf"
        sub = gguf_dir / "UD-Q3_K_XL"
        sub.mkdir(parents=True)
        for i in range(1, 4):
            (sub / f"Ornith-1.5-122B-UD-Q3_K_XL-0000{i}-of-00003.gguf").write_bytes(b"GGUF")
        (gguf_dir / "mmproj-F16.gguf").write_bytes(b"MMPROJ")

        code, out, cfg = self.run_setup([
            "--family", "ornith122",
            "--model", "UD-Q3_K_XL",
            "--gguf-dir", str(gguf_dir),
            "--yes", "--no-start",
        ])
        self.assertEqual(code, 0, out)
        self.assertIsNotNone(cfg)
        self.assertEqual(self.downloads, [])  # Zero downloads when local files are present
        self.assertEqual(len(self.verified_own), 3)  # Shards verified via verify_poly_own


if __name__ == "__main__":
    unittest.main()
