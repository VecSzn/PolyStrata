"""Offline Ornith 397B installation contracts; no model downloads or GPU required."""
import unittest
import tempfile
import sys
import json
import os
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup
from tools import test_setup_glm as harness


class Ornith397(unittest.TestCase):
    def install(self, quant, *args, **kwargs):
        h = harness.Main(methodName="runTest")
        h.setUp()
        self.addCleanup(h.tearDown)
        code, out, cfg = h.main(args, family="ornith397", model=quant, **kwargs)
        self.assertEqual(code, 0, out)
        return h, out, cfg

    def test_windows_amd_selects_the_hip_device_and_allows_paging(self):
        card = {"index": 1, "name": "AMD Radeon RX 9070 XT", "arch": "gfx1201", "vram_gb": 15.9,
                "driver": "windows", "count": 2}
        with mock.patch.object(setup, "hip_devices", return_value=[{**card, "index": 0, "driver": "hip"},
                                                                  {**card, "arch": "gfx1036"}]):
            _, out, cfg = self.install("IQ1_S", "--backend", "hip", "--vision", "no", win=True,
                                       ram=63, amd_cards=[card])
        self.assertEqual(cfg["backend"], "hip")
        self.assertEqual(cfg["hip_ordinal"], 0)
        self.assertEqual(cfg["gpu"], 0)
        self.assertIn("--kv-whole", cfg["args"])
        self.assertIn("going on with 63 GB of RAM", out)
        self.assertNotIn("vision", cfg)

    def test_all_sizes_install_and_round_trip(self):
        fam = setup.FAMILIES["ornith397"]
        expected = {"IQ1_S": (3, 81.78), "IQ1_M": (3, 91.36), "IQ2_XXS": (3, 106.31),
                    "IQ2_XS": (3, 118.47), "IQ2_S": (4, 120.51), "IQ2_M": (4, 133.13)}
        for q, (count, gb) in expected.items():
            with self.subTest(q=q):
                h, out, cfg = self.install(q)
                files = [f"Ornith-1.5-397B-{q}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]
                self.assertEqual([Path(u).name for u in h.downloads], files)
                self.assertTrue(all("/217875e8f6cf3989807cfe0313b43309874c4e90/" in u and
                                    f"/Ornith-1.5-397B-{q}/" in u for u in h.downloads))
                self.assertEqual(h.verified, [(f, *fam["sha256"][f]) for f in files])
                self.assertAlmostEqual(sum(fam["sha256"][f][0] for f in files) / 1e9, gb, delta=.005)
                for f in files:
                    self.assertRegex(fam["sha256"][f][1], r"^[0-9a-f]{64}$")
                self.assertEqual(Path(cfg["args"][1]).name, files[0])
                self.assertEqual(cfg["model_name"], f"ornith-1.5-397b-{q.lower()}")
                self.assertEqual([Path(r[1]).name for r in h.runs], ["strata_tokenizer.py"])
                self.assertEqual(Path(h.runs[0][h.runs[0].index("--gguf") + 1]).name, files[0])
                self.assertNotIn("--draft-model", cfg["args"])
                self.assertNotIn("--tail-skip", cfg["args"])
                self.assertNotIn("--prompt-batch", cfg["args"])
                self.assertIn("conservative estimates", out)
                self.assertNotIn("measured on a 32 GB card", out)
                path = next(h.t.glob("strata-ornith397-*.json"))
                choices = setup.choices_from_config(path)
                self.assertEqual((choices["family"], choices["model"]), ("ornith397", q))

    def test_vision_is_optional_and_uses_its_own_projector(self):
        for option, gpu in (("no", None), ("cpu", False), ("gpu", True)):
            with self.subTest(option=option):
                h, _, cfg = self.install("IQ1_S", "--vision", option)
                if gpu is None:
                    self.assertNotIn("vision", cfg)
                    self.assertEqual(len(h.downloads), 3)
                else:
                    self.assertEqual(cfg["vision"]["gpu"], gpu)
                    self.assertEqual(Path(cfg["vision"]["mmproj"]).name, "mmproj-Ornith-1.5-397B-f16.gguf")
                    self.assertEqual(len(h.downloads), 4)
                    self.assertEqual(h.verified[-1][1], 918165856)

    def test_size_fallback_and_existing_family(self):
        sizes = setup.family_models("ornith397")
        for q in ("IQ2_XXS", "IQ1_M", "IQ1_S"):
            ram = setup.poly_ram_need(sizes[q], 24)
            self.assertEqual(setup.poly_default(sizes, ram, 24), q)
        self.assertEqual(setup.poly_default(sizes, 1, 24), "IQ1_S")
        self.assertEqual(setup.poly_default(sizes, 1024, 24), "IQ2_XXS")
        self.assertEqual(setup.FAMILIES["ornith"]["tag"], "ornith-")
        self.assertEqual(list(setup.family_models("ornith")), ["APEX-I-COMPACT", "APEX-I-MINI", "APEX-I-QUALITY"])

    def test_local_shards_are_used_without_downloading(self):
        fam = setup.FAMILIES["ornith397"]
        for q, count in (("IQ1_S", 3), ("IQ2_M", 4)):
            with self.subTest(q=q), tempfile.TemporaryDirectory() as folder:
                files = [Path(folder) / setup.model_file(fam, q, i) for i in range(1, count + 1)]
                for f in files:
                    f.touch()
                self.assertEqual(setup.gguf_dir_shards(Path(folder), fam, q), files)
                h, _, cfg = self.install(q, "--gguf-dir", folder)
                self.assertEqual(h.downloads, [])
                self.assertEqual(Path(cfg["args"][1]), files[0])
                self.assertEqual([v[0] for v in h.verified], [f.name for f in files])
                files[-1].unlink()
                # Discovery retains the expected missing path for check_shards to reject.
                self.assertEqual(setup.gguf_dir_shards(Path(folder), fam, q), files)


class WindowsHipBuild(unittest.TestCase):
    def test_existing_config_gets_hip_runtime_and_ordinal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            eng = root / setup.ENGINE_POLY_DIR
            eng.mkdir()
            for name in (setup.POLY_EXE, "strata-device.exe"):
                (eng / name).touch()
            (eng / "BUILD.json").write_text(json.dumps({"backend": "hip", "src": "same", "archs": ["gfx1201"],
                                                       "windows_hip_build": 1, "lib_dirs": ["C:/HIP/bin"]}))
            cfg = {"exe": str(eng / setup.POLY_EXE), "gpu": 1, "args": []}
            path = root / "strata-ornith397-iq1_s.json"
            path.write_text(json.dumps(cfg))
            card = {"index": 1, "name": "RX 9070 XT", "arch": "gfx1201", "vram_gb": 15.9, "count": 2}
            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "WIN", True), \
                    mock.patch.object(setup, "source_hash", return_value="same"), \
                    mock.patch.object(setup, "amd_gpus_win", return_value=[card]), \
                    mock.patch.object(setup, "hip_devices", return_value=[{**card, "index": 0}]), \
                    mock.patch.object(setup, "build_engine_poly_hip", side_effect=AssertionError("unneeded build")):
                setup.update_poly_engine(cfg, cfg_path=path)
            saved = json.loads(path.read_text())
            self.assertEqual((saved["backend"], saved["hip_ordinal"]), ("hip", 0))
            self.assertEqual(saved["lib_dirs"], ["C:/HIP/bin"])

    def test_sdk_discovery_and_build_reuse(self):
        with tempfile.TemporaryDirectory(prefix="poly hip ") as td:
            root = Path(td)
            (root / "CMakeLists.txt").write_text("project(strata VERSION 0.1.0)")
            sdk = root / "HIP SDK"
            for name in ("bin/clang++.exe", "bin/clang.exe", "bin/amdhip64_7.dll",
                         "lib/cmake/hip/hip-config.cmake", "lib/cmake/hipblas/hipblas-config.cmake"):
                p = sdk / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"sdk")
            builds = []

            def compile(src, bdir, targets, defs, vcvars, bat):
                builds.append((targets, defs, vcvars))
                bdir.mkdir(exist_ok=True)
                for name in (setup.POLY_EXE, "strata-device.exe"):
                    (bdir / name).write_bytes(b"engine")

            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "WIN", True), \
                    mock.patch.dict(os.environ, {"HIP_PATH": str(sdk)}), \
                    mock.patch.object(setup, "source_hash", return_value="source1") as fingerprint, \
                    mock.patch.object(setup, "find_vcvars", return_value=root / "VS/vcvars64.bat"), \
                    mock.patch.object(setup, "cmake_build", side_effect=compile):
                eng = setup.build_engine_poly_hip({"arch": "gfx1201"}, root / "llama")
                meta = json.loads((eng / "BUILD.json").read_text())
                self.assertEqual((eng / "amdhip64_7.dll").read_bytes(), b"sdk")
                self.assertTrue((eng / "strata-device.exe").is_file())
                self.assertEqual(meta["lib_dirs"], [str(sdk / "bin")])
                self.assertEqual(meta["archs"], ["gfx1201"])
                self.assertIn("-DCMAKE_CXX_COMPILER=" + (sdk / "bin/clang++.exe").as_posix(), builds[0][1])
                self.assertIn("-DCMAKE_HIP_COMPILER_ROCM_ROOT=" + sdk.as_posix(), builds[0][1])
                self.assertEqual(builds[0][0], ["strata-poly", "strata-device"])
                setup.build_engine_poly_hip({"arch": "gfx1201"}, root / "llama")
                self.assertEqual(len(builds), 1)
                fingerprint.return_value = "source2"
                setup.build_engine_poly_hip({"arch": "gfx1201"}, root / "llama")
                self.assertEqual(json.loads((eng / "BUILD.json").read_text())["src"], "source2")
                self.assertEqual(len(builds), 2)

    def test_missing_explicit_sdk_gives_windows_instructions(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"HIP_PATH": td}):
            code, out = harness.quiet(setup.rocm_windows)
        self.assertEqual(code, 1)
        self.assertIn("Windows HIP SDK", out)
        self.assertNotIn("Ubuntu", out)


if __name__ == "__main__":
    unittest.main()
