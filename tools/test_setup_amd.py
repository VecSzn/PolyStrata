"""Tests for setup.py's AMD card detection on a mocked KFD topology (/sys/class/kfd + /sys/class/drm): the arch
names from gfx_target_version, the CPU node skipped, HIP numbering, product names, which cards are supported and
which TheRock index each family installs from.  No GPU, no ROCm, no downloads.

    python -m unittest tools.test_setup_amd
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def fake_sysfs(root: Path, nodes: list) -> None:
    """nodes: (gfx_target_version, simd_count, render_minor, product_name or None, vram_bytes)"""
    for i, (ver, simd, minor, name, vram) in enumerate(nodes):
        n = root / "class/kfd/kfd/topology/nodes" / str(i)
        n.mkdir(parents=True)
        (n / "properties").write_text(f"cpu_cores_count {0 if simd else 12}\nsimd_count {simd}\n"
                                      f"gfx_target_version {ver}\ndrm_render_minor {minor}\n")
        if simd:
            d = root / f"class/drm/renderD{minor}/device"
            d.mkdir(parents=True)
            (d / "mem_info_vram_total").write_text(str(vram))
            if name is not None:
                (d / "product_name").write_text(name + "\n")


class KfdDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.win = setup.WIN
        setup.WIN = False

    def tearDown(self):
        setup.WIN = self.win
        self.tmp.cleanup()

    def test_every_family(self):
        fake_sysfs(self.root, [
            (0, 0, 0, None, 0),                                   # the CPU node: skipped
            (110001, 120, 128, "", 16 << 30),                     # gfx1101 without a product name
            (120000, 64, 129, None, 16 << 30),                    # gfx1200, no product_name file
            (120001, 128, 130, "AMD Radeon AI PRO R9700", 32 << 30),
            (110002, 64, 131, None, 8 << 30),                     # gfx1102: supported, unvalidated (#938)
            (100306, 4, 132, None, 512 << 20),                    # an integrated gfx1036: listed, not supported
            (110000, 192, 133, "Radeon RX 7900 XTX", 24 << 30),
        ])
        g = setup.amd_gpus(str(self.root))
        self.assertEqual([x["arch"] for x in g], ["gfx1101", "gfx1200", "gfx1201", "gfx1102", "gfx1036", "gfx1100"])
        self.assertEqual([x["index"] for x in g], [0, 1, 2, 3, 4, 5])          # HIP numbers: GPU nodes only
        self.assertEqual(g[0]["name"], setup.AMD_NAMES["gfx1101"])
        self.assertEqual(g[1]["name"], setup.AMD_NAMES["gfx1200"])
        self.assertEqual(g[2]["name"], "AMD Radeon AI PRO R9700")
        self.assertEqual(g[3]["name"], setup.AMD_NAMES["gfx1102"])
        self.assertAlmostEqual(g[2]["vram_gb"], 32.0)
        ok = [x["arch"] for x in g if setup.amd_problem(x) is None]
        self.assertEqual(ok, ["gfx1101", "gfx1200", "gfx1201", "gfx1102", "gfx1100"])
        self.assertIn("gfx1036", setup.amd_problem(g[4]))

    def test_no_kfd(self):
        self.assertEqual(setup.amd_gpus(str(self.root)), [])

    def test_rocm_index_per_family(self):
        for arch in setup.AMD_ARCHS:
            self.assertIn(arch, setup.ROCM_INDEXES)
        self.assertTrue(setup.ROCM_INDEXES["gfx1101"].endswith("/gfx110X-dgpu/"))
        self.assertTrue(setup.ROCM_INDEXES["gfx1200"].endswith("/gfx120X-all/"))
        self.assertEqual(setup.ROCM_INDEXES["gfx1101"], setup.ROCM_INDEXES["gfx1100"])
        self.assertEqual(setup.ROCM_INDEXES["gfx1200"], setup.ROCM_INDEXES["gfx1201"])
        # #524: the RX 6700 XT (gfx1031) takes the RDNA2 wheels, as the RX 6800 / 6900 (gfx1030)
        self.assertEqual(setup.ROCM_INDEXES["gfx1031"], setup.ROCM_INDEXES["gfx1030"])
        self.assertIsNone(setup.amd_problem({"arch": "gfx1031"}))
        # the RX 6600 / 6600 XT / 6650 XT (gfx1032) too: one owner built for it and ran Qwen3.6 on these wheels
        self.assertEqual(setup.ROCM_INDEXES["gfx1032"], setup.ROCM_INDEXES["gfx1030"])
        self.assertIsNone(setup.amd_problem({"arch": "gfx1032"}))


class GpuLists(unittest.TestCase):
    """--gpus with AMD cards: every chosen card must be supported; the first is the main one."""
    AMD = [{"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "arch": "gfx1201"},
           {"index": 1, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201"},
           {"index": 2, "name": "AMD Radeon (gfx1036)", "vram_gb": 0.5, "arch": "gfx1036"},
           {"index": 3, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 24.0, "arch": "gfx1100"}]

    def setUp(self):
        self.say = setup.say
        setup.say = lambda *a, **k: None

    def tearDown(self):
        setup.say = self.say

    def test_list_in_order(self):
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("1,0", self.AMD)], [1, 0])
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus(" 0, 3 ", self.AMD)], [0, 3])

    def test_all_is_every_supported_card_most_vram_first(self):
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("all", self.AMD)], [1, 3, 0])

    def test_refusals(self):
        for text in ("1,2", "1,7", "1", "1,1", "x,y"):
            with self.assertRaises(SystemExit, msg=text):
                setup.amd_parse_gpus(text, self.AMD)

    def test_wheels_hold_one_family(self):
        with tempfile.TemporaryDirectory() as d:        # no system ROCm there: the wheels would be needed
            old = setup.os.environ.get("ROCM_PATH")
            setup.os.environ["ROCM_PATH"] = d
            try:
                with self.assertRaises(SystemExit):
                    setup.rocm_root(["gfx1100", "gfx1201"])
            finally:
                if old is None:
                    del setup.os.environ["ROCM_PATH"]
                else:
                    setup.os.environ["ROCM_PATH"] = old

    def test_runtime_only_system_rocm_falls_back_to_the_wheels(self):
        """#446: a system ROCm 7 with hipcc and libhipblas but no HIP development files (no hip-lang CMake package, no
        hip_runtime.h) is not used for the build: a warning says what is missing and the wheels path follows (here
        it stops at the two-family check, which only the wheels path makes); with the files it is used as before."""
        said = []
        setup.say = lambda msg="": said.append(msg)          # tearDown puts the real one back
        with tempfile.TemporaryDirectory() as d:
            sysroot = Path(d)
            for rel, text in (("bin/hipcc", ""), ("lib/libhipblas.so.3", ""),
                              ("include/rocm-core/rocm_version.h",
                               "#define ROCM_VERSION_MAJOR 7\n#define ROCM_VERSION_MINOR 14\n")):
                (sysroot / rel).parent.mkdir(parents=True, exist_ok=True)
                (sysroot / rel).write_text(text)
            with mock.patch.dict(setup.os.environ, {"ROCM_PATH": d}):
                with self.assertRaises(SystemExit):
                    setup.rocm_root(["gfx1100", "gfx1201"])
                self.assertIn(f"the ROCm in {sysroot} has no HIP development files (lib/cmake/hip-lang/hip-lang-"
                              "config.cmake, include/hip/hip_runtime.h): using AMD's wheels", "\n".join(said))
                self.assertIn("two GPU families", "\n".join(said))
                for lib in ("lib64", "lib"):                 # either place CMake looks
                    with self.subTest(lib=lib):
                        cfg = sysroot / lib / "cmake/hip-lang/hip-lang-config.cmake"
                        cfg.parent.mkdir(parents=True, exist_ok=True)
                        cfg.write_text("")
                        said.clear()
                        with self.assertRaises(SystemExit):
                            setup.rocm_root(["gfx1100", "gfx1201"])
                        self.assertIn("(include/hip/hip_runtime.h)", "\n".join(said))
                        (sysroot / "include/hip").mkdir(parents=True, exist_ok=True)
                        (sysroot / "include/hip/hip_runtime.h").write_text("")
                        self.assertEqual(setup.rocm_root(["gfx1100", "gfx1201"]), (sysroot, [str(sysroot / "lib")]))
                        cfg.unlink()
                        (sysroot / "include/hip/hip_runtime.h").unlink()

    def test_build_for_every_arch(self):
        """build_engine_hip compiles for the set of the chosen cards' archs and records it in BUILD.json."""
        calls = {}
        saved = {k: getattr(setup, k) for k in ("ROOT", "rocm_root", "cmake_build", "source_hash", "source_version",
                                                "ok", "shutil")}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            def fake_build(src, bdir, target, defs, vcvars, bat):
                calls["defs"] = defs
                (bdir).mkdir(parents=True, exist_ok=True)
                (bdir / setup.EXE).write_text("engine")

            class Sh:
                which = staticmethod(lambda name: "/usr/bin/" + name)
                copy2 = staticmethod(lambda a, b: Path(b).write_text(Path(a).read_text()))
            setup.ROOT, setup.cmake_build, setup.ok, setup.shutil = root, fake_build, lambda *a: None, Sh
            setup.source_hash, setup.source_version = (lambda *a: "h"), (lambda: "0.1.31")
            setup.rocm_root = lambda archs: (calls.setdefault("archs", archs) and root, [str(root / "lib")])
            try:
                setup.build_engine_hip({"arch": "gfx1201", "archs": ["gfx1201", "gfx1100", "gfx1201"]}, root)
                self.assertEqual(calls["archs"], ["gfx1100", "gfx1201"])
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201", calls["defs"])
                import json
                self.assertEqual(json.loads((root / "engine" / "BUILD.json").read_text())["archs"],
                                 ["gfx1100", "gfx1201"])
                calls.clear()                            # one of those cards alone: already built, no compile
                setup.build_engine_hip({"arch": "gfx1100"}, root)
                self.assertNotIn("defs", calls)
                setup.build_engine_hip({"arch": "gfx1101"}, root)   # another arch: compiled again
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1101", calls["defs"])
            finally:
                for k, v in saved.items():
                    setattr(setup, k, v)


class WindowsDetection(unittest.TestCase):
    """Windows: the AMD cards from the display adapters (mocked Win32_VideoController rows and display-class registry
    values), the HIP runtime's numbering from `strata-device --list-devices`, and the card setup chose matched to its
    HIP ordinal (#325: an integrated Radeon is HIP's device 0)."""
    ADAPTERS = [  # Win32_VideoController: name, PNPDeviceID, AdapterRAM (32-bit: at most 4 GB)
        {"name": "NVIDIA GeForce RTX 5070", "pnp": r"PCI\VEN_10DE&DEV_2F04&SUBSYS_1234&REV_A1\4&1", "ram": 4293918720},
        {"name": "AMD Radeon RX 9070 XT", "pnp": r"PCI\VEN_1002&DEV_7550&SUBSYS_0E3B1002&REV_C0\6&2", "ram": 4293918720},
        {"name": "AMD Radeon(TM) Graphics", "pnp": r"PCI\VEN_1002&DEV_164E&SUBSYS_88771043&REV_C1\4&3", "ram": 536870912},
        {"name": "AMD Radeon RX 7800 XT", "pnp": r"PCI\VEN_1002&DEV_7499&SUBSYS_0000&REV_C8\6&4", "ram": 4293918720},
    ]
    REGISTRY = [  # the display class's driver instances: 64-bit VRAM size (as int, or as bytes)
        {"DriverDesc": "NVIDIA GeForce RTX 5070", "MatchingDeviceId": r"PCI\VEN_10DE&DEV_2F04",
         "HardwareInformation.qwMemorySize": 12 << 30},
        {"DriverDesc": "AMD Radeon RX 6600", "MatchingDeviceId": r"PCI\VEN_1002&DEV_73FF",     # a removed card
         "HardwareInformation.qwMemorySize": 8 << 30},
        {"DriverDesc": "AMD Radeon RX 9070 XT", "MatchingDeviceId": r"PCI\VEN_1002&DEV_7550&REV_C0",
         "HardwareInformation.qwMemorySize": (16 << 30).to_bytes(8, "little"), "DriverVersion": "32.0.21013.1000"},
        {"DriverDesc": "AMD Radeon(TM) Graphics", "MatchingDeviceId": r"PCI\VEN_1002&DEV_164E&REV_C1",
         "HardwareInformation.qwMemorySize": 512 << 20},
        {"DriverDesc": "AMD Radeon RX 7800 XT", "MatchingDeviceId": r"PCI\VEN_1002&DEV_7499",
         "HardwareInformation.qwMemorySize": 16 << 30},
    ]
    LIST = ("device 0: AMD Radeon(TM) Graphics\n  arch gfx1036, 28.1 GiB, wave32\n"
            "  cannot run: GPU 0 (AMD Radeon(TM) Graphics, gfx1036) is not an architecture this PolyStrata engine was "
            "compiled for (gfx1100,gfx1101,gfx1102,gfx1200,gfx1201,gfx1030); ...\n"
            "device 1: AMD Radeon RX 9070 XT\n  arch gfx1201, 15.9 GiB, wave32\n"
            "device 2: AMD Radeon RX 7800 XT\n  arch gfx1101, 16.0 GiB, wave32\n")

    def test_adapters(self):
        g = setup.amd_gpus_windows(self.ADAPTERS, self.REGISTRY)
        self.assertEqual([x["name"] for x in g], ["AMD Radeon RX 9070 XT", "AMD Radeon(TM) Graphics",
                                                  "AMD Radeon RX 7800 XT"])     # AMD only, present only, in order
        self.assertEqual([x["index"] for x in g], [0, 1, 2])
        self.assertEqual(g[0]["arch"], "gfx1201")                               # by PCI device id
        self.assertEqual(g[2]["arch"], "gfx1101")                               # an unlisted id: by its name
        self.assertTrue(g[1]["arch"].startswith("unknown"))                     # the iGPU: listed, not supported
        self.assertAlmostEqual(g[0]["vram_gb"], 16.0)                           # the registry's 64-bit size
        self.assertEqual(g[0]["driver"], "32.0.21013.1000")
        self.assertEqual([setup.amd_problem(x) is None for x in g], [True, False, True])

    def test_rx_6800m_is_gfx1031(self):
        """#881: PCI 73DF (RX 6700 XT / 6750 XT / 6800M) is gfx1031, not an unknown id."""
        self.assertEqual(setup.win_amd_arch(0x73DF, "AMD Radeon RX 6800M"), "gfx1031")
        self.assertIsNone(setup.amd_problem({"arch": "gfx1031"}))

    def test_cpu_vision_build_uses_the_visual_studio_environment(self):
        """#881: build_vision_cpu on Windows hands cmake_build the vcvars file and a real .bat name (it passed None and
        an empty name before)."""
        calls = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "WIN", True),                 mock.patch.object(setup, "find_vcvars", return_value=Path("C:/vs/vcvars64.bat")),                 mock.patch.object(setup, "cmake_build", lambda *a: calls.append(a)),                 mock.patch.object(setup, "ROOT", Path(d)), mock.patch.object(setup.shutil, "copy2"):
            eng = Path(d) / "engine"
            eng.mkdir()
            setup.build_vision_cpu(eng, eng / "BUILD.json", {}, Path("llama"), "src1")
        self.assertEqual(calls[0][4], Path("C:/vs/vcvars64.bat"))
        self.assertTrue(calls[0][5].endswith(".bat"))

    def test_registry_alone(self):
        """No WMI answer: the registry's own list (which can hold a removed card)."""
        g = setup.amd_gpus_windows([], self.REGISTRY)
        self.assertEqual([x["arch"] for x in g if not x["arch"].startswith("unknown")], ["gfx1201", "gfx1101"])

    def test_arch_names(self):
        for name, arch in (("AMD Radeon RX 9070 GRE", "gfx1201"), ("AMD Radeon AI PRO R9700", "gfx1201"),
                           ("AMD Radeon RX 9060 XT", "gfx1200"), ("AMD Radeon RX 7900 GRE", "gfx1100"),
                           ("AMD Radeon PRO W7800", "gfx1100"), ("AMD Radeon RX 7700 XT", "gfx1101"),
                           ("AMD Radeon RX 7600", "gfx1102"), ("AMD Radeon RX 6950 XT", "gfx1030"),
                           ("AMD Radeon RX 6800M", ""), ("AMD Radeon 780M Graphics", "gfx1103"), ("AMD Radeon RX 7700S", "")):
            self.assertEqual(setup.win_amd_arch(None, name), arch, name)
        self.assertEqual(setup.win_amd_arch(0x744C, "whatever"), "gfx1100")

    def test_list_devices(self):
        h = setup.hip_devices(text=self.LIST)
        self.assertEqual([(x["index"], x["arch"]) for x in h], [(0, "gfx1036"), (1, "gfx1201"), (2, "gfx1101")])
        self.assertAlmostEqual(h[1]["vram_gb"], 15.9)
        self.assertIn("not an architecture", h[0]["cannot_run"])
        self.assertIsNotNone(setup.amd_problem(h[0]))
        self.assertIsNone(setup.amd_problem(h[1]))
        cannot = setup.hip_devices(text="device 0: AMD Radeon RX 9070 XT\n  arch gfx1201, 15.9 GiB, wave32\n"
                                        "  cannot run: GPU 0 runs wave64\n")
        self.assertEqual(setup.amd_problem(cannot[0]), "GPU 0 runs wave64")   # the engine's own check counts
        self.assertEqual(setup.hip_devices(text="(no GPU device)\n"), [])

    def test_the_card_by_its_hip_ordinal(self):
        """#325: setup's GPU 0 (the 9070 XT, display order) is HIP's device 1 behind the iGPU."""
        listed = setup.amd_gpus_windows(self.ADAPTERS, self.REGISTRY)
        hip = setup.hip_devices(text=self.LIST)
        self.assertEqual(setup.hip_match(listed[0], listed, hip)["index"], 1)
        self.assertEqual(setup.hip_match(listed[2], listed, hip)["index"], 2)
        two = [{"index": 0, "arch": "gfx1201"}, {"index": 1, "arch": "gfx1201"}]   # two of a kind: by rank
        self.assertEqual(setup.hip_match(two[1], two, [{"index": 0, "arch": "gfx1036"}, {"index": 1, "arch": "gfx1201"},
                                                       {"index": 2, "arch": "gfx1201"}])["index"], 2)
        self.assertIsNone(setup.hip_match({"index": 0, "arch": "gfx1100"}, [{"index": 0, "arch": "gfx1100"}], hip))

    def test_hip_card(self):
        listed = setup.amd_gpus_windows(self.ADAPTERS, self.REGISTRY)
        with mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: setup_hip(self.LIST)), \
                mock.patch.object(setup, "ok", lambda *a: None):
            g = setup.hip_card(Path("engine"), listed[0], listed)
        self.assertEqual((g["index"], g["count"], g["arch"], g["vram_gb"]), (1, 3, "gfx1201", 15.9))
        with mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: []), \
                mock.patch.object(setup, "say", lambda *a, **k: None), self.assertRaises(SystemExit):
            setup.hip_card(Path("engine"), listed[0], listed)          # no AMD driver: stops before the download
        with mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: setup_hip(self.LIST)[:1]), \
                mock.patch.object(setup, "say", lambda *a, **k: None), self.assertRaises(SystemExit):
            setup.hip_card(Path("engine"), listed[0], listed)          # HIP does not list the card

    def test_windows_dispatch(self):
        """amd_gpus() on Windows: HIP's numbering once the HIP engine answers, the display adapters before."""
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: None), \
                mock.patch.object(setup, "amd_gpus_windows", lambda adapters=None, registry=None: ["display"]):
            self.assertEqual(setup.amd_gpus(), ["display"])
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "hip_devices", lambda probe=None, text=None: ["hip"]), \
                mock.patch.object(setup, "amd_gpus_windows", lambda adapters=None, registry=None: ["display"]):
            self.assertEqual(setup.amd_gpus(), ["hip"])

    def test_no_hip_engine_no_probe(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            (eng / "strata-device.exe").write_text("x")
            (eng / "BUILD.json").write_text('{"backend": "cuda"}')     # an NVIDIA engine is never asked
            self.assertIsNone(setup.hip_devices(eng / "strata-device.exe"))

    def test_prebuilt_hip_zip(self):
        """get_prebuilt_hip: a published zip (here a local folder) is unpacked into engine/; one without code for the
        card, or older than the first Windows HIP release, is refused."""
        import json
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pub = root / "pub"
            pub.mkdir()

            def publish(meta):
                with zipfile.ZipFile(pub / setup.WIN_HIP_ASSET, "w") as z:
                    z.writestr(setup.EXE, "engine")       # #975: "strata" on Linux
                    z.writestr("strata-device.exe", "probe")
                    z.writestr("rocm/bin/amdhip64_7.dll", "dll")
                    z.writestr("BUILD.json", json.dumps(meta))
            ver = ".".join(map(str, setup.WIN_HIP_MIN_ENGINE))
            good = {"source": "prebuilt", "backend": "hip", "version": ver, "archs": ["gfx1100", "gfx1201"],
                    "lib_dirs": ["rocm/bin"]}
            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "say", lambda *a, **k: None), \
                    mock.patch.object(setup, "ok", lambda *a: None), mock.patch.object(setup, "warn", lambda *a: None):
                publish({**good, "archs": ["gfx1100"]})
                self.assertIsNone(setup.get_prebuilt_hip(str(pub) + "/", {"arch": "gfx1201"}))
                publish({**good, "version": "0.1.30"})
                self.assertIsNone(setup.get_prebuilt_hip(str(pub) + "/", {"arch": "gfx1201"}))
                publish(good)
                eng = setup.get_prebuilt_hip(str(pub) + "/", {"arch": "gfx1201"})
                self.assertEqual(eng, root / "engine")
                self.assertTrue((eng / "rocm" / "bin" / "amdhip64_7.dll").exists())
                self.assertEqual(setup.hip_lib_dirs(eng), [eng / "rocm" / "bin"])
                (pub / setup.WIN_HIP_ASSET).unlink()           # installed: kept, nothing downloaded again
                self.assertEqual(setup.get_prebuilt_hip(str(pub) + "/", {"arch": "gfx1201"}), eng)


_HIP_DEVICES = setup.hip_devices                      # the real parser, for the tests that mock setup.hip_devices


def setup_hip(text):
    return _HIP_DEVICES(text=text)


class CalibrationKey(unittest.TestCase):
    """#566: a calibration is saved and found again per PC and model (hardware_key).  A HIP config's cards are AMD's,
    in HIP's numbering - nvidia-smi named them "?" (or another card with that number) before - and setup reuses a
    saved calibration on Linux HIP.  The NVIDIA key and the settings file's format are unchanged."""
    AMD = [{"index": 0, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 23.98, "arch": "gfx1100"},
           {"index": 1, "name": "AMD Radeon RX 7900 XT", "vram_gb": 19.98, "arch": "gfx1100"},
           {"index": 2, "name": "AMD Radeon (gfx1201)", "vram_gb": 15.92, "arch": "gfx1201"},
           {"index": 3, "name": "AMD Radeon (gfx1036)", "vram_gb": 0.5, "arch": "gfx1036"}]

    def setUp(self):
        self.patches = [mock.patch.object(setup, "amd_gpus", lambda *a, **k: [dict(g) for g in self.AMD]),
                        mock.patch.object(setup, "cpu_info", lambda: ("AMD Ryzen 9 7950X", 16)),
                        mock.patch.object(setup, "ram_gb", lambda: 63.6)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    @staticmethod
    def cfg(gpu, backend="hip", ctx="32768"):
        c = {"model_name": "qwen3.8-flash-next-iq3_xxs", "args": ["--max-context", ctx, "--kv", "int8"], "gpu": gpu}
        if backend:
            c["backend"] = backend
        return c

    def test_hip_key_names_the_amd_card_and_its_arch(self):
        def no_nvidia(*a, **k):                     # a HIP config never asks nvidia-smi (a PC with both kinds)
            raise AssertionError("gpu_info called for a HIP config")
        with mock.patch.object(setup, "gpu_info", no_nvidia):
            self.assertEqual(setup.hardware_key(self.cfg(0)),
                             "AMD Radeon RX 7900 XTX (gfx1100)|24GB|AMD Ryzen 9 7950X|64GB|qwen3.8-flash-next-iq3_xxs|"
                             "32768|text")
            # the arch is not repeated when the name already holds it (no product name in sysfs)
            self.assertTrue(setup.hardware_key(self.cfg(2)).startswith("AMD Radeon (gfx1201)|16GB|"))

    def test_two_cards_of_one_arch_are_two_keys(self):
        keys = {setup.hardware_key(self.cfg(i)) for i in (0, 1, 2)}
        self.assertEqual(len(keys), 3)

    def test_a_split_names_every_card(self):
        self.assertTrue(setup.hardware_key(self.cfg([1, 0])).startswith(
            "AMD Radeon RX 7900 XT (gfx1100) + AMD Radeon RX 7900 XTX (gfx1100)|44GB|"))

    def test_no_gpu_in_the_config_is_the_supported_card_with_the_most_vram(self):
        self.assertTrue(setup.hardware_key(self.cfg(None)).startswith("AMD Radeon RX 7900 XTX (gfx1100)|24GB|"))

    def test_a_card_that_is_gone_is_unknown(self):
        self.assertTrue(setup.hardware_key(self.cfg(7)).startswith("?|0GB|"))

    def test_nvidia_key_unchanged(self):
        def amd(*a, **k):
            raise AssertionError("amd_gpus called for an NVIDIA config")
        with mock.patch.object(setup, "amd_gpus", amd), \
                mock.patch.object(setup, "gpu_info", lambda i=None: {"name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.94}):
            self.assertEqual(setup.hardware_key(self.cfg(0, backend=None)),
                             "NVIDIA GeForce RTX 5070|12GB|AMD Ryzen 9 7950X|64GB|qwen3.8-flash-next-iq3_xxs|32768|text")
            self.assertEqual(setup.hardware_key(self.cfg([0, 1], backend="cuda")),
                             "NVIDIA GeForce RTX 5070 + NVIDIA GeForce RTX 5070|24GB|AMD Ryzen 9 7950X|64GB|"
                             "qwen3.8-flash-next-iq3_xxs|32768|text")

    def test_setup_reuses_a_saved_calibration_on_linux_hip(self):
        cfg = self.cfg(1)
        saved = {"settings": {"pcie_frac": 0.0}, "tok_s": 40.1, "date": "2026-10-03"}
        other = {"settings": {"pcie_frac": 0.5}, "tok_s": 50.0, "date": "2026-10-02"}
        store = {"calibration": {setup.hardware_key(cfg): saved, setup.hardware_key(self.cfg(0)): other}}
        with mock.patch.object(setup, "load_settings", lambda: store):
            with mock.patch.object(setup, "WIN", False):
                self.assertEqual(setup.setup_calibration(cfg, hip=True), saved)       # its own card's, not the XTX's
                self.assertIsNone(setup.setup_calibration(self.cfg(2), hip=True))     # never tuned on that card
            with mock.patch.object(setup, "WIN", True):
                self.assertIsNone(setup.setup_calibration(cfg, hip=True))             # Windows HIP: defaults for now


class WindowsHipVision(unittest.TestCase):
    def test_no_cpu_encoder_on_windows(self):
        with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "warn", lambda *a: None):
            self.assertEqual(setup.hip_vision("cpu"), "none")
            self.assertEqual(setup.hip_vision("yes"), "none")
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "warn", lambda *a: None):
            self.assertEqual(setup.hip_vision("cpu"), "cpu")


class HipRuntimeBesideExe(unittest.TestCase):
    """#468 #461: the bundled HIP runtime (and amd_comgr) goes next to strata.exe, so an AMD driver's System32 copy is
    not found first; rocBLAS and the rest stay in rocm/bin."""

    def test_copies_only_the_runtime(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            rb = eng / "rocm" / "bin"
            rb.mkdir(parents=True)
            (eng / "BUILD.json").write_text(json.dumps({"backend": "hip", "lib_dirs": ["rocm/bin"]}))
            for n, data in (("amdhip64_7.dll", b"hip-3686"), ("amd_comgr.dll", b"comgr"), ("rocblas.dll", b"blas")):
                (rb / n).write_bytes(data)
            setup.hip_runtime_beside_exe(eng)
            self.assertEqual((eng / "amdhip64_7.dll").read_bytes(), b"hip-3686")
            self.assertEqual((eng / "amd_comgr.dll").read_bytes(), b"comgr")
            self.assertFalse((eng / "rocblas.dll").exists())          # finds its kernels relative to rocm/bin
            (rb / "amdhip64_7.dll").write_bytes(b"hip-3690-newer")    # a newer zip: replaced on the next start
            setup.hip_runtime_beside_exe(eng)
            self.assertEqual((eng / "amdhip64_7.dll").read_bytes(), b"hip-3690-newer")

    def test_no_rocm_bin_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            setup.hip_runtime_beside_exe(Path(d))                   # no BUILD.json (a CUDA or Linux engine)
            self.assertEqual(list(Path(d).iterdir()), [])


class DeviceAccess(unittest.TestCase):
    """Linux AMD: /dev/kfd and the render nodes must be openable by the user; setup warns, never refuses."""

    def dev(self, d, kfd=True, nodes=("renderD128",)):
        root = Path(d)
        (root / "dri").mkdir()
        if kfd:
            (root / "kfd").write_text("")
        for n in nodes:
            (root / "dri" / n).write_text("")
        return str(root)

    def test_accessible_is_silent(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.amd_device_access_problem(self.dev(d), access=lambda p, m: True))

    def test_no_kfd_is_not_this_problem(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.amd_device_access_problem(self.dev(d, kfd=False), access=lambda p, m: False))

    def test_kfd_denied_names_the_fix(self):
        with tempfile.TemporaryDirectory() as d:
            msg = setup.amd_device_access_problem(self.dev(d), access=lambda p, m: not p.endswith("kfd"))
            self.assertIn("kfd", msg)
            self.assertNotIn("renderD128", msg)
            self.assertIn("sudo usermod -aG render,video $USER", msg)
            self.assertIn("log out and in", msg)

    def test_render_node_denied(self):
        with tempfile.TemporaryDirectory() as d:
            msg = setup.amd_device_access_problem(self.dev(d), access=lambda p, m: "renderD" not in p)
            self.assertIn("renderD128", msg)
            self.assertNotIn("/kfd", msg)

    def test_engine_names_the_permission_not_another_program(self):
        src = (Path(setup.__file__).resolve().parent / "src/program/generate.cpp").read_text(encoding="utf-8")
        i = src.index("cannot open /dev/kfd")
        guard = src[src.rindex("#if", 0, i):i]
        self.assertIn("cudaGetDeviceCount", guard)
        self.assertIn('access("/dev/kfd", R_OK | W_OK)', guard)
        self.assertIn("STRATA_USE_HIP", guard)               # CUDA builds keep the "another program" text
        self.assertIn("another program (or an engine that is still exiting)", src)


class EngineLaunchEnv(unittest.TestCase):
    """#2372: every engine program setup runs gets the ready-made HIP engine's rocm/bin on PATH, or Windows shows a
    "hipblas.dll not found" dialog per DLL and `strata --families` prints nothing (setup then said the engine does
    not run Qwen3.6)."""

    def engine(self, tmp: Path) -> Path:
        eng = tmp / "engine"
        (eng / "rocm" / "bin").mkdir(parents=True)
        (eng / "BUILD.json").write_text('{"backend": "hip", "lib_dirs": ["rocm/bin"]}', encoding="utf-8")
        return eng

    def test_engine_env_puts_rocm_bin_first(self):
        with tempfile.TemporaryDirectory() as t:
            eng = self.engine(Path(t))
            with mock.patch.dict("os.environ", {"PATH": "C:\\Windows"}):
                env = setup.engine_env(eng)
            self.assertEqual(env["PATH"].split(setup.os.pathsep)[0], str(eng / "rocm" / "bin"))
            self.assertTrue(env["PATH"].endswith("C:\\Windows"))

    def test_engine_without_lib_dirs_keeps_the_environment(self):
        with tempfile.TemporaryDirectory() as t:
            with mock.patch.dict("os.environ", {"PATH": "/usr/bin"}):
                self.assertEqual(setup.engine_env(Path(t))["PATH"], "/usr/bin")

    def test_families_probe_runs_with_that_env(self):
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"], seen["env"] = cmd, kw.get("env")
            return mock.Mock(stdout="qwen4exp qwen35moe\n")
        with tempfile.TemporaryDirectory() as t:
            eng = self.engine(Path(t))
            with mock.patch.object(setup.subprocess, "run", fake_run):
                names = setup.engine_families(eng)
            self.assertEqual(names, {"qwen4exp", "qwen35moe"})
            self.assertEqual(seen["cmd"][1], "--families")
            self.assertEqual(seen["env"]["PATH"].split(setup.os.pathsep)[0], str(eng / "rocm" / "bin"))

    def test_device_probe_runs_with_that_env(self):
        seen = {}

        def fake_run(cmd, **kw):
            seen["env"] = kw.get("env")
            return mock.Mock(returncode=0, stdout="device 0: AMD Radeon RX 7900 XT\n  arch gfx1100, 20.0 GiB, wave32\n")
        with tempfile.TemporaryDirectory() as t:
            eng = self.engine(Path(t))
            probe = eng / ("strata-device.exe" if setup.WIN else "strata-device")
            probe.write_text("x")
            with mock.patch.object(setup.subprocess, "run", fake_run),                     mock.patch.object(setup, "hip_runtime_beside_exe", lambda d: None):
                found = setup.hip_devices(probe)
            self.assertEqual(found[0]["arch"], "gfx1100")
            self.assertEqual(seen["env"]["PATH"].split(setup.os.pathsep)[0], str(eng / "rocm" / "bin"))


class PolyHipWindows(unittest.TestCase):
    """strata-poly on an AMD card under Windows: setup compiles it itself (AMD's TheRock wheels in .rocm-win), the
    wheels' version coming from STRATA_ROCM_WIN_VERSION (or build_windows.bat's STRATA_ROCM_VERSION).
    No ROCm, no GPU, no downloads."""

    def test_an_installed_rocm_win_is_kept(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "rocm"
            (root / "lib" / "llvm" / "bin").mkdir(parents=True)
            (root / "lib" / "llvm" / "bin" / "clang++.exe").write_text("x")
            (root / ".kpack").mkdir()
            (root / ".kpack" / "blas_lib_gfx1201.kpack").write_text("x")
            (root / "bin").mkdir()
            venv = Path(t) / ".rocm-win" / "Scripts"
            venv.mkdir(parents=True)
            (venv / "python.exe").write_text("x")
            (venv / "rocm-sdk.exe").write_text("x")
            with mock.patch.object(setup, "ROOT", Path(t)), \
                    mock.patch.object(setup, "out", lambda cmd: str(root) + "\n"), \
                    mock.patch.object(setup, "run", mock.Mock(side_effect=AssertionError("installed"))):
                got, dirs = setup.rocm_win_root(["gfx1201"])
            self.assertEqual((got, dirs), (root, [str(root / "bin")]))

    @staticmethod
    def rocm_win_tree(t):
        """A .rocm-win holding every arch's kernels and a working root: the layout setup keeps as it is."""
        root = Path(t) / "rocm"
        (root / "lib" / "llvm" / "bin").mkdir(parents=True)
        (root / "lib" / "llvm" / "bin" / "clang++.exe").write_text("x")
        (root / ".kpack").mkdir()
        (root / ".kpack" / "blas_lib_gfx1201.kpack").write_text("x")
        venv = Path(t) / ".rocm-win"
        (venv / "Scripts").mkdir(parents=True)
        (venv / "Scripts" / "python.exe").write_text("x")
        (venv / "Scripts" / "rocm-sdk.exe").write_text("x")
        return root, venv

    def test_another_wheels_version_is_installed_in_its_place(self):
        with tempfile.TemporaryDirectory() as t:
            root, venv = self.rocm_win_tree(t)
            (venv / "Lib" / "site-packages" / "rocm-9.9.9.dist-info").mkdir(parents=True)
            ran = []
            with mock.patch.object(setup, "ROOT", Path(t)), \
                    mock.patch.object(setup, "out", lambda cmd: str(root) + "\n"), \
                    mock.patch.object(setup, "run", lambda cmd: ran.append(cmd)):
                got, dirs = setup.rocm_win_root(["gfx1201"])
            self.assertEqual((got, dirs), (root, [str(root / "bin")]))
            pip = [c for c in ran if "-m" in c]
            self.assertEqual(len(pip), 1, ran)                       # exactly one install, and it is the wanted version
            self.assertIn(f"rocm[libraries,devel,device-gfx1201]=={setup.ROCM_WIN_VERSION}", pip[0])
            self.assertIn([str(venv / "Scripts" / "rocm-sdk.exe"), "init"], ran)

    def test_a_venv_that_does_not_say_its_version_is_kept(self):
        with tempfile.TemporaryDirectory() as t:
            root, venv = self.rocm_win_tree(t)                       # no rocm-*.dist-info: not pip's doing
            warned = []
            with mock.patch.object(setup, "ROOT", Path(t)), \
                    mock.patch.object(setup, "out", lambda cmd: str(root) + "\n"), \
                    mock.patch.object(setup, "ROCM_WIN_VERSION_OVERRIDE", "9.9.9"), \
                    mock.patch.object(setup, "ROCM_WIN_VERSION", "9.9.9"), \
                    mock.patch.object(setup, "warn", lambda msg: warned.append(msg)), \
                    mock.patch.object(setup, "run", mock.Mock(side_effect=AssertionError("installed"))):
                got, dirs = setup.rocm_win_root(["gfx1201"])
            self.assertEqual((got, dirs), (root, [str(root / "bin")]))
            self.assertTrue(any("does not say which ROCm" in m for m in warned), warned)
            self.assertTrue(any("9.9.9" in m for m in warned), warned)

    def test_the_env_var_picks_the_wheels_version(self):
        base = {k: v for k, v in os.environ.items()
                if k not in ("STRATA_ROCM_WIN_VERSION", "STRATA_ROCM_VERSION")}

        def constant(**env):
            r = subprocess.run([sys.executable, "-c", "import setup; print(setup.ROCM_WIN_VERSION)"],
                               cwd=Path(setup.__file__).resolve().parent, capture_output=True, text=True,
                               env={**base, **env})
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.strip()

        self.assertEqual(constant(STRATA_ROCM_WIN_VERSION="9.9.9"), "9.9.9")
        self.assertEqual(constant(STRATA_ROCM_VERSION="8.8.8"), "8.8.8")   # build_windows.bat's name
        self.assertEqual(constant(STRATA_ROCM_WIN_VERSION="9.9.9", STRATA_ROCM_VERSION="8.8.8"), "9.9.9")
        self.assertEqual(constant(), setup.ROCM_WIN_VERSION_DEFAULT)

    def test_an_engine_of_this_source_is_kept(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            eng = root / setup.ENGINE_POLY_DIR
            eng.mkdir()
            (eng / setup.POLY_EXE).write_text("x")
            (eng / "BUILD.json").write_text(json.dumps({"source": "local-hip", "backend": "hip",
                                                        "archs": ["gfx1201"], "src": "abc"}), encoding="utf-8")
            compiled = mock.Mock(side_effect=AssertionError("compiled"))
            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "source_hash", lambda parts: "abc"), \
                    mock.patch.object(setup, "cmake_build", compiled):
                got = setup.build_engine_poly_hip_win({"arch": "gfx1201", "archs": ["gfx1201"]}, root / "llama.cpp")
            self.assertEqual(got, eng)
            compiled.assert_not_called()

    def test_the_windows_engine_is_dispatched_from_the_amd_build(self):
        seen = {}
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "build_engine_poly_hip_win",
                                  lambda gpu, llama, portable=False: seen.update(gpu=gpu, portable=portable)):
            setup.build_engine_poly_hip({"arch": "gfx1201"}, Path("llama.cpp"), True)
        self.assertEqual(seen, {"gpu": {"arch": "gfx1201"}, "portable": True})

    def test_the_check_line_names_amd_on_windows(self):
        d = setup.FAMILIES["tiel"]["models"]["UD-Q3_K_XL"]
        with mock.patch.object(setup, "WIN", True):
            line = setup.poly_verdict(d, 63, 15.9, True)
        self.assertIn("measured on one AMD card on Windows", line)
        self.assertIn("fits by setup's estimate", line)

    def build(self, root, bundled, portable=False):
        """The Windows HIP build's compile with cmake mocked: returns (the cmake args, the BUILD.json)."""
        built = []

        def cmake(src_dir, bdir, target, defs, vcvars, bat):
            built.append((target, defs))
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / setup.POLY_EXE).write_text("x")

        with mock.patch.object(setup, "ROOT", root), \
                mock.patch.object(setup, "source_hash", lambda parts: "new"), \
                mock.patch.object(setup, "find_vcvars", lambda: "vcvars"), \
                mock.patch.object(setup, "rocm_win_root", lambda archs: (root / "rocm", [])), \
                mock.patch.object(setup, "cmake_build", cmake), \
                mock.patch.object(setup, "hipblaslt_version", lambda dirs: None), \
                mock.patch.object(setup, "source_version", lambda: "0.1.45"), \
                mock.patch.object(setup, "hip_runtime_beside_exe", lambda eng: None):
            got = setup.build_engine_poly_hip_win({"arch": "gfx1201"}, root / "llama.cpp", portable)
        self.assertEqual(got, root / setup.ENGINE_POLY_DIR)
        return built[0][1], json.loads((root / setup.ENGINE_POLY_DIR / "BUILD.json").read_text(encoding="utf-8"))

    def test_the_build_is_native_and_loads_the_bundled_rocm(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            bundled = root / "engine" / "rocm" / "bin"
            bundled.mkdir(parents=True)
            defs, meta = self.build(root, bundled)
            self.assertIn("-DSTRATA_PORTABLE=OFF", defs)           # native for the PC setup builds on (as on Linux)
            self.assertIn("-DSTRATA_PREFILL_MMQ=ON", defs)         # the ready-made Windows HIP engine's own build
            self.assertNotIn("-DSTRATA_NATIVE_EXPERTS=ON", defs)   # ON by default
            self.assertEqual(meta["lib_dirs"], [str(bundled)])     # the ready-made engine's ROCm: not on the disk twice
            self.assertFalse((root / setup.ENGINE_POLY_DIR / "rocm").exists())

    def test_max_compatibility_uses_the_portable_build_and_changes_rebuild(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            defs, meta = self.build(root, root / "engine" / "rocm" / "bin", portable=True)
            self.assertIn("-DSTRATA_PORTABLE=ON", defs)
            self.assertTrue(meta["portable"])
            defs, _ = self.build(root, root / "engine" / "rocm" / "bin")
            self.assertIn("-DSTRATA_PORTABLE=OFF", defs)

    def test_without_a_ready_made_engine_the_wheels_own_rocm_is_used(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            defs, meta = self.build(root, root / "engine" / "rocm" / "bin")
            self.assertEqual(meta["lib_dirs"], [str(root / "rocm" / "bin")])   # .rocm-win's own
            self.assertFalse((root / setup.ENGINE_POLY_DIR / "rocm").exists())


class TdrPointer(unittest.TestCase):
    def test_setup_points_windows_gfx12_to_the_entry(self):
        src = Path(setup.__file__).read_text(encoding="utf-8")
        self.assertIn('WIN and str(gpu.get("arch") or "").startswith("gfx12")', src)
        doc = (Path(setup.__file__).resolve().parent / "docs/TROUBLESHOOTING.md").read_text(encoding="utf-8")
        self.assertIn("Windows AMD: the driver resets", doc)
        for env in ("STRATA_PF_STEP_SYNC", "STRATA_KV_HOST_DMA"):
            self.assertIn(env, doc)
            self.assertIn(env, (Path(setup.__file__).resolve().parent / "src/prefill/prefill.cpp").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
