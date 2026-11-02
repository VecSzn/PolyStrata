"""Tests for resolve_custom_gguf (--custom-gguf PATH) in setup.py.

strata_inspect is imported inside resolve_custom_gguf via `from strata_inspect import ...`,
so it is not a top-level attribute of setup. We patch it through sys.modules, injected
before the function runs so its inline import picks up our stubs.

Run: python tools/test_setup_custom_gguf.py  (no GPU or server needed)
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import setup  # noqa: E402


def _write_gguf(path: Path) -> None:
    """Write minimal bytes so .is_file() is True."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00\x01\x00\x00")


class CustomGgufTests(unittest.TestCase):

    def _make_stub(self, fp_dict: dict, kind: str, table: dict | None = None):
        """Build a stub module for sys.modules['strata_inspect'].

        If table is None, use the real data/gguf_fingerprints.json. Otherwise use the given dict.
        """
        m = types.ModuleType("strata_inspect")
        m.read_headers = lambda sources: (fp_dict.get("arch", "qwen4exp"), [])
        m.fingerprint = lambda arch, tensors: fp_dict
        m.verdict = lambda fp, t: (kind, f"mock verdict: {kind}")
        m.PREVIEW_ARCHS = {}
        m.POLY_ARCHS = {}
        if table is None:
            _real_path = Path(_ROOT) / "data" / "gguf_fingerprints.json"
            m.load_fingerprints = lambda path=None: json.loads(_real_path.read_text(encoding="utf-8"))
        else:
            m.load_fingerprints = lambda path=None: table
        return m

    def test_known_verdict(self):
        """A single .gguf whose fp matches exactly one fingerprint entry → (family, model, [shard])."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "qwen36-model.gguf"
            _write_gguf(p)

            # Use the first real entry from the fingerprints file to guarantee a match
            fp_table = json.loads(
                (Path(_ROOT) / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8")
            )
            known = {k: v for k, v in fp_table.items() if not k.startswith("_")}
            first_key, first_entry = next(iter(known.items()))

            stub = self._make_stub(fp_dict=first_entry, kind="known")
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                family, model, shards = setup.resolve_custom_gguf(str(p))

        fam_model = first_key.split("/")
        self.assertEqual(family, fam_model[0])
        self.assertEqual(model, fam_model[1])
        self.assertEqual(shards, [p])
        self.assertTrue(shards[0].suffix == ".gguf")

    def test_multi_match_fails(self):
        """fp matches 2+ entries in the table → SystemExit naming both candidates."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.gguf"
            _write_gguf(p)

            # Craft a table where two keys map to identical fp values → multi-match
            fake_fp = {"arch": "qwen4exp", "tensors": 100, "bytes": 1000,
                       "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}}
            table = {"fam_a/m1": fake_fp, "fam_b/m2": fake_fp}

            stub = self._make_stub(fp_dict=fake_fp, kind="known", table=table)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                try:
                    setup.resolve_custom_gguf(str(p))
                except SystemExit:
                    pass  # expected: fail() exits with code 1
                else:
                    self.fail("expected SystemExit when fp matches multiple table entries")

    def test_no_verdict_fails(self):
        """Verdict kind is 'no' (architecture not recognised) → SystemExit."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mystery.gguf"
            _write_gguf(p)

            stub = self._make_stub(
                fp_dict={"arch": "unknown_arch", "tensors": 0, "bytes": 0,
                         "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}},
                kind="no",
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                try:
                    setup.resolve_custom_gguf(str(p))
                except SystemExit:
                    pass  # expected
                else:
                    self.fail("expected SystemExit for unrecognized GGUF")

    def test_missing_file_fails(self):
        """A .gguf path that does not exist → SystemExit with 'does not exist'."""
        try:
            setup.resolve_custom_gguf(str(Path("/nonexistent_dir_nope/nope.gguf")))
        except SystemExit:
            pass  # expected
        else:
            self.fail("expected SystemExit for missing file")

    def test_folder_mode_groups_by_stem(self):
        """A folder with two shard files of the same model → resolved to those shards.

        Shard filenames include '-NNN-of-MMM.gguf'; group-by-stem should collect them together.
        """
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            s1 = d / "model-Q4_K_M-00001-of-00010.gguf"
            s2 = d / "model-Q4_K_M-00002-of-00010.gguf"
            _write_gguf(s1)
            _write_gguf(s2)

            fp_table = json.loads(
                (Path(_ROOT) / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8")
            )
            known = {k: v for k, v in fp_table.items() if not k.startswith("_")}
            first_key, first_entry = next(iter(known.items()))

            stub = self._make_stub(fp_dict=first_entry, kind="known")
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                family, model, shards = setup.resolve_custom_gguf(str(d))

        fam_model = first_key.split("/")
        self.assertEqual(family, fam_model[0])
        self.assertEqual(model, fam_model[1])
        self.assertEqual(len(shards), 2)
        # Shards should be sorted by size (both same size here so order is alphabetical)
        names = [s.name for s in shards]
        self.assertIn("model-Q4_K_M-00001-of-00010.gguf", names)
        self.assertIn("model-Q4_K_M-00002-of-00010.gguf", names)

    def test_folder_no_gguf_files_fails(self):
        """A folder with only non-GGUF files → SystemExit saying 'has no .gguf files'."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "readme.txt").write_text("hello")
            try:
                setup.resolve_custom_gguf(str(d))
            except SystemExit:
                pass  # expected
            else:
                self.fail("expected SystemExit for folder with no .gguf files")

    def test_folder_multiple_models_fails(self):
        """A folder with shards from two different models (different stems) → SystemExit."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            a1 = d / "model_a-00001-of-00010.gguf"
            b1 = d / "model_b-00001-of-00005.gguf"
            _write_gguf(a1)
            _write_gguf(b1)
            try:
                setup.resolve_custom_gguf(str(d))
            except SystemExit:
                pass  # expected
            else:
                self.fail("expected SystemExit for folder with multiple models")

    def test_family_kind_quant_mapping(self):
        """Verdict kind 'family' → derive size from GGUF_QUANT match or filename token."""
        with tempfile.TemporaryDirectory() as tmp:
            # Find a family that is not in the fingerprints table so verdict returns "family".
            # Use qwen36 from FAMILIES; file name must end in one of its model sizes.
            p = Path(tmp) / "model-UD-Q3_K_M.gguf"
            _write_gguf(p)

            fp_dict = {"arch": "some_preview_arch", "tensors": 0, "bytes": 0,
                       "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}}
            stub = self._make_stub(fp_dict=fp_dict, kind="family")

            # Patch PREVIEW_ARCHS to map our arch to a family title in FAMILIES
            stub.PREVIEW_ARCHS = {"some_preview_arch": setup.FAMILIES["qwen36"]["title"]}

            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                family, model, shards = setup.resolve_custom_gguf(str(p))

        # FAMILIES["qwen36"] has title "Qwen 3.6" (verify) and models dict with sizes
        self.assertTrue(family in setup.FAMILIES)
        self.assertIn(model, setup.FAMILIES[family]["models"])
        self.assertEqual(shards, [p])

    def test_folder_mode_nonquant_size(self):
        """Folder mode: shard filename ending in a non-quant size (e.g. COMPACT) → resolved correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            s1 = d / "Ornith-APEX-I-COMPACT-00001-of-00004.gguf"
            s2 = d / "Ornith-APEX-I-COMPACT-00002-of-00004.gguf"
            _write_gguf(s1)
            _write_gguf(s2)

            fp_dict = {"arch": "ornith_arch", "tensors": 0, "bytes": 0,
                       "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}}
            stub = self._make_stub(fp_dict=fp_dict, kind="family")
            stub.PREVIEW_ARCHS = {"ornith_arch": setup.FAMILIES["ornith"]["title"]}

            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                family, model, shards = setup.resolve_custom_gguf(str(d))

        self.assertEqual(family, "ornith")
        self.assertEqual(model, "APEX-I-COMPACT")
        self.assertEqual(len(shards), 2)

    def test_conflicting_flags_fail(self):
        """main() with --custom-gguf + conflicting --family/--model → SystemExit (fail)."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.gguf"
            _write_gguf(p)

            fp_table = json.loads(
                (Path(_ROOT) / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8")
            )
            known = {k: v for k, v in fp_table.items() if not k.startswith("_")}
            first_key, first_entry = next(iter(known.items()))

            stub = self._make_stub(fp_dict=first_entry, kind="known")
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                # --family swift conflicts with resolved "qwen" → fail before GPU detection
                argv = ["setup.py", "--custom-gguf", str(p), "--family", "swift", "--model", "IQ3_S"]
                stack.enter_context(mock.patch.object(sys, "argv", argv))
                try:
                    setup.main()
                except SystemExit as e:
                    self.assertEqual(e.code, 1)
                else:
                    self.fail("expected SystemExit on conflicting flags")

    def test_resolved_pair_not_in_models_fails(self):
        """main() with --custom-gguf resolving to a model not in the target family's size list → fail."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.gguf"
            _write_gguf(p)

            # Fake table: key says swift/FAKE_SIZE (not a real swift model)
            fake_fp = {"arch": "swift_arch", "tensors": 42, "bytes": 999,
                       "expert_bytes": {}, "experts_bpw": None, "bpw": None, "roles": {}}
            table = {"swift/FAKE_SIZE": fake_fp}

            stub = self._make_stub(fp_dict=fake_fp, kind="known", table=table)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"strata_inspect": stub}))
                # No --family/--model flags → resolved (swift, FAKE_SIZE) must pass validation.
                # FAKE_SIZE is NOT in family_models("swift") → fail before GPU detection.
                argv = ["setup.py", "--custom-gguf", str(p)]
                stack.enter_context(mock.patch.object(sys, "argv", argv))
                try:
                    setup.main()
                except SystemExit as e:
                    self.assertEqual(e.code, 1)
                else:
                    self.fail("expected SystemExit when resolved model not in family's size list")


if __name__ == "__main__":
    unittest.main(verbosity=2)
