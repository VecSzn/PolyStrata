"""--gguf-dir with the user's own GGUF (issue 54): for a strata-poly family, another file of the family's architecture
is taken with a warning (docs/FAQ.md); another architecture or an incomplete copy is refused; and setup never deletes
a file in the user's own folder, whatever its hash."""
import contextlib
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf(path, arch, floats=64, short=0):
    """A GGUF v3 file of architecture `arch` with one F32 tensor of `floats` values, `short` bytes of its data cut."""
    kv = _str("general.architecture") + struct.pack("<I", 8) + _str(arch)
    tinfo = _str("w") + struct.pack("<I", 1) + struct.pack("<Q", floats) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + kv + tinfo
    head += b"\0" * ((32 - len(head) % 32) % 32)
    path.write_bytes(head + b"\0" * (floats * 4 - short))


def gguf_tensors(path, arch, tensors):
    """A GGUF v3 file of architecture `arch` with (name, ggml type id, values) tensors, F32 (0) or F16 (1)."""
    kv = _str("general.architecture") + struct.pack("<I", 8) + _str(arch)
    tinfo, off, data = b"", 0, b""
    for name, ttype, n in tensors:
        tinfo += _str(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<I", ttype) + struct.pack("<Q", off)
        size = n * (4 if ttype == 0 else 2)
        data += b"\0" * size + b"\0" * ((32 - size % 32) % 32)
        off = len(data)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1) + kv + tinfo
    head += b"\0" * ((32 - len(head) % 32) % 32)
    path.write_bytes(head + data)


def quiet(f, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
        try:
            f(*args, **kw)
        except SystemExit:
            return "stopped", out.getvalue()
    return "ok", out.getvalue()


class OwnGguf(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.f = self.d / "Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf"

    def test_another_file_of_the_architecture_is_taken(self):
        gguf(self.f, "qwen35moe")
        code, out = quiet(setup.verify_poly_own, self.f, 13_600_000_000, "0" * 64, "tiel")
        self.assertEqual(code, "ok", out)
        self.assertIn("not the file setup offers", out)
        self.assertIn("own file", self.f.with_name(self.f.name + ".done").read_text())
        self.assertEqual(quiet(setup.verify_poly_own, self.f, 13_600_000_000, "0" * 64, "tiel")[0], "ok")

    def test_another_architecture_is_refused_and_kept(self):
        gguf(self.f, "llama")
        self.assertEqual(quiet(setup.verify_poly_own, self.f, 13_600_000_000, "0" * 64, "tiel")[0], "stopped")
        self.assertTrue(self.f.exists())
        gguf(self.f, "qwen35moe")                  # a Qwen3.6-type file is not a Gemma 4 one
        self.assertEqual(quiet(setup.verify_poly_own, self.f, 13_600_000_000, "0" * 64, "gemma4")[0], "stopped")

    def test_gate_and_up_of_different_types_are_refused(self):
        # issue 54: Tiel APEX-I-NanoPlus, whose layer 16 stores gate and up differently; the engine refuses it at load
        gguf_tensors(self.f, "qwen35moe", [("blk.16.ffn_gate_exps.weight", 0, 64), ("blk.16.ffn_up_exps.weight", 1, 64)])
        code, out = quiet(setup.verify_poly_own, self.f, 13_600_000_000, "0" * 64, "tiel")
        self.assertEqual(code, "stopped")
        self.assertIn("blk.16: F32 and F16", out)
        gguf_tensors(self.f, "qwen35moe", [("blk.16.ffn_gate_exps.weight", 1, 64), ("blk.16.ffn_up_exps.weight", 1, 64)])
        self.assertEqual(quiet(setup.verify_poly_own, self.f, 13_600_000_000, "0" * 64, "tiel")[0], "ok")

    def test_an_incomplete_copy_is_refused(self):
        gguf(self.f, "qwen35moe", short=100)
        self.assertEqual(quiet(setup.verify_poly_own, self.f, 13_600_000_000, "0" * 64, "tiel")[0], "stopped")
        self.assertTrue(self.f.exists())

    def test_ones_own_file_in_setups_folder_is_pointed_to_gguf_dir(self):
        # issue 54: renamed into setup's own models folder, without --gguf-dir
        gguf(self.f, "qwen35moe")
        code, out = quiet(setup.verify_poly_download, self.f, 13_600_000_000, "0" * 64, "tiel")
        self.assertEqual(code, "stopped")
        self.assertIn("--gguf-dir", out)
        self.assertNotIn("delete it and run setup again (the download restarts)", out)
        self.assertTrue(self.f.exists())
        gguf(self.f, "qwen35moe", short=100)       # a truncated download is still one to fetch again
        code, out = quiet(setup.verify_poly_download, self.f, 13_600_000_000, "0" * 64, "tiel")
        self.assertIn("the download restarts", out)

    def test_a_wrong_hash_deletes_only_what_setup_downloaded(self):
        gguf(self.f, "qwen35moe")
        size = self.f.stat().st_size
        self.assertEqual(quiet(setup.verify_sha256, self.f, size, "0" * 64, own=True)[0], "stopped")
        self.assertTrue(self.f.exists())           # the user's own folder: left alone
        self.assertEqual(quiet(setup.verify_sha256, self.f, size, "0" * 64)[0], "stopped")
        self.assertFalse(self.f.exists())          # setup's download: deleted so the next run fetches it again


if __name__ == "__main__":
    unittest.main()
