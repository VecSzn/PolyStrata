"""src/poly/fattn_mma.cuh is llama.cpp's fattn-mma-f16.cuh with strata-poly's changes (src/poly/fattn_mma.patch).
These tests fail when the llama.cpp the build pins has a different fattn-mma-f16.cuh than the one the copy was made
from, or when the patch no longer turns that file into the copy, so a move of the pin cannot leave the copy behind
unnoticed.  They need the pinned checkout (third_party/llama.cpp, which setup fetches, or STRATA_GGML_DIR) and git;
without them they are skipped.  No GPU.

    python -m unittest tools.test_fattn_sync

To re-port after the pin moved: copy the new ggml/src/ggml-cuda/fattn-mma-f16.cuh to src/poly/fattn_mma.cuh, apply
the patch (git apply --reject, then fix what did not fit), check with poly_fattn_test, then write the patch again
(diff -u of the upstream file and the copy, paths a/fattn_mma.cuh and b/fattn_mma.cuh) and the sha256 in the
copy's header.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / "src/poly/fattn_mma.cuh"
PATCH = ROOT / "src/poly/fattn_mma.patch"


def upstream() -> Path | None:
    base = Path(os.environ.get("STRATA_GGML_DIR") or ROOT / "third_party/llama.cpp")
    f = base / "ggml/src/ggml-cuda/fattn-mma-f16.cuh"
    return f if f.is_file() else None


@unittest.skipUnless(upstream(), "no pinned llama.cpp checkout (third_party/llama.cpp or STRATA_GGML_DIR)")
class FattnSync(unittest.TestCase):
    def test_upstream_is_the_one_copied(self):
        want = re.search(r"sha256 is\n// ([0-9a-f]{64})", COPY.read_text()).group(1)
        got = hashlib.sha256(upstream().read_bytes()).hexdigest()
        self.assertEqual(got, want, "llama.cpp's fattn-mma-f16.cuh changed at the pinned commit: re-port "
                                    "src/poly/fattn_mma.cuh (see this file's docstring)")

    @unittest.skipUnless(shutil.which("git"), "no git")
    def test_patch_makes_the_copy(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "fattn_mma.cuh"
            shutil.copyfile(upstream(), work)
            # outside any repository Git for Windows' system core.autocrlf=true would still write CRLF
            r = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn", str(PATCH)], cwd=d,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, "src/poly/fattn_mma.patch does not apply to the pinned upstream file: "
                                              + r.stderr)
            self.assertEqual(work.read_bytes(), COPY.read_bytes(),
                             "src/poly/fattn_mma.patch applied to the upstream file is not src/poly/fattn_mma.cuh: "
                             "write the patch again (see this file's docstring)")


if __name__ == "__main__":
    unittest.main()
