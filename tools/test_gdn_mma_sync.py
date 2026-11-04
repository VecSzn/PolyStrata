"""Verify the GDN copy/provenance and the pinned MMA dependency; no GPU or network.

STRATA_GGML_DIR selects the pinned llama.cpp checkout. To independently recheck the
kernel body, STRATA_GDN_UPSTREAM_SOURCE may name gated-delta-net-mma.cu from the
recorded upstream revision. Runtime integration is tested by poly_gdn_mma_test.
"""
import hashlib
import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / 'src/kernels/cuda'
META = json.loads((KERNEL / 'gdn_mma_provenance.json').read_text())

def sha(data):
    return hashlib.sha256(data).hexdigest()

def kernel_body():
    text = (KERNEL / 'gdn_mma.cu').read_text()
    return text[text.index('namespace {\nconstexpr int D = 128;'):
                text.index('// The kernel above is unchanged')].strip()

class GdnMmaSync(unittest.TestCase):
    def test_kernel_body_matches_recorded_upstream(self):
        self.assertEqual(sha(kernel_body().encode()), META['kernel_body_sha256'],
                         'GDN body changed: compare with the recorded upstream before updating provenance')

    def test_imported_helpers_match_recorded_copy(self):
        self.assertEqual(sha((KERNEL / 'gdn_mma_helpers.cuh').read_bytes()), META['helpers_sha256'])

    def test_pinned_mma_dependency(self):
        ggml = Path(os.environ.get('STRATA_GGML_DIR', ROOT / 'third_party/llama.cpp'))
        path = ggml / 'ggml/src/ggml-cuda/mma.cuh'
        if not path.is_file():
            self.skipTest('no pinned llama.cpp checkout')
        self.assertEqual(sha(path.read_bytes()), META['pinned_mma_sha256'],
                         'Pinned MMA changed: audit the imported GDN helpers and rerun GPU tests')

    def test_original_upstream_source_when_available(self):
        source = os.environ.get('STRATA_GDN_UPSTREAM_SOURCE')
        if not source:
            self.skipTest('STRATA_GDN_UPSTREAM_SOURCE not supplied; recorded copy checks still run')
        data = Path(source).read_bytes()
        self.assertEqual(sha(data), META['upstream_file_sha256'])
        text = data.decode()
        self.assertTrue(text[text.index('namespace {\nconstexpr int D = 128;'):].startswith(kernel_body()))

if __name__ == '__main__':
    unittest.main()
