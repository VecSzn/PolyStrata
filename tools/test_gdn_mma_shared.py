"""The shared GDN consumer must build without model or engine objects."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SharedGdnBoundary(unittest.TestCase):
    def test_public_kernel_is_independent_of_poly(self):
        header = ROOT / 'include/strata/kernels/gdn_mma.hpp'
        self.assertTrue(header.is_file(), 'missing model-independent GDN consumer API')
        text = header.read_text(encoding='utf-8')
        self.assertNotIn('strata::poly', text)
        self.assertNotIn('engine.hpp', text)
        cmake = (ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')
        target = re.search(r'add_executable\(gdn_mma_test\s+([^)]*)\)', cmake)
        self.assertIsNotNone(target, 'missing standalone shared-kernel consumer')
        self.assertNotIn('src/poly/', target.group(1))
        link = re.search(r'target_link_libraries\(gdn_mma_test\s+([^)]*)\)', cmake)
        self.assertIsNotNone(link)
        self.assertNotIn('strata_poly', link.group(1))
        self.assertIn('strata_gdn_mma', link.group(1))


if __name__ == '__main__':
    unittest.main()
