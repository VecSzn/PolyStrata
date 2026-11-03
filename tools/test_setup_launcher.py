"""Model -> quant -> context choices and platform handoff, without installation or downloads."""
from __future__ import annotations
import contextlib
import io
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import setup_launcher as L


class Menu(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(L, 'saved_choices', return_value=[]))
        self.stack.enter_context(mock.patch.object(L.S, 'ram_gb', return_value=64))
        from metal import setup_mac as M
        self.stack.enter_context(mock.patch.object(M, 'mac_info', return_value={'ram_gb': 32}))
        self.stack.enter_context(mock.patch.object(sys.stdin, 'isatty', return_value=True))
        self.stack.enter_context(mock.patch.object(sys.stdout, 'isatty', return_value=True))

    def run_menu(self, platform, argv=(), answers=()):
        output = io.StringIO()
        with mock.patch('builtins.input', side_effect=answers) as prompt, \
                mock.patch.object(L.os, 'execv') as execute, contextlib.redirect_stdout(output):
            # StringIO is not a TTY; make its terminal status explicit for this test.
            with mock.patch.object(output, 'isatty', return_value=True):
                status = L.main(argv, platform)
        return status, execute, prompt, output.getvalue()

    def test_mac_catalog_is_only_native_supported_families_and_quants(self):
        from metal import setup_mac as M
        entries = L.catalog('darwin')
        self.assertEqual(list(entries), list(L.S.FAMILIES) + list(L.S.OWN_PATH_FAMILIES))
        for family, entry in entries.items():
            expected = (L.S.OWN_PATH_FAMILIES[family]['models'] if family in L.S.OWN_PATH_FAMILIES
                        else L.S.family_models(family))
            self.assertEqual(set(entry['models']), set(expected))
        self.assertIn('IQ2_XS', entries['qwen']['models'])
        self.assertIn('IQ1_M', entries['coder']['models'])

    def test_linux_catalog_matches_installer_including_own_engine(self):
        entries = L.catalog('linux')
        self.assertEqual(list(entries), L.S.menu_families(False) + list(L.S.OWN_PATH_FAMILIES))
        for family, entry in entries.items():
            expected = (L.S.OWN_PATH_FAMILIES[family]['models'] if family in L.S.OWN_PATH_FAMILIES
                        else L.S.family_models(family))
            self.assertEqual(entry['models'], expected)
        self.assertNotIn('nvfp4', L.catalog('linux', 'hip'))

    def test_both_platforms_prompt_in_order_then_handoff_selected_values(self):
        for platform, backend in [('darwin', ROOT / 'metal/setup_mac.py'), ('linux', ROOT / 'setup.py')]:
            with self.subTest(platform=platform):
                status, execute, prompt, output = self.run_menu(platform, ['--no-start', '--data-dir', '/some folder'],
                                                               ['lfm25', 'UD-Q3_K_XL', '2048'])
                self.assertEqual(status, 0)
                self.assertEqual(prompt.call_count, 3)
                self.assertLess(output.index('1. Model'), output.index('2. Quantization'))
                self.assertLess(output.index('2. Quantization'), output.index('3. Context'))
                execute.assert_called_once_with(sys.executable, [sys.executable, str(backend), '--no-start',
                    '--data-dir', '/some folder', '--family', 'lfm25', '--model', 'UD-Q3_K_XL', '--context', '2048'])

    def test_partial_explicit_choices_only_ask_for_missing_fields(self):
        _, execute, prompt, output = self.run_menu('darwin', ['--family=gemma4', '--context', '4096'], ['UD-Q3_K_M'])
        self.assertEqual(prompt.call_count, 1)
        self.assertNotIn('1. Model', output)
        self.assertNotIn('3. Context', output)
        self.assertEqual(execute.call_args.args[1][-2:], ['--model', 'UD-Q3_K_M'])

    def test_complete_explicit_choices_preserve_arguments_and_do_not_prompt(self):
        args = ['--family', 'lfm25', '--model', 'QAD-Q4_0', '--context=4096', '--api-key', '$literal secret']
        _, execute, prompt, _ = self.run_menu('darwin', args)
        prompt.assert_not_called()
        self.assertEqual(execute.call_args.args[1][2:], args)

    def test_hf_token_is_forwarded_to_both_installers_without_printing(self):
        for platform in ('darwin', 'linux'):
            for option in (['--hf-token', 'hf_test_only'], ['--hf-token=hf_test_only']):
                args = ['--family', 'lfm25', '--model', 'QAD-Q4_0', '--context=4096', *option]
                _, execute, prompt, output = self.run_menu(platform, args)
                prompt.assert_not_called()
                self.assertEqual(execute.call_args.args[1][2:], args)
                self.assertNotIn('hf_test_only', output)

    def test_administrative_and_yes_commands_bypass_choices(self):
        for flag in ('--yes', '--check', '--help', '--update', '--inspect', '--calibrate'):
            with self.subTest(flag=flag):
                with mock.patch.object(L, 'choose', side_effect=AssertionError('menu was called')):
                    _, execute, prompt, _ = self.run_menu('darwin', [flag])
                    prompt.assert_not_called()
                    self.assertEqual(execute.call_args.args[1][2:], [flag])

    def test_list_models_on_both_platforms_never_runs_installer(self):
        for platform in ('darwin', 'linux'):
            status, execute, prompt, output = self.run_menu(platform, ['--list-models'])
            self.assertEqual(status, 0)
            execute.assert_not_called()
            prompt.assert_not_called()
            for family, entry in L.catalog(platform).items():
                self.assertIn(family + ':', output)
                for quant in entry['models']:
                    self.assertIn(quant, output)
            self.assertIn('Context tokens:', output)

    def test_redirected_input_bypasses_menu(self):
        with mock.patch.object(sys.stdin, 'isatty', return_value=False):
            _, execute, prompt, _ = self.run_menu('linux', ['--no-start'])
        prompt.assert_not_called()
        self.assertEqual(execute.call_args.args[1][2:], ['--no-start'])

    def test_saved_family_quant_and_custom_context_are_defaults(self):
        L.saved_choices.return_value = [{'family': 'lfm25', 'model': 'QAD-Q4_0', 'context': 12345}]
        _, execute, _, output = self.run_menu('darwin', answers=['', '', ''])
        self.assertEqual(execute.call_args.args[1][-6:], ['--family', 'lfm25', '--model', 'QAD-Q4_0', '--context', '12345'])
        self.assertIn('(installed)', output)
        self.assertIn('12,345 tokens', output)

    def test_bad_entries_retry_and_custom_context_is_validated(self):
        _, execute, prompt, _ = self.run_menu('darwin', ['--family', 'lfm25', '--model', 'UD-Q3_K_XL'],
                                            ['invalid', 'custom', '0', 'not a number', '1048577', '4096'])
        self.assertEqual(prompt.call_count, 6)
        self.assertEqual(execute.call_args.args[1][-2:], ['--context', '4096'])

    def test_numbered_choices(self):
        entries = L.catalog('darwin')
        family_number = str(list(entries).index('lfm25') + 1)
        quant_number = str(list(entries['lfm25']['models']).index('UD-Q3_K_XL') + 1)
        _, execute, _, _ = self.run_menu('darwin', answers=[family_number, quant_number, '1'])
        self.assertEqual(execute.call_args.args[1][-6:], ['--family', 'lfm25', '--model', 'UD-Q3_K_XL', '--context', '2048'])

    def test_quit_and_eof_do_not_run_installer(self):
        for answers in (['q'], [EOFError], ['lfm25', 'q'], ['lfm25', 'UD-Q3_K_XL', 'q']):
            status, execute, _, _ = self.run_menu('darwin', answers=answers)
            self.assertEqual(status, 130)
            execute.assert_not_called()

    def test_invalid_family_quant_and_context_do_not_run_installer(self):
        for args in (['--family', 'unknown'], ['--family', 'lfm25', '--model', 'fake'],
                     ['--family', 'lfm25', '--model', 'UD-Q3_K_XL', '--context', '0']):
            with contextlib.redirect_stderr(io.StringIO()):
                status, execute, _, _ = self.run_menu('darwin', args)
            self.assertEqual(status, 2)
            execute.assert_not_called()

    def test_quant_without_family_only_offers_compatible_families(self):
        _, execute, _, output = self.run_menu('darwin', ['--model', 'QAD-Q4_0', '--context', '2048'], ['lfm25'])
        self.assertIn('LFM2.5', output)
        self.assertNotIn('Qwen3.8', output)
        self.assertEqual(execute.call_args.args[1][-2:], ['--family', 'lfm25'])


@unittest.skipIf(os.name == 'nt', 'POSIX launcher')
class TerminalHandoff(unittest.TestCase):
    def test_shell_menu_and_real_exec_for_both_platform_routes(self):
        import pty
        # Run the actual shell and menu with real Python. Only hardware/config discovery and the final
        # installers are replaced; the latter record argv so this cannot install packages or models.
        for platform in ('darwin', 'linux'):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory(prefix='strata menu ') as folder:
                root = Path(folder)
                shutil.copyfile(ROOT / 'setup.sh', root / 'setup.sh')
                (root / '.venv/bin').mkdir(parents=True)
                (root / '.venv/bin/python').symlink_to(sys.executable)
                (root / 'tools').mkdir()
                (root / 'metal').mkdir()
                for target in ('setup.py', 'metal/setup_mac.py'):
                    (root / target).write_text('import json, sys\nprint("HANDOFF=" + json.dumps(sys.argv), flush=True)\n')
                wrapper = (f'import sys\nsys.path.insert(0, {str(ROOT / "tools")!r})\nimport setup_launcher as L\n'
                           f'from pathlib import Path\nL.ROOT = Path({str(root)!r})\n'
                           'L.saved_choices = lambda *a: []\nL.S.ram_gb = lambda: 64\n'
                           'from metal import setup_mac as M\nM.mac_info = lambda: {"ram_gb": 32}\n'
                           f'raise SystemExit(L.main(platform={platform!r}))\n')
                (root / 'tools/setup_launcher.py').write_text(wrapper)
                master, slave = pty.openpty()
                try:
                    with subprocess.Popen(['/bin/sh', str(root / 'setup.sh'), '--no-start'], stdin=slave,
                                          stdout=slave, stderr=slave, cwd=folder) as process:
                        os.close(slave)
                        os.write(master, b'lfm25\nUD-Q3_K_XL\n2048\n')
                        output = b''
                        deadline = time.monotonic() + 15
                        try:
                            while time.monotonic() < deadline:
                                if select.select([master], [], [], .1)[0]:
                                    try: chunk = os.read(master, 4096)
                                    except OSError: break
                                    if not chunk: break
                                    output += chunk
                                if process.poll() is not None: break
                            self.assertEqual(process.wait(timeout=3), 0, output.decode())
                        finally:
                            if process.poll() is None: process.kill()
                finally:
                    os.close(master)
                text = output.decode()
                handoff = next(line.removeprefix('HANDOFF=') for line in text.splitlines() if line.startswith('HANDOFF='))
                args = json.loads(handoff)
                backend = 'metal/setup_mac.py' if platform == 'darwin' else 'setup.py'
                self.assertEqual(Path(args[0]).resolve(), (root / backend).resolve())
                self.assertEqual(args[1:], ['--no-start', '--family', 'lfm25', '--model', 'UD-Q3_K_XL', '--context', '2048'])
                self.assertLess(text.index('1. Model'), text.index('2. Quantization'))
                self.assertLess(text.index('2. Quantization'), text.index('3. Context'))


if __name__ == '__main__':
    unittest.main()
