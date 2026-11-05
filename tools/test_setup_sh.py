"""Exercise the actual POSIX launcher without network, sudo, GPU setup, or model downloads.

Only the external Python/package/download commands are fixtures; setup.sh itself runs unchanged,
from a directory outside a checkout whose path contains spaces. Run on macOS or Linux.
"""
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

if os.name != 'nt':
    import pty

ROOT = Path(__file__).resolve().parents[1]
PIN = "83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8"
# One fixture executable, installed under the names a scenario makes available on PATH.
STUB = r'''
import json, os, signal, sys, time
from pathlib import Path
root = Path(os.environ['BOOTSTRAP_FIXTURE'])
state_path = root / 'state.json'
state = json.loads(state_path.read_text())
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with (root / 'calls.jsonl').open('a') as f:
    f.write(json.dumps({'name': name, 'args': args, 'cwd': os.getcwd(), 'pid': os.getpid()}) + '\n')
def save(): state_path.write_text(json.dumps(state))
def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((root / 'stub').read_text()); path.chmod(0o755)
def venv():
    if state.get('venv_fail'): sys.exit(8)
    executable(Path('.venv/bin/python'))
    state['broken_pip'] = state.get('created_without_pip', False)
    state['old_venv'] = False
    save()
if name == 'uname': print(state.get('platform', 'Linux') if args == ['-s'] else state.get('arch', 'arm64'))
elif name == 'grep': sys.exit(0 if state.get('nixos') else 1)
elif name == 'id': print(state.get('uid', 501))
elif name == 'sudo': os.execv(str(root / 'bin' / args[0]), args)
elif name in ('apt-get', 'dnf', 'pacman'):
    if state.get('package_fail'): sys.exit(17)
    if args[0] != 'update': executable(root / 'bin' / ('python' if name == 'pacman' else 'python3'))
elif name == 'uv':
    assert args == ['venv', '--managed-python', '--seed', '--python', '3.12', '.venv'], args
    venv()
elif name == 'curl':
    if state.get('download_fail'): sys.exit(22)
    Path(args[args.index('-o') + 1]).write_bytes(b'archive fixture')
elif name == 'shasum':
    print(('bad-hash' if state.get('hash_fail') else os.environ['BOOTSTRAP_PIN']) + '  ' + args[-1])
elif name == 'tar':
    executable(Path(args[args.index('-C') + 1]) / 'python/bin/python3')
    if state.get('extract_fail'): sys.exit(2)
elif name.startswith('python'):
    in_venv = '.venv' in Path(sys.argv[0]).parts
    if args[0] == '-c':
        sys.exit(1 if (in_venv and state.get('old_venv')) or (not in_venv and name in state.get('bad_python', [])) else 0)
    elif args[:2] == ['-m', 'pip']: sys.exit(1 if state.get('broken_pip') else 0)
    elif args[:2] == ['-m', 'venv']: venv()
    elif args[0] == 'tools/setup_launcher.py':
        if state.get('read_input'):
            print('INPUT=' + input(), flush=True)
        if state.get('wait'):
            print('READY', flush=True)
            signal.pause()
        sys.exit(state.get('setup_status', 0))
    else: raise AssertionError(args)
else: raise AssertionError((name, args))
'''


@unittest.skipIf(os.name == 'nt', 'POSIX shell launcher')
class ShellBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='strata launcher ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'checkout with spaces'
        self.repo.mkdir()
        shutil.copyfile(ROOT / 'setup.sh', self.repo / 'setup.sh')
        (self.repo / 'setup.py').write_text('# The fixture interpreter records this handoff.\n')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.stub = '#!' + sys.executable + '\n' + STUB
        (self.root / 'stub').write_text(self.stub)
        self.state = {}
        self.env = dict(os.environ, PATH=str(self.bin), BOOTSTRAP_FIXTURE=str(self.root), BOOTSTRAP_PIN=PIN)
        self.env.pop('STRATA_UV', None)
        # No real installer/downloader/Python can leak through this isolated PATH.
        for name in ('dirname', 'rm', 'mkdir', 'mv', 'mktemp'):
            (self.bin / name).symlink_to(shutil.which(name))
        for name in ('uname', 'grep', 'id'):
            self.command(name)

    def command(self, name):
        path = self.bin / name
        path.write_text(self.stub)
        path.chmod(0o755)

    def private_python(self, folder='.venv/bin/python'):
        path = self.repo / folder
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.stub)
        path.chmod(0o755)

    def prepare(self):
        (self.root / 'state.json').write_text(json.dumps(self.state))

    def run_launcher(self, *args):
        self.prepare()
        return subprocess.run(['/bin/sh', str(self.repo / 'setup.sh'), *args], cwd=self.root,
                              env=self.env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)

    def calls(self, name=None):
        path = self.root / 'calls.jsonl'
        calls = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        return [c for c in calls if name is None or c['name'] == name]

    def setup_calls(self):
        return [c for c in self.calls() if c['args'] and c['args'][0] == 'tools/setup_launcher.py']

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.setup_calls()), 1)
        self.assertEqual(Path(self.setup_calls()[0]['cwd']).resolve(), self.repo.resolve())

    def mac_download(self):
        self.state['platform'] = 'Darwin'
        for name in ('curl', 'shasum', 'tar'):
            self.command(name)

    def test_reuse_environment_and_forward_arguments_exactly(self):
        self.private_python()
        args = ('--family', 'lfm25', '--data-dir', '/path with spaces/$literal', '--yes', '--no-start',
                '--hf-token', 'hf_test_only', '')
        self.assert_success(self.run_launcher(*args))
        self.assertEqual(self.setup_calls()[0]['args'], ['tools/setup_launcher.py', *args])
        self.assertFalse(self.calls('uname'))  # no platform/bootstrap work for an installed environment

    def test_setup_failure_preserves_status_without_noninteractive_pause(self):
        self.private_python()
        self.state['setup_status'] = 37
        result = self.run_launcher('--check')
        self.assertEqual(result.returncode, 37)
        self.assertNotIn('Press Enter', result.stdout)

    def test_find_python_and_create_private_environment_on_both_platforms(self):
        self.command('python3')
        for platform in ('Linux', 'Darwin'):
            with self.subTest(platform=platform):
                self.state['platform'] = platform
                result = self.run_launcher('--check')
                self.assertEqual(result.returncode, 0, result.stderr)
                shutil.rmtree(self.repo / '.venv')
        self.assertEqual(len([c for c in self.calls('python3') if c['args'][:2] == ['-m', 'venv']]), 2)
        self.assertFalse(self.calls('curl'))

    def test_skip_incompatible_python_and_find_versioned_python(self):
        self.command('python3')
        self.command('python3.12')
        self.state['bad_python'] = ['python3']
        self.assert_success(self.run_launcher())
        self.assertTrue(any(c['args'][:2] == ['-m', 'venv'] for c in self.calls('python3.12')))

    def test_repair_environment_without_pip(self):
        self.private_python()
        self.state['broken_pip'] = True
        self.command('python3')
        self.assert_success(self.run_launcher())
        self.assertTrue(any(c['args'][:2] == ['-m', 'venv'] for c in self.calls('python3')))

    def test_repair_environment_with_old_or_32bit_python(self):
        self.private_python()
        self.state['old_venv'] = True
        self.command('python3')
        self.assert_success(self.run_launcher())
        self.assertTrue(any(c['args'][:2] == ['-m', 'venv'] for c in self.calls('python3')))

    def test_repair_an_activated_broken_environment_does_not_select_its_python(self):
        self.private_python()
        self.private_python('.venv/bin/python3')
        self.state['broken_pip'] = True
        self.env['PATH'] = str(self.repo / '.venv/bin') + os.pathsep + str(self.bin)
        self.command('apt-get')
        self.command('sudo')
        self.assert_success(self.run_launcher())
        self.assertTrue(self.calls('apt-get'))

    def test_real_python_creates_a_working_venv_in_a_path_with_spaces(self):
        (self.bin / 'python3').symlink_to(sys.executable)
        (self.repo / 'tools').mkdir()
        (self.repo / 'tools/setup_launcher.py').write_text(
            'import json, sys\nprint("LAUNCH=" + json.dumps([sys.prefix, *sys.argv[1:]]))\n')
        result = self.run_launcher('--data-dir', '/another path with spaces', '--check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = json.loads(next(s[7:] for s in result.stdout.splitlines() if s.startswith('LAUNCH=')))
        self.assertEqual(Path(args[0]).resolve(), (self.repo / '.venv').resolve())
        self.assertEqual(args[1:], ['--data-dir', '/another path with spaces', '--check'])

    def test_failed_environment_creation_does_not_run_setup(self):
        self.command('python3')
        self.state['venv_fail'] = True
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('Could not create', result.stderr)
        self.assertFalse(self.setup_calls())

    def test_created_environment_without_pip_is_rejected(self):
        self.command('python3')
        self.state['created_without_pip'] = True
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('incomplete', result.stderr)
        self.assertFalse(self.setup_calls())

    def test_mac_reuses_downloaded_python(self):
        self.state['platform'] = 'Darwin'
        self.private_python('.python/python/bin/python3')
        self.assert_success(self.run_launcher())
        self.assertFalse(self.calls('curl'))

    def test_mac_download_checks_hash_and_publishes_complete_python(self):
        self.mac_download()
        self.assert_success(self.run_launcher())
        self.assertTrue(self.calls('shasum'))
        self.assertTrue((self.repo / '.python/python/bin/python3').exists())
        self.assertFalse(list((self.repo / '.python').glob('bootstrap.*')))
        self.assertFalse(self.calls('sudo'))

    def test_mac_failures_never_publish_partial_python(self):
        self.mac_download()
        for failure in ('download_fail', 'hash_fail', 'extract_fail'):
            with self.subTest(failure=failure):
                self.state[failure] = True
                result = self.run_launcher()
                self.assertEqual(result.returncode, 1)
                self.assertFalse((self.repo / '.python/python').exists())
                self.assertFalse(list((self.repo / '.python').glob('bootstrap.*')))
                self.assertFalse(self.setup_calls())
                del self.state[failure]

    def test_mac_missing_download_tool_has_actionable_error(self):
        self.state['platform'] = 'Darwin'
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('curl is missing', result.stderr)
        self.assertFalse(self.setup_calls())

    def test_mac_does_not_download_arm_python_for_intel_or_rosetta(self):
        self.mac_download()
        self.state['arch'] = 'x86_64'
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('native terminal', result.stderr)
        self.assertFalse(self.calls('curl'))

    def test_linux_package_managers(self):
        for manager in ('apt-get', 'dnf', 'pacman'):
            with self.subTest(manager=manager):
                self.command(manager)
                self.command('sudo')
                self.assertEqual(self.run_launcher().returncode, 0)
                self.assertTrue(self.calls(manager))
                shutil.rmtree(self.repo / '.venv')
                (self.bin / manager).unlink()
                (self.bin / ('python' if manager == 'pacman' else 'python3')).unlink()
        self.assertEqual(len(self.setup_calls()), 3)
        self.assertTrue(self.calls('sudo'))
        self.assertEqual([c['args'] for c in self.calls('apt-get')],
                         [['update'], ['install', '-y', 'python3', 'python3-venv', 'python3-pip']])

    def test_linux_root_does_not_require_sudo(self):
        self.state['uid'] = 0
        self.command('apt-get')
        self.assert_success(self.run_launcher())
        self.assertFalse(self.calls('sudo'))

    def test_linux_package_failure_stops_before_environment_creation(self):
        self.command('apt-get')
        self.command('sudo')
        self.state['package_fail'] = True
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('package-manager error', result.stderr)
        self.assertEqual(len(self.calls('apt-get')), 1)  # failed update must not run install
        self.assertFalse(self.setup_calls())

    def test_linux_without_sudo_does_not_try_package_manager(self):
        self.command('apt-get')
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('needs sudo', result.stderr)
        self.assertFalse(self.calls('apt-get'))

    def test_linux_without_package_manager_gives_manual_instructions(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('No supported Python installer', result.stderr)
        self.assertFalse(self.setup_calls())

    def test_uv_opt_in_and_nixos_use_managed_python(self):
        self.command('uv')
        self.command('python3')
        for nixos in (False, True):
            with self.subTest(nixos=nixos):
                self.state['nixos'] = nixos
                self.env['STRATA_UV'] = '0' if nixos else '1'
                self.assertEqual(self.run_launcher().returncode, 0)
                shutil.rmtree(self.repo / '.venv')
        self.assertEqual(len(self.calls('uv')), 2)
        self.assertFalse(self.calls('python3'))

    def test_uv_is_not_selected_just_because_installed(self):
        self.command('uv')
        self.command('python3')
        self.assert_success(self.run_launcher())
        self.assertFalse(self.calls('uv'))

    def test_requested_uv_missing_does_not_silently_change_interpreter(self):
        self.command('python3')
        self.env['STRATA_UV'] = '1'
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('Install uv', result.stderr)
        self.assertFalse(self.setup_calls())

    def test_unsupported_platform_does_not_install(self):
        self.state['platform'] = 'FreeBSD'
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('supports macOS and Linux', result.stderr)
        self.assertFalse(self.setup_calls())

    def test_interactive_setup_keeps_stdin_and_pauses_on_failure(self):
        self.private_python()
        self.state.update(setup_status=37, read_input=True)
        self.prepare()
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        p = subprocess.Popen(['/bin/sh', str(self.repo / 'setup.sh')], cwd=self.root, env=self.env,
                             stdin=slave, stdout=slave, stderr=slave)
        os.close(slave)
        self.addCleanup(lambda: p.poll() is None and p.kill())
        os.write(master, b'from the terminal\n')
        output = b''
        until = time.monotonic() + 10
        while b'Press Enter to close' not in output and time.monotonic() < until:
            if select.select([master], [], [], 0.1)[0]:
                output += os.read(master, 4096)
        self.assertIn(b'INPUT=from the terminal', output)
        self.assertIn(b'Press Enter to close', output)
        self.assertIsNone(p.poll())
        os.write(master, b'\n')
        self.assertEqual(p.wait(timeout=5), 37)

    def test_noninteractive_exec_preserves_pid_and_shutdown_signal(self):
        self.private_python()
        self.state['wait'] = True
        self.prepare()
        with subprocess.Popen(['/bin/sh', str(self.repo / 'setup.sh')], cwd=self.root, env=self.env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            try:
                self.assertTrue(select.select([p.stdout], [], [], 10)[0])
                self.assertEqual(p.stdout.readline(), b'READY\n')
                self.assertEqual(self.setup_calls()[0]['pid'], p.pid)
                p.terminate()
                self.assertEqual(p.wait(timeout=5), -signal.SIGTERM)
            finally:
                if p.poll() is None: p.kill()


if __name__ == '__main__':
    unittest.main()
