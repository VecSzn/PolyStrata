"""Mac poly controls and GGUF-derived memory estimates; no GPU/network/model payloads."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import metal_poly_setup as P
from tools import test_setup_mac as H

M = H.M


def args(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--family')
    P.add_arguments(parser)
    return parser.parse_args(argv)


def header(draft=True):
    meta = {'general.architecture': 'qwen35moe', 'qwen35moe.block_count': 41 if draft else 40,
            'qwen35moe.nextn_predict_layers': int(draft), 'qwen35moe.attention.head_count_kv': 2,
            'qwen35moe.attention.key_length': 256, 'qwen35moe.attention.value_length': 256, 'qwen35moe.ssm.state_size': 128,
            'qwen35moe.ssm.group_count': 16, 'qwen35moe.ssm.time_step_rank': 32,
            'qwen35moe.ssm.inner_size': 4096}
    names = [f'blk.{i}.attn_k.weight' if i % 4 == 3 or i == 40 else f'blk.{i}.attn_qkv.weight'
             for i in range(41 if draft else 40)]
    if draft:
        names.append('blk.40.nextn.eh_proj.weight')
    return SimpleNamespace(metadata=meta, tensors=[SimpleNamespace(name=n) for n in names])


class Options(unittest.TestCase):
    def test_defaults_enable_optimization_and_balanced_memory(self):
        a = args('--family', 'tiel')
        self.assertFalse(P.requested(a))
        command = ['--model', 'existing.gguf']
        P.apply_options(command, a, {})
        self.assertEqual(command, ['--model', 'existing.gguf', '--metal-optimize', '--metal-memory', 'balanced'])

    def test_explicit_flags_and_scoped_preservation(self):
        old = {'args': ['--metal-optimize', '--metal-memory', 'balanced', '--kv', 'q8_0', '--custom', 'x']}
        command = []
        P.apply_options(command, args('--family', 'tiel'), old)
        self.assertEqual(command, old['args'][:-2])
        command = []
        P.apply_options(command, args('--family', 'tiel', '--metal-memory', 'legacy', '--poly-kv', 'fp16'), old)
        self.assertEqual(command, ['--metal-optimize', '--metal-memory', 'legacy', '--kv', 'fp16'])
        command = []
        P.apply_options(command, args('--family', 'tiel', '--metal-optimize'), {})
        self.assertEqual(command, ['--metal-optimize', '--metal-memory', 'balanced'])

    def test_saved_opt_outs_and_explicit_overrides(self):
        old = {'args': ['--no-metal-optimize', '--metal-memory', 'legacy', '--kv', 'fp16']}
        command = []
        P.apply_options(command, args('--family', 'tiel'), old)
        self.assertEqual(command, old['args'])
        command = []
        P.apply_options(command, args('--family', 'tiel', '--metal-optimize', '--metal-memory', 'balanced',
                                      '--poly-kv', 'q8_0'), old)
        self.assertEqual(command, ['--metal-optimize', '--metal-memory', 'balanced', '--kv', 'q8_0'])
        command = []
        P.apply_options(command, args('--family', 'tiel', '--no-metal-optimize'), {'args': ['--metal-optimize']})
        self.assertEqual(command, ['--no-metal-optimize', '--metal-memory', 'balanced'])
        for argv, enabled in ((['--metal-optimize', '--no-metal-optimize'], False),
                              (['--no-metal-optimize', '--metal-optimize'], True)):
            self.assertEqual(args('--family', 'tiel', *argv).metal_optimize, enabled)
            command = []
            P.apply_options(command, args('--family', 'tiel'), {'args': argv})
            self.assertEqual(command[0], '--metal-optimize' if enabled else '--no-metal-optimize')

    def test_non_poly_and_unsupported_q8_rejected_before_setup(self):
        for argv in (['--family', 'qwen', '--metal-optimize'],
                     ['--family', 'qwen', '--no-metal-optimize'],
                     ['--family', 'lfm25', '--poly-kv', 'q8_0'],
                     ['--metal-memory', 'balanced']):
            with self.subTest(argv=argv), mock.patch.object(M.S, 'data_folder', side_effect=AssertionError('side effect')):
                with self.assertRaises(SystemExit) as ex:
                    M.main(argv)
                self.assertEqual(ex.exception.code, 2)
        for family in P.Q8_FAMILIES:
            P.validate(args('--family', family, '--poly-kv', 'q8_0'), M.MAC_POLY_FAMILIES)
        P.validate(args('--family', 'lfm25', '--poly-kv', 'fp16'), M.MAC_POLY_FAMILIES)

    def test_broken_existing_config_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'cfg.json'
            self.assertEqual(P.read_config(p), {})
            p.write_text('[]')
            self.assertEqual(P.read_config(p), {})
            p.write_text(json.dumps({'args': ['--metal-optimize']}))
            self.assertEqual(P.read_config(p)['args'], ['--metal-optimize'])

    def test_actual_setup_defaults_and_estimate_policy(self):
        captured = []
        original_apply = P.apply_options
        def apply(command, a, old):
            original_apply(command, a, {})
            captured.append(command.copy())
        with mock.patch.object(P, 'apply_options', side_effect=apply), \
                mock.patch.object(P, 'read_headers', return_value=([header()], None)), \
                mock.patch.object(P, 'estimate', wraps=P.estimate) as estimate:
            H.SmallerModelSetup('test_existing_files_keep_shared_server_features').test_existing_files_keep_shared_server_features()
        self.assertTrue(captured)
        for command in captured:
            self.assertIn('--metal-optimize', command)
            self.assertEqual(P.flag_value(command, '--metal-memory'), 'balanced')
            self.assertIsNone(P.flag_value(command, '--kv'))
        self.assertTrue(estimate.call_args_list)
        for call in estimate.call_args_list:
            self.assertEqual(call.kwargs['memory'], 'balanced')
            self.assertEqual(call.kwargs['kv'], 'auto')

    def test_actual_setup_forwards_options(self):
        # Reuse the no-download installer harness over every poly family, with legal options.
        original = M.main
        captured = []
        def run(argv):
            family = argv[argv.index('--family') + 1]
            kv = 'q8_0' if family in P.Q8_FAMILIES else 'fp16'
            extra = ['--metal-optimize', '--metal-memory', 'balanced', '--poly-kv', kv]
            return original(argv + extra)
        original_apply = P.apply_options
        def apply(command, a, old):
            original_apply(command, a, old)
            captured.append(command.copy())
        with mock.patch.object(M, 'main', side_effect=run), mock.patch.object(P, 'apply_options', side_effect=apply), \
                mock.patch.object(P, 'read_headers', return_value=([header()], None)):
            H.SmallerModelSetup('test_existing_files_keep_shared_server_features').test_existing_files_keep_shared_server_features()
        self.assertTrue(captured)
        for command in captured:
            self.assertIn('--metal-optimize', command)
            self.assertEqual(P.flag_value(command, '--metal-memory'), 'balanced')
            self.assertIn(P.flag_value(command, '--kv'), ('fp16', 'q8_0'))
        self.assertTrue(any(P.flag_value(command, '--kv') == 'q8_0' for command in captured))


class Estimates(unittest.TestCase):
    def test_auto_format_uses_geometry_and_unknown_stays_unknown(self):
        h = header()
        auto = P.estimate([h], 256000, kv='auto', memory='balanced')
        explicit = P.estimate([h], 256000, kv='q8_0', memory='balanced')
        self.assertEqual(auto, explicit)
        h.metadata['qwen35moe.attention.key_length'] = 48
        h.metadata['qwen35moe.attention.value_length'] = 48
        self.assertEqual(P.estimate([h], 8192, kv='auto')['kv_format'], 'fp16')
        del h.metadata['qwen35moe.attention.key_length']
        unknown = P.estimate([h], 8192, kv='auto')
        self.assertIsNone(unknown['kv_max_bytes'])
        self.assertNotIn('kv_format', unknown)
        self.assertIsNone(P.estimate([], 8192, kv='auto')['kv_initial_bytes'])

    def test_tiel_256k_includes_draft_and_balanced_initial(self):
        legacy = P.estimate([header()], 256000)
        self.assertEqual(legacy['kv_max_bytes'], 5767168000)
        self.assertEqual(legacy['kv_initial_bytes'], legacy['kv_max_bytes'])
        self.assertEqual(legacy['attention_layers'], 11)
        self.assertTrue(legacy['draft_enabled'])
        balanced = P.estimate([header()], 256000, memory='balanced')
        self.assertEqual(balanced['kv_initial_bytes'], 11 * 2048 * 4096)
        self.assertEqual(balanced['kv_max_bytes'], legacy['kv_max_bytes'])
        self.assertEqual(P.estimate([header()], 1024, memory='balanced')['kv_initial_bytes'], 11 * 2048 * 1024)

    def test_q8_and_parallel_base_slot(self):
        estimate = P.estimate([header()], 8192, kv='q8_0', parallel=3)
        self.assertEqual(estimate['sequence_count'], 4)
        self.assertEqual(estimate['kv_max_bytes'], 11 * (2 * 2 * (256 + 8 * 2)) * 8192 * 4)
        self.assertEqual(P.estimate([header()], 8192, batch=2)['sequence_count'], 3)
        self.assertEqual(P.estimate([header()], 8192, parallel=1)['sequence_count'], 1)

    def test_disabled_missing_and_separate_draft(self):
        self.assertEqual(P.estimate([header()], 8192, draft=False)['attention_layers'], 10)
        plain = header(False)
        self.assertFalse(P.estimate([plain], 8192)['draft_enabled'])
        draft = header()
        draft.tensors = [t for t in draft.tensors if t.name.startswith('blk.40.')]
        self.assertEqual(P.estimate([plain, draft], 8192)['attention_layers'], 11)

    def test_recurrent_state_is_separate_from_unknown_working_memory(self):
        e = P.estimate([header()], 8192, parallel=2)
        self.assertEqual(e['recurrent_bytes'], 30 * (3 * (2 * 16 * 128 + 4096) + 32 * 128 * 128) * 4 * 3)
        self.assertIsNone(e['working_bytes'])
        h = header()
        del h.metadata['qwen35moe.ssm.state_size']
        self.assertIsNone(P.estimate([h], 8192)['recurrent_bytes'])

    def test_missing_geometry_and_unsupported_architecture_are_unknown(self):
        for h in (SimpleNamespace(metadata={'general.architecture': 'gemma4'}, tensors=[]),
                  SimpleNamespace(metadata={'general.architecture': 'qwen35moe'}, tensors=[])):
            self.assertIsNone(P.estimate([h], 8192)['kv_max_bytes'])
        h = header()
        h.tensors.pop(0)
        self.assertIsNone(P.estimate([h], 8192)['kv_max_bytes'])
        self.assertIsNone(P.estimate([], 8192)['kv_max_bytes'])
        self.assertIsNone(P.estimate([header()], 8192, parallel='invalid')['kv_max_bytes'])
        lines = []
        P.report(P.estimate([], 8192), lines.append)
        self.assertIn('initially unknown', lines[0])
        self.assertIn('not a total memory requirement', lines[-1])

    def test_bad_gguf_is_unknown_without_download(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'tiny.gguf'
            p.write_bytes(b'bad')
            headers, why = P.read_headers([p])
            self.assertEqual(headers, [])
            self.assertIn('unavailable', why)


if __name__ == '__main__':
    unittest.main()
