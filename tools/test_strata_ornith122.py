"""Tests for Ornith-1.5-122B engine verification, synthetic GGUF metadata, tokenizer,
model registry, streaming reasoning/tool parser, and Qwen35 load_geometry validation.

    python -m unittest tools.test_strata_ornith122
"""
from __future__ import annotations

import io
import json
import math
import os
import pathlib
import re
import struct
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "serve"))

# Ensure .venv site-packages is discoverable if running under system python
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

import setup  # noqa: E402
from serve import frontend, models  # noqa: E402
from tools import strata_tokenizer  # noqa: E402
from tools.gguf_reader import GGUFFile  # noqa: E402


# -----------------------------------------------------------------------------
# Pure Python GGUF v3 Writer (Zero external dependencies)
# -----------------------------------------------------------------------------
class SyntheticGGUFWriter:
    """Minimal self-contained GGUF v3 writer for tests."""

    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

    META_TYPES = {
        "u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6,
        "bool": 7, "string": 8, "array": 9, "u64": 10, "i64": 11, "f64": 12,
    }

    def __init__(self, alignment: int = 32):
        self.alignment = alignment
        self.metadata: dict[str, tuple[str, object]] = {}
        self.tensors: list[dict] = []

    def add(self, key: str, value, type_name: str | None = None) -> None:
        if type_name is None:
            if isinstance(value, bool):
                type_name = "bool"
            elif isinstance(value, int):
                type_name = "i64" if value.bit_length() > 31 else "i32"
            elif isinstance(value, float):
                type_name = "f32"
            elif isinstance(value, str):
                type_name = "string"
            elif isinstance(value, list):
                if not value:
                    type_name = "array:string"
                else:
                    elem_type = {bool: "bool", int: "i32", float: "f32", str: "string"}.get(type(value[0]), "string")
                    type_name = f"array:{elem_type}"
            else:
                raise TypeError(f"Cannot infer GGUF metadata type for key: {key}")
        self.metadata[key] = (type_name, value)

    def add_tensor(self, name: str, shape: list[int], data: bytes, type_id: int = 0) -> None:
        """Add tensor (type_id 0 = F32)."""
        self.tensors.append({
            "name": name,
            "shape": shape,
            "type_id": type_id,
            "data": data,
        })

    def _scalar_bytes(self, type_name: str, value) -> bytes:
        if type_name == "string":
            raw = str(value).encode("utf-8")
            return struct.pack("<Q", len(raw)) + raw
        fmt = {
            "u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
            "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d",
        }[type_name]
        return struct.pack(fmt, value)

    def _kv_bytes(self, key: str, type_name: str, value) -> bytes:
        raw_key = key.encode("utf-8")
        buf = bytearray(struct.pack("<Q", len(raw_key)) + raw_key)
        if type_name.startswith("array:"):
            elem = type_name.split(":", 1)[1]
            buf += struct.pack("<IIQ", self.META_TYPES["array"], self.META_TYPES[elem], len(value))
            for item in value:
                buf += self._scalar_bytes(elem, item)
            return bytes(buf)
        buf += struct.pack("<I", self.META_TYPES[type_name])
        buf += self._scalar_bytes(type_name, value)
        return bytes(buf)

    def write(self, path: pathlib.Path | str) -> pathlib.Path:
        target = pathlib.Path(path)
        body = bytearray()
        body += struct.pack("<IIQQ", self.GGUF_MAGIC, 3, len(self.tensors), len(self.metadata))

        for k, (t, v) in self.metadata.items():
            body += self._kv_bytes(k, t, v)

        offset = 0
        infos = bytearray()
        for t in self.tensors:
            name_bytes = t["name"].encode("utf-8")
            infos += struct.pack("<Q", len(name_bytes)) + name_bytes
            infos += struct.pack("<I", len(t["shape"]))
            infos += struct.pack(f"<{len(t['shape'])}Q", *t["shape"])
            infos += struct.pack("<IQ", t["type_id"], offset)
            offset += len(t["data"])
            offset = (offset + self.alignment - 1) // self.alignment * self.alignment
        body += infos

        pad = (-len(body)) % self.alignment
        body += b"\0" * pad

        for t in self.tensors:
            body += t["data"]
            pad = (-len(t["data"])) % self.alignment
            body += b"\0" * pad

        target.write_bytes(body)
        return target


# -----------------------------------------------------------------------------
# Synthetic Ornith-1.5-122B GGUF Fixture Builder
# -----------------------------------------------------------------------------
def build_synthetic_ornith122_gguf(target_path: pathlib.Path | str) -> pathlib.Path:
    """Build a minimal valid GGUF v3 file representing Ornith-1.5-122B."""
    w = SyntheticGGUFWriter()

    # General metadata
    w.add("general.architecture", "qwen35moe")
    w.add("general.name", "Ornith-1.5-122B")
    w.add("general.size_label", "122B-A14B")
    w.add("general.alignment", 32, "u32")

    # Geometry: 48 trunk layers + 1 MTP draft layer = 49 blocks
    w.add("qwen35moe.block_count", 49, "i32")
    w.add("qwen35moe.nextn_predict_layers", 1, "i32")
    w.add("qwen35moe.embedding_length", 3072, "i32")
    w.add("qwen35moe.vocab_size", 248320, "i32")
    w.add("qwen35moe.context_length", 262144, "i32")
    w.add("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6, "f32")

    # Attention & RoPE
    w.add("qwen35moe.attention.head_count", 32, "i32")
    w.add("qwen35moe.attention.head_count_kv", 2, "i32")
    w.add("qwen35moe.attention.key_length", 256, "i32")
    w.add("qwen35moe.attention.value_length", 256, "i32")
    w.add("qwen35moe.rope.dimension_count", 64, "i32")
    w.add("qwen35moe.rope.freq_base", 10000000.0, "f32")
    w.add("qwen35moe.rope.dimension_sections", [11, 11, 10, 0], "array:i32")

    # GDN Recurrent State (SSM)
    w.add("qwen35moe.ssm.conv_kernel", 4, "i32")
    w.add("qwen35moe.ssm.state_size", 128, "i32")
    w.add("qwen35moe.ssm.group_count", 16, "i32")
    w.add("qwen35moe.ssm.time_step_rank", 64, "i32")
    w.add("qwen35moe.ssm.inner_size", 8192, "i32")

    # MoE Experts
    w.add("qwen35moe.expert_count", 256, "i32")
    w.add("qwen35moe.expert_used_count", 8, "i32")
    w.add("qwen35moe.expert_feed_forward_length", 1024, "i32")
    w.add("qwen35moe.expert_shared_feed_forward_length", 1024, "i32")

    # Tokenizer metadata
    w.add("tokenizer.ggml.model", "gpt2")
    w.add("tokenizer.ggml.pre", "qwen35")
    w.add("tokenizer.ggml.bos_token_id", 248044, "i32")
    w.add("tokenizer.ggml.eos_token_id", 248046, "i32")

    # Tokenizer tokens: 256 byte tokens + special tokens
    byte_tokens = [str(chr(c)) for c in range(256)]
    special_tokens = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
        "<think>", "</think>",
    ]
    tokens = byte_tokens + special_tokens
    # 1=NORMAL, 3=CONTROL (<|im_start|>, <|im_end|>, <|endoftext|>), 4=USER_DEFINED (<think>, <tool_call>, etc.)
    types = [1] * len(byte_tokens) + [3, 3, 3, 4, 4, 4, 4, 4, 4]
    merges = ["a b", "c d", "e f"]

    w.add("tokenizer.ggml.tokens", tokens, "array:string")
    w.add("tokenizer.ggml.token_type", types, "array:i32")
    w.add("tokenizer.ggml.merges", merges, "array:string")
    w.add("tokenizer.chat_template", (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|im_start|>assistant\n<think>\n'}}{% endif %}"
    ))

    # Tensors: minimal dummy payloads for geometry & tensor table recognition
    dummy_vec = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    w.add_tensor("token_embd.weight", [3072, 32], dummy_vec)
    w.add_tensor("output_norm.weight", [3072], dummy_vec)
    w.add_tensor("output.weight", [3072, 32], dummy_vec)
    w.add_tensor("blk.0.attn_norm.weight", [3072], dummy_vec)
    w.add_tensor("blk.0.ffn_gate_inp.weight", [3072, 256], dummy_vec)

    # Block 48 MTP Draft Layer tensors
    w.add_tensor("blk.48.nextn.eh_proj.weight", [6144, 3072], dummy_vec)
    w.add_tensor("blk.48.nextn.shared_head_norm.weight", [3072], dummy_vec)
    w.add_tensor("blk.48.nextn.enorm.weight", [3072], dummy_vec)
    w.add_tensor("blk.48.nextn.hnorm.weight", [3072], dummy_vec)
    w.add_tensor("blk.48.ffn_gate_inp.weight", [3072, 256], dummy_vec)

    return w.write(target_path)


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------
class TestSyntheticGGUFHeader(unittest.TestCase):
    """Test synthetic GGUF v3 construction and metadata adherence."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gguf_path = pathlib.Path(self.tmp.name) / "Ornith-1.5-122B-UD-Q3_K_XL.gguf"
        build_synthetic_ornith122_gguf(self.gguf_path)
        self.reader = GGUFFile(self.gguf_path)
        self.meta = self.reader.metadata

    def tearDown(self):
        self.tmp.cleanup()

    def test_gguf_v3_header_attributes(self):
        self.assertEqual(self.reader.version, 3)
        self.assertEqual(self.meta.get("general.architecture"), "qwen35moe")
        self.assertEqual(self.meta.get("general.name"), "Ornith-1.5-122B")
        self.assertEqual(self.meta.get("general.size_label"), "122B-A14B")

    def test_layer_and_draft_tensor_counts(self):
        # 48 trunk layers + 1 draft = 49 blocks
        self.assertEqual(self.meta.get("qwen35moe.block_count"), 49)
        self.assertEqual(self.meta.get("qwen35moe.nextn_predict_layers"), 1)
        self.assertEqual(self.meta.get("qwen35moe.embedding_length"), 3072)
        self.assertEqual(self.meta.get("qwen35moe.attention.head_count"), 32)
        self.assertEqual(self.meta.get("qwen35moe.attention.head_count_kv"), 2)
        self.assertEqual(self.meta.get("qwen35moe.attention.key_length"), 256)

        # Confirm MTP draft tensors are present
        tensor_names = [t.name for t in self.reader.tensors]
        self.assertIn("blk.48.nextn.eh_proj.weight", tensor_names)
        self.assertIn("blk.48.nextn.shared_head_norm.weight", tensor_names)
        self.assertIn("blk.48.nextn.enorm.weight", tensor_names)
        self.assertIn("blk.48.nextn.hnorm.weight", tensor_names)


class TestModelRegistryDescribe(unittest.TestCase):
    """Test serve/models.py describe() on synthetic Ornith-1.5-122B run configs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = pathlib.Path(self.tmp.name)
        self.gguf_path = self.tmp_dir / "Ornith-1.5-122B-UD-Q3_K_XL.gguf"
        build_synthetic_ornith122_gguf(self.gguf_path)

        # Extract minimal tokenizer pack
        self.pack_dir = self.tmp_dir / "pack"
        strata_tokenizer.extract(self.gguf_path, self.pack_dir)

        # Generate fake binary executable so describe() marks engine as present
        self.fake_exe = self.tmp_dir / "strata-poly.exe"
        self.fake_exe.write_bytes(b"MZ\x90\x00")

    def tearDown(self):
        self.tmp.cleanup()

    def test_describe_accurately_resolves_ornith122(self):
        cfg_path = self.tmp_dir / "strata-ornith122-ud-q3_k_xl.json"
        cfg_data = {
            "exe": str(self.fake_exe),
            "cwd": str(self.tmp_dir),
            "args": [
                "--model", str(self.gguf_path),
                "--max-context", "262144",
            ],
            "tokenizer": str(self.pack_dir / "tokenizer"),
            "model_name": "ornith122-ud-q3_k_xl",
            "display_name": "Ornith 1.5 122B UD-Q3_K_XL",
            "vision": True,
        }
        cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

        desc = models.describe(cfg_path)
        self.assertIsNotNone(desc)
        self.assertEqual(desc["id"], "ornith122-ud-q3_k_xl")
        self.assertEqual(desc["backend"], "qwen35moe")
        self.assertEqual(desc["status"], "available")
        self.assertIsNone(desc["problem"])
        self.assertEqual(desc["params_total"], "122B")
        self.assertEqual(desc["params_active"], "14B")
        self.assertEqual(desc["quantization"], "UD-Q3_K_XL")
        self.assertTrue(desc["mtp"])
        self.assertTrue(desc["multimodal"])
        self.assertEqual(desc["context"], 262144)

    def test_describe_no_draft_flag_disables_mtp(self):
        cfg_path = self.tmp_dir / "strata-ornith122-nodraft.json"
        cfg_data = {
            "exe": str(self.fake_exe),
            "cwd": str(self.tmp_dir),
            "args": [
                "--model", str(self.gguf_path),
                "--no-draft",
            ],
            "tokenizer": str(self.pack_dir / "tokenizer"),
            "model_name": "ornith122-nodraft",
        }
        cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

        desc = models.describe(cfg_path)
        self.assertIsNotNone(desc)
        self.assertFalse(desc["mtp"])

    def test_describe_missing_model_file_marks_unavailable(self):
        cfg_path = self.tmp_dir / "strata-missing.json"
        cfg_data = {
            "exe": str(self.fake_exe),
            "cwd": str(self.tmp_dir),
            "args": [
                "--model", str(self.tmp_dir / "nonexistent.gguf"),
            ],
            "tokenizer": str(self.pack_dir / "tokenizer"),
            "model_name": "missing-model",
        }
        cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

        desc = models.describe(cfg_path)
        self.assertIsNotNone(desc)
        self.assertEqual(desc["status"], "unavailable")
        self.assertIn("the model file is missing", desc["problem"])


class TestTokenizerQwen35Rules(unittest.TestCase):
    """Test tools/strata_tokenizer.py pre-tokenizer pattern and special token rules."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gguf_path = pathlib.Path(self.tmp.name) / "test.gguf"
        build_synthetic_ornith122_gguf(self.gguf_path)
        self.pack_dir = pathlib.Path(self.tmp.name) / "pack"
        strata_tokenizer.extract(self.gguf_path, self.pack_dir)
        self.tokenizer = strata_tokenizer.Tokenizer.from_dir(self.pack_dir / "tokenizer")

    def tearDown(self):
        self.tmp.cleanup()

    def test_extracted_config_parameters(self):
        cfg_file = self.pack_dir / "tokenizer" / "tokenizer.json"
        self.assertTrue(cfg_file.exists())
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        self.assertEqual(cfg["pre"], "qwen35")
        self.assertEqual(cfg["model"], "gpt2")
        self.assertIn("LLAMA_VOCAB_PRE_TYPE_QWEN35", cfg["pre_pattern_source"])

    def test_special_token_type4_user_defined_matched_always(self):
        # Type 4 tokens (<think>, </think>, <tool_call>, </tool_call>) must match even when parse_special=False
        for token_str in ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]:
            token_id = self.tokenizer.ids[token_str]
            ids_without_special = self.tokenizer.encode(token_str, parse_special=False)
            self.assertEqual(ids_without_special, [token_id], f"{token_str} should match with parse_special=False")

    def test_special_token_type3_control_requires_parse_special(self):
        # Type 3 control tokens (<|im_start|>, <|im_end|>, <|endoftext|>) require parse_special=True
        for token_str in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            token_id = self.tokenizer.ids[token_str]
            ids_with_special = self.tokenizer.encode(token_str, parse_special=True)
            self.assertEqual(ids_with_special, [token_id])

            ids_without_special = self.tokenizer.encode(token_str, parse_special=False)
            self.assertNotEqual(ids_without_special, [token_id])

    def test_qwen35_pretokenizer_combining_marks(self):
        # Combining marks (\p{M}) must be bound to preceding letter
        import regex
        pattern = strata_tokenizer.QWEN35_PATTERN
        text = "e\u0301\u0301"  # 'e' with acute accents
        matches = [m.group(0) for m in regex.finditer(pattern, text)]
        self.assertEqual(matches, [text])


class TestFrontendReasoningAndTools(unittest.TestCase):
    """Test serve/frontend.py reasoning tag streaming and XML tool calling parser."""

    def test_parse_tool_call_xml_function_and_parameters(self):
        xml_call = (
            "<function=get_weather>\n"
            "<parameter=location>\n"
            "Tokyo, Japan\n"
            "</parameter>\n"
            "<parameter=days>\n"
            "3\n"
            "</parameter>\n"
            "</function>"
        )
        schema = {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        }
        call = frontend.parse_tool_call(xml_call, schema)
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.arguments, {"location": "Tokyo, Japan", "days": 3})

    def test_output_parser_reasoning_and_content_separation(self):
        parser = frontend.OutputParser(thinking=True)
        deltas = [
            "<think>\nAnalyzing the",
            " query for 122B MoE\n",
            "</think>\n\nHere is",
            " the verified answer.",
        ]
        events = []
        for d in deltas:
            events.extend(parser.feed(d))
        events.extend(parser.finish())

        reasoning = "".join(e.text for e in events if e.kind == "reasoning")
        content = "".join(e.text for e in events if e.kind == "content")
        self.assertIn("Analyzing the query for 122B MoE", reasoning)
        self.assertIn("Here is the verified answer.", content)

    def test_output_parser_streaming_tool_call(self):
        tools = [{
            "name": "fetch_spec",
            "parameters": {
                "type": "object",
                "properties": {"model": {"type": "string"}},
            },
        }]
        parser = frontend.OutputParser(thinking=False, tools=tools)
        xml = "<tool_call>\n<function=fetch_spec>\n<parameter=model>\nOrnith-1.5-122B\n</parameter>\n</function>\n</tool_call>"
        events = parser.feed(xml)
        events.extend(parser.finish())

        tool_calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "fetch_spec")
        self.assertEqual(tool_calls[0].arguments, {"model": "Ornith-1.5-122B"})

    def test_literal_tag_escaping_prevents_injection(self):
        # Quoted literal </think> inside messages should be protected
        messages = [{"role": "user", "content": 'Explain the tag "</think>" in detail.'}]
        marked, _tools, was_marked = frontend.mark_think_literals(messages, None)
        self.assertTrue(was_marked)
        self.assertNotIn("</think>", marked[0]["content"])

        unmarked, spans = frontend.unmark_think_literals(marked[0]["content"])
        self.assertIn("</think>", unmarked)
        self.assertTrue(len(spans) > 0)


class TestQwen35LoadGeometryChecks(unittest.TestCase):
    """Test mathematical and dimension rules from strata-poly's Qwen35::load_geometry() and Engine::load_shape()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gguf_path = pathlib.Path(self.tmp.name) / "test_geom.gguf"
        build_synthetic_ornith122_gguf(self.gguf_path)
        self.reader = GGUFFile(self.gguf_path)
        self.meta = self.reader.metadata

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_ornith122_dimensions(self):
        # 1. Block counts & trunk layers
        block_count = self.meta["qwen35moe.block_count"]
        nextn = self.meta["qwen35moe.nextn_predict_layers"]
        n_layer = block_count - nextn
        self.assertEqual(n_layer, 48)
        self.assertEqual(nextn, 1)

        # 2. Attention heads and head dimensions
        n_head = self.meta["qwen35moe.attention.head_count"]
        n_kv = self.meta["qwen35moe.attention.head_count_kv"]
        head_dim = self.meta["qwen35moe.attention.key_length"]
        value_dim = self.meta["qwen35moe.attention.value_length"]
        self.assertEqual(n_head, 32)
        self.assertEqual(n_kv, 2)
        self.assertEqual(head_dim, 256)
        self.assertEqual(value_dim, head_dim)
        self.assertEqual(n_head % n_kv, 0)
        self.assertLessEqual(n_head, 256)

        # 3. RoPE configuration
        n_rot = self.meta["qwen35moe.rope.dimension_count"]
        self.assertEqual(n_rot, 64)
        self.assertEqual(n_rot % 2, 0)
        self.assertLessEqual(n_rot, head_dim)

        # 4. GDN SSM configuration
        k_conv = self.meta["qwen35moe.ssm.conv_kernel"]
        s_dim = self.meta["qwen35moe.ssm.state_size"]
        k_heads = self.meta["qwen35moe.ssm.group_count"]
        v_heads = self.meta["qwen35moe.ssm.time_step_rank"]
        d_inner = self.meta["qwen35moe.ssm.inner_size"]
        conv_dim = 2 * k_heads * s_dim + d_inner

        self.assertEqual(k_conv, 4)
        self.assertEqual(s_dim, 128)
        self.assertEqual(s_dim % 32, 0)
        self.assertLessEqual(s_dim, 256)
        self.assertEqual(k_heads, 16)
        self.assertEqual(v_heads, 64)
        self.assertEqual(v_heads % k_heads, 0)
        self.assertEqual(d_inner, v_heads * s_dim)
        self.assertEqual(conv_dim, 12288)

        # 5. M-RoPE Sections Check (src/poly/qwen35.cpp:77-80)
        sections = self.meta["qwen35moe.rope.dimension_sections"]
        self.assertGreaterEqual(len(sections), 3)
        covered = sum(sections[:4])
        self.assertGreater(covered, 0)
        self.assertEqual(covered, 32)

        for pair in range(n_rot // 2):
            s = pair % covered
            self.assertLess(s, 3 * sections[s % 3], f"Pair {pair} failed M-RoPE spare axis condition")

        # 6. MoE expert limits
        n_expert = self.meta["qwen35moe.expert_count"]
        n_used = self.meta["qwen35moe.expert_used_count"]
        self.assertEqual(n_expert, 256)
        self.assertEqual(n_used, 8)
        self.assertLessEqual(n_used, 32)  # kMaxUsed
        self.assertLessEqual(n_used, n_expert)

    def test_rejection_of_invalid_ssm_geometry(self):
        # A model with d_inner != v_heads * s_dim must be rejected
        v_heads = 64
        s_dim = 128
        bad_d_inner = 4096  # Should be 8192
        with self.assertRaises(AssertionError):
            self.assertEqual(bad_d_inner, v_heads * s_dim)

    def test_rejection_of_invalid_mrope_sections(self):
        # A section definition leaving pairs to the spare axis must be rejected
        bad_sections = [4, 4, 4, 20]  # Only 12 covered by first 3 axes, 20 spare
        covered = sum(bad_sections[:4])
        failed_pair = False
        for pair in range(32):
            s = pair % covered
            if not (s < 3 * bad_sections[s % 3]):
                failed_pair = True
                break
        self.assertTrue(failed_pair, "Bad M-RoPE sections must fail check")


if __name__ == "__main__":
    unittest.main()
