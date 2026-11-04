"""GLM 5.3 Flash behind the server: its pre-tokenizer, its tool-call format, its stop tokens and thinking off.

    python -m unittest serve.test_glm
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402


class PreTokenizer(unittest.TestCase):
    TOKENS = ["1", "2", "3", "4", "5", "12", "123", "45", "a", "b", "ab"]
    MERGES = ["1 2", "12 3", "4 5"]

    def pieces(self, pre, text):
        tok = ST.Tokenizer(self.TOKENS, self.MERGES, pre=pre)
        return [self.TOKENS[i] for i in tok.encode(text)]

    def test_digits_in_runs_of_three(self):
        self.assertEqual(self.pieces("glm4", "12345"), ["123", "45"])
        self.assertEqual(self.pieces("qwen35", "12345"), ["1", "2", "3", "4", "5"])

    def test_a_whole_piece_skips_the_merges(self):
        # "ab" is a token no merge builds
        self.assertEqual(self.pieces("glm4", "ab"), ["ab"])
        self.assertEqual(self.pieces("qwen35", "ab"), ["a", "b"])


class ToolCalls(unittest.TestCase):
    SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "n": {"type": "integer"},
                                                              "content": {"type": "string"}}}}]
    CONTENT = "close a call with </tool_call>, a value with </arg_value>\nend"
    TEXT = ("thinking</think>\n\nI will write it.\n<tool_call>write\n<arg_key>path</arg_key><arg_value>doc.md"
            "</arg_value>\n<arg_key>n</arg_key><arg_value>3</arg_value><arg_key>content</arg_key><arg_value>"
            + CONTENT + "</arg_value></tool_call>")

    def events(self, stream_tools, step):
        p = OutputParser(thinking=True, tools=self.SCHEMA, stream_tools=stream_tools, call_format="glm", opened=True)
        evs = []
        for i in range(0, len(self.TEXT), step):
            evs += p.feed(self.TEXT[i:i + step])
        return evs + p.finish()

    def test_call_is_parsed_however_it_arrives(self):
        for stream_tools in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = self.events(stream_tools, step)
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].name, "write")
                    self.assertEqual(calls[0].arguments, {"path": "doc.md", "n": 3, "content": self.CONTENT})
                    self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "thinking")
                    self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "I will write it.")

    def test_streamed_call_is_announced_first(self):
        kinds = [e.kind for e in self.events(True, 1) if e.kind.startswith("tool_")]
        self.assertEqual(kinds, ["tool_start", "tool_args", "tool_call"])

    def test_call_without_arguments(self):
        p = OutputParser(thinking=False, call_format="glm")
        evs = p.feed("<tool_call>now</tool_call>") + p.finish()
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], [("now", {})])


class Reasoning(unittest.TestCase):
    def test_an_answer_right_after_a_quote_is_content(self):
        # GLM writes its answer right after </think>, also when the thinking ends on a quote
        text = 'so I should just say "Paris."</think>Paris'
        for size in (1, 5, len(text)):
            with self.subTest(size=size):
                p = OutputParser(thinking=True, call_format="glm", opened=True)   # the prompt ended on <think>
                evs = [e for i in range(0, len(text), size) for e in p.feed(text[i:i + size])] + p.finish()
                self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), 'so I should just say "Paris."')
                self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Paris")


class GlmTokens(ByteTokenizer):
    SPECIALS = ["<|endoftext|>", "<|user|>", "<|observation|>", "<|assistant|>"]
    pre = "glm4"
    special_ids = {"tokenizer.ggml.eos_token_id": 256, "tokenizer.ggml.eot_token_id": 257,
                   "tokenizer.ggml.eom_token_id": 258, "tokenizer.ggml.bos_token_id": 259}


class ServiceForGlm(unittest.TestCase):
    def service(self, tok, template):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text(template, encoding="utf-8")
            return Service(MockEngine(tok, "hi"), tok, ChatTemplate(path))

    def test_stop_ids_and_call_format(self):
        svc = self.service(GlmTokens(), "<|assistant|><think>")
        self.assertEqual(svc.stop_ids, {256, 257, 258})
        self.assertEqual(svc.call_format, "glm")
        chatml = self.service(ByteTokenizer(), "x")
        self.assertEqual(chatml.stop_ids, {257, 258})       # <|im_end|>, <|endoftext|>
        self.assertEqual(chatml.call_format, "function")

    def test_the_turn_ends_when_the_gguf_names_only_eos(self):
        # #70: REAP50-Q3_K_M names <|endoftext|> and no eot or eom; the reply still ends at <|user|> and a call at
        # <|observation|>
        class EosOnly(GlmTokens):
            special_ids = {"tokenizer.ggml.eos_token_id": 256}
        self.assertEqual(self.service(EosOnly(), "<|assistant|>").stop_ids, {256, 257, 258})

    def test_thinking_off_closes_the_open_block(self):
        svc = self.service(GlmTokens(), "{{ messages[0].content }}<|assistant|><think>")
        msgs = [{"role": "user", "content": "q"}]
        self.assertEqual(svc.render_prompt(msgs, None, {}), "q<|assistant|><think>")
        self.assertEqual(svc.render_prompt(msgs, None, {"enable_thinking": False}), "q<|assistant|><think></think>")

    def test_the_reply_goes_on_inside_the_thinking_the_prompt_opened(self):
        # GLM's prompt ends on <think> and Flint's does not (serve/test_flint.py): the prompt's last tokens tell them
        tok = GlmTokens()
        svc = self.service(tok, "{{ messages[0].content }}<|assistant|><think>")
        for kwargs, reply, reasoning in (({}, "say Paris</think>Paris<|user|>", "say Paris"),
                                         ({"enable_thinking": False}, "Paris<|user|>", "")):
            with self.subTest(**kwargs):
                ids = svc.encode_prompt([{"role": "user", "content": "q"}], None, kwargs)
                svc.engine = MockEngine(tok, reply)
                evs = [e for kind, e in svc.run(ids, not kwargs, None, 64, {}, threading.Event()) if kind == "event"]
                self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), reasoning)
                self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Paris")


if __name__ == "__main__":
    unittest.main()
