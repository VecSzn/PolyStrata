"""Flint behind the server: its pre-tokenizer, and the GLM tool calls, turns and stops it is served with.

    python -m unittest serve.test_flint
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402


class PreTokenizer(unittest.TestCase):
    def pieces(self, text):
        return ST.Tokenizer(["a"], [], pre="flint")._pieces(text)

    def test_digits_then_ideographs_then_words_cut_where_capitals_end(self):
        self.assertEqual(self.pieces("1234567 dollars"), ["123", "456", "7", " dollars"])
        self.assertEqual(self.pieces("你好ab"), ["你好", "ab"])
        self.assertEqual(self.pieces("HTTPServer and McDonald's"), ["HTTPServer", " and", " Mc", "Donald's"])
        self.assertEqual(self.pieces("they've  said.\n\n  x"), ["they've", " ", " said", ".\n\n", " ", " x"])


class Reasoning(unittest.TestCase):
    TEXT = "<think>the user wants the capital</think>The capital of France is Paris."

    def parts(self, thinking, size):
        # Flint's template ends the prompt on <|assistant|>: the reply opens <think> itself (not `opened`)
        p = OutputParser(thinking=thinking, call_format="glm")
        evs = [e for i in range(0, len(self.TEXT), size) for e in p.feed(self.TEXT[i:i + size])] + p.finish()
        return ("".join(e.text for e in evs if e.kind == "reasoning"),
                "".join(e.text for e in evs if e.kind == "content"))

    def test_the_reply_opens_its_own_thinking(self):
        for size in (1, 3, len(self.TEXT)):
            with self.subTest(size=size):
                self.assertEqual(self.parts(True, size), ("the user wants the capital", "The capital of France is Paris."))

    def test_a_thought_it_writes_unasked_is_still_reasoning(self):
        # its template has no switch to turn thinking off, so it thinks whatever the request says
        self.assertEqual(self.parts(False, 4), ("the user wants the capital", "The capital of France is Paris."))


class FlintTokens(ByteTokenizer):
    SPECIALS = ["<|end_of_text|>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|developer|>"]
    pre = "flint"
    special_ids = {"tokenizer.ggml.eos_token_id": 256, "tokenizer.ggml.bos_token_id": 256}


class ServiceForFlint(unittest.TestCase):
    def setUp(self):
        tok = FlintTokens()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text("<|assistant|>", encoding="utf-8")
            self.svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(path))

    def test_glm_calls_and_a_reply_ends_at_every_turn_opener_llama_cpp_ends_it_at(self):
        # llama.cpp's end-of-generation tokens for Flint: <|end_of_text|>, <|user|>, <|observation|>, <|developer|>
        self.assertEqual(self.svc.call_format, "glm")
        self.assertEqual(self.svc.stop_ids, {256, 258, 260, 261})
        self.assertEqual(self.svc.turn_ids, frozenset({257, 258, 259, 260, 261}))


if __name__ == "__main__":
    unittest.main()
