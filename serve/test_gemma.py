"""Gemma 4 on the Python side: its tokenizer (a BPE over UTF-8 pieces with a visible space), its reasoning, which
the model opens and closes with tokens of its own, and its tool calls."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser, openai_to_messages, parse_gemma_tool_call  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine  # noqa: E402

SP = ST.SPACE
BYTES = ["<0x%02X>" % b for b in range(256)]
TOKENS = ["<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>", "\n", "\n\n", SP, "a", "b", "c", "ab", SP + "ab",
          SP + "c", "abc"] + BYTES
TYPES = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [6] * 256
# a merge may pass through a piece the vocabulary does not keep ("bc"), and a piece may hold newlines
MERGES = ["a b", SP + " ab", SP + " c", "b c", "\n \n"]


class Tokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = ST.Tokenizer(TOKENS, MERGES, TYPES, pre="gemma4")
        self.id = {t: i for i, t in enumerate(TOKENS)}

    def test_spaces_are_pieces_and_merges_go_by_rank(self):
        ids = self.tok.encode("ab ab c")
        self.assertEqual([TOKENS[i] for i in ids], ["ab", SP + "ab", SP + "c"])
        self.assertEqual(self.tok.decode(ids), "ab ab c")

    def test_a_run_of_newlines_that_is_a_token_is_that_token(self):
        self.assertEqual([TOKENS[i] for i in self.tok.encode("a\n\nb\nc")], ["a", "\n\n", "b", "\n", "c"])

    def test_what_the_vocabulary_lacks_goes_byte_by_byte(self):
        ids = self.tok.encode("aé")
        self.assertEqual([TOKENS[i] for i in ids], ["a", "<0xC3>", "<0xA9>"])
        self.assertEqual(self.tok.decode(ids), "aé")
        # a merge to a piece it lacks: its bytes, as llama.cpp writes it, not the pieces it was merged from
        self.assertEqual([TOKENS[i] for i in self.tok.encode("bc")], ["<0x62>", "<0x63>"])
        self.assertEqual(self.tok.decode(self.tok.encode("bc")), "bc")

    def test_lines_seen_before_give_the_same_ids(self):
        text = "ab ab c\n\naé bc\nab ab c\nabc\n" * 20
        off = ST.Tokenizer(TOKENS, MERGES, TYPES, pre="gemma4")
        off.PIECE_CACHE_MAX = 0
        want = off.encode(text)
        self.assertEqual(self.tok.encode(text), want)      # the lines are remembered here,
        self.assertEqual(self.tok.encode(text), want)      # and looked up here
        self.assertGreater(len(self.tok._piece_ids), 0)
        self.assertEqual(self.tok.decode(want), text)
        small = ST.Tokenizer(TOKENS, MERGES, TYPES, pre="gemma4")
        small.PIECE_CACHE_MAX = 2
        self.assertEqual(small.encode(text), want)
        self.assertLessEqual(len(small._piece_ids), 2)

    def test_control_tokens_only_when_asked(self):
        self.assertEqual(self.tok.encode("<|turn>a", parse_special=True), [self.id["<|turn>"], self.id["a"]])
        self.assertNotIn(self.id["<|turn>"], self.tok.encode("<|turn>a"))

    def test_a_directory_keeps_merges_with_newlines(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "vocab.json").write_text(json.dumps(self.tok.ids, ensure_ascii=False), encoding="utf-8")
            (d / "merges.json").write_text(json.dumps(MERGES, ensure_ascii=False), encoding="utf-8")
            (d / "token_type.json").write_text(json.dumps(TYPES))
            (d / "tokenizer.json").write_text(json.dumps({"pre": "gemma4", "special_ids": {}}))
            again = ST.Tokenizer.from_dir(d)
        self.assertEqual(again.encode("ab ab\n\nc"), self.tok.encode("ab ab\n\nc"))
        self.assertEqual(again.ranks[("\n", "\n")], 4)


class Reasoning(unittest.TestCase):
    def events(self, pieces, thinking=True):
        p = OutputParser(thinking=thinking, call_format="gemma")
        out = [e for piece in pieces for e in p.feed(piece)] + p.finish()
        return ("".join(e.text for e in out if e.kind == "reasoning"),
                "".join(e.text for e in out if e.kind == "content"))

    def test_the_thought_channel_is_the_reasoning(self):
        text = "<|channel>thought\nThe user asks for a capital.<channel|>\nParis."
        for size in (1, 3, len(text)):
            pieces = [text[i:i + size] for i in range(0, len(text), size)]
            self.assertEqual(self.events(pieces), ("The user asks for a capital.", "Paris."))

    def test_an_answer_right_after_a_quote_is_content(self):
        # a thought that ends on a backtick: its answer follows <channel|> at once, as in every Gemma turn
        text = "<|channel>thought\nOutput: `«x» 🙂`<channel|>«x» 🙂"
        for size in (1, 3, len(text)):
            pieces = [text[i:i + size] for i in range(0, len(text), size)]
            self.assertEqual(self.events(pieces), ("Output: `«x» 🙂`", "«x» 🙂"))

    def test_a_reply_without_a_thought_is_content(self):
        self.assertEqual(self.events(["Par", "is."]), ("", "Paris."))
        self.assertEqual(self.events(["<|chan", "ged> x"]), ("", "<|changed> x"))
        # asked not to think, it can still open with a thought (an empty one after a tool's answer)
        self.assertEqual(self.events(["<|channel>thought\nx<channel|>y"], thinking=False), ("x", "y"))
        self.assertEqual(self.events(["<|channel>thought\n<chan", "nel|>y"], thinking=False), ("", "y"))

    def test_a_prompt_that_opened_the_thought_starts_inside_it(self):
        p = OutputParser(thinking=True, call_format="gemma", opened=True)
        out = p.feed("The tool says 5.<channel|>It is 5.") + p.finish()
        self.assertEqual([(e.kind, e.text) for e in out], [("reasoning", "The tool says 5."), ("content", "It is 5.")])
        p = OutputParser(thinking=False, call_format="gemma", opened=True)    # asked not to think: nothing to open
        self.assertEqual([e.kind for e in p.feed("It is 5.") + p.finish()], ["content"])


Q = '<|"|>'


class ToolCalls(unittest.TestCase):
    SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "n": {"type": "integer"},
                                                              "content": {"type": "string"}}}}]
    CONTENT = "a brace }, a quote \", a call's end <tool_call|> and a comma, all inside\nend"
    TEXT = (f"<|channel>thought\nI need the file.<channel|>I will write it.<|tool_call>call:write{{path:{Q}doc.md{Q},"
            f"n:3,content:{Q}{CONTENT}{Q}}}<tool_call|>")

    def events(self, stream_tools, step, text=None):
        p = OutputParser(thinking=True, tools=self.SCHEMA, stream_tools=stream_tools, call_format="gemma")
        text = self.TEXT if text is None else text
        return [e for i in range(0, len(text), step) for e in p.feed(text[i:i + step])] + p.finish()

    def test_the_notation(self):
        call = parse_gemma_tool_call(
            f"call:get_weather{{location:{Q}Paris, France{Q},days:3,ratio:-1.5e1,units:{{temp:{Q}C{Q},wind:null}},"
            f"tags:[{Q}a{Q},2,true,[false]],{Q}quoted key{Q}:{Q}{Q},empty:{{}},none:[]}}")
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.arguments, {"location": "Paris, France", "days": 3, "ratio": -15.0,
                                          "units": {"temp": "C", "wind": None}, "tags": ["a", 2, True, [False]],
                                          "quoted key": "", "empty": {}, "none": []})
        self.assertEqual(parse_gemma_tool_call("call:now{}").arguments, {})
        self.assertEqual((parse_gemma_tool_call("call:now").name, parse_gemma_tool_call("call:now").arguments),
                         ("now", {}))

    def test_a_string_parameter_written_bare_stays_a_string(self):
        call = parse_gemma_tool_call("call:write{path:7,n:7}", self.SCHEMA[0])
        self.assertEqual(call.arguments, {"path": "7", "n": 7})

    def test_a_call_in_any_pieces(self):
        for stream_tools in (False, True):
            for step in (1, 3, 7, len(self.TEXT)):
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = self.events(stream_tools, step)
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual([c.name for c in calls], ["write"])
                    self.assertEqual(calls[0].arguments, {"path": "doc.md", "n": 3, "content": self.CONTENT})
                    self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "I need the file.")
                    self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "I will write it.")

    def test_a_streamed_call_is_announced_first(self):
        evs = [e for e in self.events(True, 1) if e.kind.startswith("tool_")]
        self.assertEqual([e.kind for e in evs], ["tool_start", "tool_args", "tool_call"])
        self.assertEqual(json.loads(evs[1].text)["content"], self.CONTENT)
        self.assertEqual({e.call.id for e in evs}, {evs[0].call.id})

    def test_calls_in_a_row(self):
        text = f"<|tool_call>call:a{{x:1}}<tool_call|><|tool_call>call:b{{y:{Q}2{Q}}}<tool_call|>"
        calls = [(e.call.name, e.call.arguments) for e in self.events(False, 5, text) if e.kind == "tool_call"]
        self.assertEqual(calls, [("a", {"x": 1}), ("b", {"y": "2"})])

    def test_a_call_the_output_ended_in_is_text(self):
        evs = self.events(False, 4, f"<|tool_call>call:write{{path:{Q}doc")
        self.assertEqual([(e.kind, e.text) for e in evs], [("content", f"<|tool_call>call:write{{path:{Q}doc")])


class GemmaTokens(ByteTokenizer):
    SPECIALS = ["<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>",
                "<|tool_response>", "<tool_response|>", Q]
    pre = "gemma4"
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 258}


# what of Gemma 4's chat template the server has to meet: it thinks only when told to, reads a tool as OpenAI sends
# it, names a tool's answer by the call's id, and opens no model turn after an answer that came with tool calls
TEMPLATE = """{{- bos_token -}}{%- set think = enable_thinking | default(false) -%}
{%- set ns = namespace(last=none) -%}
{%- if think or tools -%}<|turn>system
{% if think %}<|think|>{% endif %}{% for tool in tools or [] %}<|tool>{{ tool['function']['name'] }}<tool|>{% endfor %}<turn|>
{% endif -%}
{%- for m in messages if m['role'] != 'tool' -%}
<|turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}
{% set ns.last = none -%}
{%- for c in m.get('tool_calls') or [] -%}<|tool_call>call:{{ c['function']['name'] }}{}<tool_call|>{%- set ns.last = 'call' -%}{%- endfor -%}
{%- if m.get('tool_calls') -%}{%- for t in messages[loop.index0 + 1:] if t['role'] == 'tool' -%}
{%- for c in m['tool_calls'] if c.get('id') == t.get('tool_call_id') -%}<|tool_response>response:{{ c['function']['name'] }}{value:{{ t['content'] }}}<tool_response|>{%- endfor -%}
{%- set ns.last = 'response' -%}{%- endfor -%}{%- endif -%}
{{- m['content'] -}}
{%- if not (ns.last == 'response' and not m['content']) %}<turn|>
{% endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{%- if ns.last != 'response' -%}<|turn>model
{% if not think %}<|channel>thought
<channel|>{% endif %}
{%- elif think -%}<|channel>thought
{% endif -%}
{%- endif -%}"""


class ServiceForGemma(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {"name": "add", "parameters": {"properties": {"a": {"type": "integer"}}}}}]
    ASKED = [{"role": "user", "content": "1+2?"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"id": "c1", "type": "function", "function": {"name": "add", "arguments": "{\"a\": 1}"}},
                 {"id": "c2", "type": "function", "function": {"name": "sub", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c2", "content": "0"},
             {"role": "tool", "tool_call_id": "c1", "content": "3"}]

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        path = Path(self.dir.name) / "chat_template.jinja"
        path.write_text(TEMPLATE, encoding="utf-8")
        self.tok = GemmaTokens()
        self.svc = Service(MockEngine(self.tok, "hi"), self.tok, ChatTemplate(path))

    def prompt(self, req):
        messages, tools, kwargs = openai_to_messages(req)
        return self.svc.render_prompt(messages, tools, kwargs), tools

    def test_the_reply_ends_where_the_tools_answer_would_start(self):
        self.assertEqual(self.svc.call_format, "gemma")
        self.assertEqual(self.svc.stop_ids, {258, 256 + GemmaTokens.SPECIALS.index("<|tool_response>")})

    def test_the_turn_end_stops_a_reply_when_the_gguf_names_only_eos(self):
        # a finetune's GGUF (HauhauCS 1M) names eos and no eot: the reply still ends at <turn|>
        class EosOnly(GemmaTokens):
            special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 1}
        tok = EosOnly()
        svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(Path(self.dir.name) / "chat_template.jinja"))
        self.assertIn(256 + GemmaTokens.SPECIALS.index("<turn|>"), svc.stop_ids)

    def test_thinking_is_on_unless_asked_off_and_a_tool_is_wrapped(self):
        text, _ = self.prompt({"messages": self.ASKED[:1], "tools": self.TOOLS})
        self.assertEqual(text, "<bos><|turn>system\n<|think|><|tool>add<tool|><turn|>\n<|turn>user\n1+2?<turn|>\n"
                               "<|turn>model\n")
        text, _ = self.prompt({"messages": self.ASKED[:1], "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(text, "<bos><|turn>user\n1+2?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
        self.assertEqual(self.svc.encode_prompt(*openai_to_messages({"messages": self.ASKED[:1]}))[0], 256)   # <bos>

    def test_no_messages_is_refused(self):
        # its template renders a prompt for none, where Qwen's raises: the request is a 400 either way
        with self.assertRaisesRegex(ValueError, "No messages provided"):
            self.prompt({"messages": []})

    def test_answers_go_in_the_order_of_their_calls_under_their_names(self):
        # sent sub's first: the model reads answers by position, so they are put back in the calls' order
        text, _ = self.prompt({"messages": self.ASKED, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertTrue(text.endswith("<|tool_call>call:add{}<tool_call|><|tool_call>call:sub{}<tool_call|>"
                                      "<|tool_response>response:add{value:3}<tool_response|>"
                                      "<|tool_response>response:sub{value:0}<tool_response|>"), text)

    def test_the_reasoning_the_prompt_opened(self):
        messages, tools, kwargs = openai_to_messages({"messages": self.ASKED, "tools": self.TOOLS})
        ids = self.svc.encode_prompt(messages, tools, kwargs)
        self.assertEqual(ids[-len(self.svc.think_open_ids):], self.svc.think_open_ids)
        # the engine's reply goes on inside the thought: it is reasoning, then the answer
        self.svc.engine = MockEngine(self.tok, "Three.<channel|>It is 3.<turn|>")
        evs = [e for kind, e in self.svc.run(ids, True, tools, 64, {}, __import__("threading").Event())
               if kind == "event"]
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "Three.")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "It is 3.")

    def test_the_thinking_budget_closes_the_thought_with_its_own_tag(self):
        # Discussion 60: the wrap-up closed it with </think>, which is text inside Gemma's thought, and the model
        # thought on to max_tokens with no answer
        tok = self.tok

        class Thinks(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                done = ids[-1] == 256 + GemmaTokens.SPECIALS.index("<channel|>")
                text = "It is 3." if done else "<|channel>thought\nLet me count. " + "Then more. " * 40 + "<channel|>It is 3."
                yield from (tok.encode(text, parse_special=True) + tok.encode("<turn|>", parse_special=True))[:max_new]

        ids = self.svc.encode_prompt(*openai_to_messages({"messages": self.ASKED[:1], "tools": self.TOOLS}))
        self.svc.engine = Thinks(self.tok, "")
        evs = [e for kind, e in self.svc.run(ids, True, self.TOOLS, 400, {"reasoning_budget_tokens": 20},
                                             __import__("threading").Event()) if kind == "event"]
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"),
                         "Let me count. Then \n\nI have thought about this long enough; time to give my answer.\n")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "It is 3.")

    def test_a_turn_for_the_model_after_an_answer_with_calls(self):
        said = [dict(self.ASKED[1], content="Let me add.")] + self.ASKED[2:]
        text, _ = self.prompt({"messages": self.ASKED[:1] + said})
        self.assertTrue(text.endswith("Let me add.<turn|>\n<|turn>model\n"), text)
        text, _ = self.prompt({"messages": self.ASKED[:1] + said, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertTrue(text.endswith("Let me add.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"), text)


class ImageTokens(GemmaTokens):
    SPECIALS = GemmaTokens.SPECIALS + ["<|image>", "<|image|>", "<image|>"]


class AnImage(unittest.TestCase):
    """Gemma 4's template writes <|image|> for an image and no more; the tokens around its rows are the server's."""

    class FakeVision:
        def __init__(self, d):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        def encode(self, source):
            return self.rows, 3

    def test_an_images_rows_between_the_models_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text("{{- bos_token -}}{%- for m in messages -%}<|turn>{{ m['role'] }}\n"
                            "{%- for item in m['content'] -%}"
                            "{%- if item['type'] == 'image' -%}<|image|>{%- else -%}{{ item['text'] }}{%- endif -%}"
                            "{%- endfor -%}<turn|>\n{%- endfor -%}<|turn>model\n", encoding="utf-8")
            class Asked(MockEngine):
                def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                    self.asked = (StrataEngine.sampling_keys(sampling), embeddings)
                    yield from super().generate(ids, max_new, sampling, cancel, embeddings)

            tok = ImageTokens()
            svc = Service(Asked(tok, "a cat"), tok, ChatTemplate(path), vision=self.FakeVision(d))
            begin, pad, end = (256 + ImageTokens.SPECIALS.index(s) for s in ("<|image>", "<|image|>", "<image|>"))
            msgs = [{"role": "user", "content": [{"type": "image", "source": "x.png"},
                                                 {"type": "text", "text": "what is this?"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            at = ids.index(begin)
            self.assertEqual(ids[at:at + 5], [begin, pad, pad, pad, end])
            self.assertEqual((ids.count(pad), svc.embeddings.pad), (3, pad))
            # the engine is told which token stands for an image's rows
            rows = svc.embeddings.path
            list(svc.run(ids, False, None, 8, {}, __import__("threading").Event()))
            self.assertEqual(svc.engine.asked, (f" image_pad={pad}", rows))


if __name__ == "__main__":
    unittest.main()
