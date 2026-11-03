"""docs/media/make_figures.py - the README's "How does it work?" illustrations, in the PolyStrata app's style: its colors
(serve/web/tokens.css), its icons (serve/web/sprite.svg) and its font (Outfit, embedded, SIL OFL 1.1).  Light and dark
follow the reader's system setting.

Every model has the same two: where its experts are, and how guessed words are checked.

    python docs/media/make_figures.py        # writes docs/media/how-it-works*.svg and docs/media/guess-and-check*.svg
"""
import base64
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "media"
SPRITE = (ROOT / "serve" / "web" / "sprite.svg").read_text(encoding="utf-8")
FONT = base64.b64encode((ROOT / "serve" / "web" / "fonts" / "outfit-latin-wght.woff2").read_bytes()).decode()

# the app's tokens (light, then dark)
CSS = """
@font-face{font-family:"Outfit";src:url(data:font/woff2;base64,%s) format("woff2");font-weight:100 900}
text{font-family:"Outfit",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;fill:#221e1f}
.bg{fill:#f9fafb}.card{fill:#ffffff;stroke:#dee5ee;stroke-width:1}.soft{fill:#f7f7f7}
.muted{fill:#6f6f6f}.ink{fill:#221e1f}.icon{color:#2F384C}
.acc{fill:#10b981}.acc-t{fill:#0c8a60}.acc-tint{fill:#ecfdf5}.acc-line{stroke:#10b981}
.info{fill:#0f97ff}.info-t{fill:#0668c2}.info-tint{fill:#e8f4ff}
.warn{fill:#e39b0b}.warn-t{fill:#8a5a00}.warn-tint{fill:#fff6e0}
.dan{fill:#e0283f}.dan-tint{fill:#fdecee}.dan-t{fill:#b01a2e}
.dot{fill:#dee5ee}.line{stroke:#dee5ee}.arrow{stroke:#6f6f6f;fill:none}.arrowhead{fill:#6f6f6f}
.dash{stroke:#6f6f6f;fill:none;stroke-dasharray:4 4}
.on-acc{fill:#ffffff}.icon-acc{color:#0c8a60}.icon-dan{color:#b01a2e}
@media (prefers-color-scheme:dark){
text,.ink{fill:#eef1f3}.bg{fill:#0e1113}.card{fill:#161a1d;stroke:#2a3036}.soft{fill:#1d2226}
.muted{fill:#8e979f}.icon{color:#c7cdd3}
.acc{fill:#34d399}.acc-t{fill:#4ade80}.acc-tint{fill:rgba(52,211,153,.12)}.acc-line{stroke:#34d399}
.info{fill:#3aa9ff}.info-t{fill:#7cc4ff}.info-tint{fill:rgba(58,169,255,.12)}
.warn{fill:#f5b53d}.warn-t{fill:#f8c96a}.warn-tint{fill:rgba(245,181,61,.12)}
.dan{fill:#ff5a6e}.dan-tint{fill:rgba(255,90,110,.12)}.dan-t{fill:#ff8a98}
.dot{fill:#2a3036}.line{stroke:#2a3036}.arrow{stroke:#8e979f}.arrowhead{fill:#8e979f}.dash{stroke:#8e979f}
.on-acc{fill:#052e20}.icon-acc{color:#4ade80}.icon-dan{color:#ff8a98}
}
""" % FONT

# What the two figures say of each model.  A card is (subtitle, its three lines, the caption under its drawing);
# its number is how many of the card's 130 dots, or of the RAM's 260, are drawn in color.  The first model's files have
# the names the README had before the others were drawn.
MODELS = {
    "": dict(
        head="The model is made of 24,576 small specialists - “experts”.",
        sub=("For each word it writes, it asks only 10 of them. PolyStrata keeps the busy ones",
             "on your graphics card and the rest in RAM - so it runs on a normal PC."),
        at_work=(10, "10 at work for one word"),
        gpu=("Small but very fast: 12-24 GB",
             ["Runs the core of the model", "for every word, and keeps the", "~4,000 experts used most"],
             "the most-used experts", 130),
        ram=("Big: 32-64 GB of memory",
             ["Holds all 24,576 experts. When a", "word needs one the card doesn't", "have, the CPU does it - at once"],
             "all 24,576 experts", 18),
        ssd=("Holds a 29 GB lookup table",
             ["The model looks up a few", "small rows of it per word -", "no need to keep it in RAM"],
             "a few rows per word", (1, 4)),
        guess=("A small, fast helper inside the model guesses the next words. The big model checks all the "
               "guesses in one go",
               "and keeps the ones it agrees with. The big model decides every word - it just arrives 1.6-1.8x sooner."),
        rows=("Helper guesses", "Big model checks"),
    ),
    "glm": dict(
        head="The model is made of 12,096 small specialists - “experts”.",
        sub=("Each word needs only 8 of the 288 in a layer. PolyStrata keeps the busy ones",
             "on your graphics card and the rest in RAM - so one PC runs it, not a server."),
        at_work=(8, "8 of a layer's 288 at work"),
        gpu=("It has run on a 32 GB card",
             ["Runs the core of the model", "for every word, and keeps the", "~2,470 experts used most"],
             "the most-used experts", 130),
        ram=("Big: 96 GB of memory or more",
             ["Holds the other ~9,600 experts.", "When a word needs one of", "them, the CPU does it - at once"],
             "the experts the card lacks", 18),
        ssd=("The model's file: 79-114 GB",
             ["PolyStrata uses the file where", "it is: no second copy. A start", "takes seconds once it is in RAM"],
             "read once, at the first start", (0, 1, 2, 3, 4, 5)),
        guess=("A block of the model made for guessing proposes the next words. The model checks all the "
               "guesses in one go",
               "and keeps the ones it agrees with. The model decides every word - it just arrives 1.1-1.2x sooner."),
        rows=("Block guesses", "Model checks"),
    ),
    "qwen36": dict(
        head="The model is made of 10,240 small specialists - “experts”.",
        sub=("Each word needs only 8 of the 256 in a layer. PolyStrata keeps the busy ones",
             "on your graphics card and the rest in RAM - so it runs on a normal PC."),
        at_work=(8, "8 of a layer's 256 at work"),
        gpu=("8 GB or more; 24 GB holds all",
             ["Runs the core of the model for", "every word, and keeps the busy", "experts: ~2,600 on an 8 GB card"],
             "the experts an answer keeps using", 130),
        ram=("32 GB of memory",
             ["Holds the experts the card", "lacks. When a word needs one,", "the CPU does it - at once"],
             "the experts the card lacks", 18),
        ssd=("The model's file: 13-23 GB",
             ["PolyStrata uses the file where", "it is: no second copy, and no", "lookup table to build"],
             "read once, at the first start", (0, 1, 2, 3, 4, 5)),
        guess=("The model's own draft block, which is in its file, guesses the next words. The model checks all "
               "the guesses in one go",
               "and keeps the ones it agrees with. The model decides every word - it just arrives 1.2-1.5x sooner."),
        rows=("Block guesses", "Model checks"),
    ),
    "ornith": dict(
        head="The model is made of 10,240 small specialists - “experts”.",
        sub=("Each word needs only 8 of the 256 in a layer. PolyStrata keeps the busy ones",
             "on your graphics card and the rest in RAM - so it runs on a normal PC."),
        at_work=(8, "8 of a layer's 256 at work"),
        gpu=("8 GB or more; 24 GB holds all",
             ["Runs the core of the model for", "every word, and keeps the busy", "experts: ~3,100 on an 8 GB card"],
             "the experts an answer keeps using", 130),
        ram=("32 GB of memory",
             ["Holds the experts the card", "lacks. When a word needs one,", "the CPU does it - at once"],
             "the experts the card lacks", 18),
        ssd=("The model's file: 14-24 GB",
             ["PolyStrata uses the file where", "it is: no second copy, and no", "lookup table to build"],
             "read once, at the first start", (0, 1, 2, 3, 4, 5)),
        guess=("The model's own draft block, which is in its file, guesses the next words. The model checks all "
               "the guesses in one go",
               "and keeps the ones it agrees with. The model decides every word - it just arrives 1.1-1.2x sooner."),
        rows=("Block guesses", "Model checks"),
    ),
    "gemma4": dict(
        head="The model is made of 3,840 small specialists - “experts”.",
        sub=("Each word needs only 8 of the 128 in a layer. PolyStrata keeps the busy ones",
             "on your graphics card and the rest in RAM - so it runs on a normal PC."),
        at_work=(8, "8 of a layer's 128 at work"),
        gpu=("8 GB or more; 16 GB holds all",
             ["Runs the core of the model for", "every word, and keeps the busy", "experts: ~1,000 on an 8 GB card"],
             "the experts an answer keeps using", 130),
        ram=("32 GB of memory",
             ["Holds the experts the card", "lacks. When a word needs one,", "the CPU does it - at once"],
             "the experts the card lacks", 18),
        ssd=("The model's file: 11-17 GB",
             ["PolyStrata uses the file where", "it is: no second copy, and no", "lookup table to build"],
             "read once, at the first start", (0, 1, 2, 3, 4, 5)),
        guess=("Google's small draft model for it, a file of 0.5 GB, guesses the next words. The model checks all "
               "the guesses in one go",
               "and keeps the ones it agrees with. The model decides every word - it just arrives 1.3-1.6x sooner."),
        rows=("Draft guesses", "Model checks"),
    ),
    # LFM2.5 guesses only from what the conversation repeats: it has no figure of its guesses
    "lfm25": dict(
        head="The model is made of 2,432 small specialists - “experts”.",
        sub=("Each word needs only 4 of the 64 in a layer. PolyStrata keeps the busy ones",
             "on your graphics card and the rest in RAM - so it runs on a normal PC."),
        at_work=(4, "4 of a layer's 64 at work"),
        gpu=("8 GB or more; 24 GB holds 4-bit",
             ["Runs the core of the model for", "every word, and keeps the busy", "experts: ~900-1,300 on 8 GB"],
             "the experts an answer keeps using", 130),
        ram=("32 GB of memory",
             ["Holds the experts the card", "lacks. When a word needs one,", "the CPU does it - at once"],
             "the experts the card lacks", 18),
        ssd=("The model's file: 11-25 GB",
             ["PolyStrata uses the file where", "it is: no second copy, and no", "lookup table to build"],
             "read once, at the first start", (0, 1, 2, 3, 4, 5)),
    ),
    # Flint has no draft block either
    "flint": dict(
        head="The model is made of 12,032 small specialists - “experts”.",
        sub=("Each word needs only 6 of the 256 in a layer. PolyStrata keeps the busy ones",
             "on your graphics card and the rest in RAM - so it runs on a normal PC."),
        at_work=(6, "6 of a layer's 256 at work"),
        gpu=("8 GB or more; 32 GB holds 4-bit",
             ["Runs the core of the model for", "every word, and keeps the busy", "experts: ~1,600-2,500 on 8 GB"],
             "the experts an answer keeps using", 130),
        ram=("32 GB of memory",
             ["Holds the experts the card", "lacks. When a word needs one,", "the CPU does it - at once"],
             "the experts the card lacks", 18),
        ssd=("The model's file: 18-42 GB",
             ["PolyStrata uses the file where", "it is: no second copy, and no", "lookup table to build"],
             "read once, at the first start", (0, 1, 2, 3, 4, 5)),
    ),
    # Kolibri 1 has no draft block either: its file has none, so it answers one token at a time
    "kolibri": dict(
        head="The model is made of 19,200 small specialists - “experts”.",
        sub=("Each word needs only 6 of the 384 in a layer. PolyStrata keeps the busy ones",
             "on your graphics card and the rest in RAM - so one PC runs it, not a server."),
        at_work=(6, "6 of a layer's 384 at work"),
        gpu=("It has run on a 12 GB card",
             ["Runs the core of the model", "for every word, and keeps the", "~6,000 experts used most"],
             "the most-used experts", 130),
        ram=("Big: 24 GB of memory or more",
             ["Holds the other ~13,200 experts.", "When a word needs one of", "them, the CPU does it - at once"],
             "the experts the card lacks", 36),
        ssd=("The model's file: 29-48 GB",
             ["PolyStrata uses the file where", "it is: no second copy, and no", "lookup table to build"],
             "read once, at the first start", (0, 1, 2, 3, 4, 5)),
    ),
}


def icon(name, x, y, size=28, cls="icon"):
    """A sprite icon at (x, y), `size` px, drawn in the class's color (the icons use currentColor)."""
    m = re.search(r'<symbol id="%s"[^>]*>(.*?)</symbol>' % name, SPRITE, re.S)
    return (f'<g class="{cls}" transform="translate({x},{y}) scale({size / 24:.4f})">{m.group(1)}</g>')


def text(x, y, s, size=14, weight=400, cls="ink", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" class="{cls}" '
            f'text-anchor="{anchor}">{s}</text>')


def rect(x, y, w, h, r=14, cls="card"):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>'


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{title}"><title>{title}</title><style>{CSS}</style>'
            f'{rect(0, 0, w, h, 20, "bg")}{body}</svg>')


def dots(x, y, cols, rows, step, lit, lit_cls="acc", r=2.6, seed=1):
    """A grid of expert dots; `lit` of them (spread at random) in the accent color."""
    rnd = random.Random(seed)
    n = cols * rows
    on = set(rnd.sample(range(n), min(lit, n)))
    out = []
    for i in range(n):
        cx, cy = x + (i % cols) * step, y + (i // cols) * step
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" class="{lit_cls if i in on else "dot"}"/>')
    return "".join(out)


def arrow(x1, y1, x2, y2, cls="arrow"):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2 - 7}" y2="{y2}" class="{cls}" stroke-width="1.6" stroke-linecap="round"/>'
            f'<path d="M{x2} {y2} l-8 -5 v10 z" class="arrowhead"/>')


def how_it_works(m):
    W, H = 960, 470
    b = []
    # the headline strip: one word, a few experts
    b.append(rect(24, 24, W - 48, 92))
    b.append(icon("i-chat", 48, 50, 36))
    b.append(text(100, 62, m["head"], 19, 600))
    b.append(text(100, 88, m["sub"][0], 15, 400, "muted"))
    b.append(text(100, 106, m["sub"][1], 15, 400, "muted"))
    b.append(dots(W - 262, 46, 28, 4, 7.6, m["at_work"][0], seed=3))
    b.append(text(W - 262, 94, m["at_work"][1], 12, 500, "acc-t"))
    # three cards
    cw, ch, cy = 288, 316, 136
    xs = [24, 24 + cw + 24, 24 + 2 * (cw + 24)]
    cards = [
        ("i-gpu", "Graphics card", "acc", m["gpu"][0], m["gpu"][1], m["gpu"][2], 5, 26, m["gpu"][3], "acc"),
        ("i-memory", "RAM + processor", "info", m["ram"][0], m["ram"][1], m["ram"][2], 10, 26, m["ram"][3], "info"),
        ("i-disk", "SSD", "warn", m["ssd"][0], m["ssd"][1], m["ssd"][2], 0, 0, 0, "warn"),
    ]
    for (ic, title, tone, sub, lines, dlabel, rows, cols, lit, lit_cls), x in zip(cards, xs):
        b.append(rect(x, cy, cw, ch))
        b.append(rect(x + 20, cy + 20, 48, 48, 12, tone + "-tint"))
        b.append(icon(ic, x + 30, cy + 30, 28, "icon"))
        b.append(text(x + 82, cy + 42, title, 19, 700))
        b.append(text(x + 82, cy + 62, sub, 13, 500, tone + "-t"))
        for i, ln in enumerate(lines):
            b.append(text(x + 20, cy + 104 + i * 21, ln, 15, 400, "ink"))
        if rows:
            b.append(dots(x + 26, cy + 196, cols, rows, 9.4, lit, lit_cls, seed=len(title)))
            b.append(text(x + 20, cy + 196 + rows * 9.4 + 14, dlabel, 12, 500, "muted"))
        else:                                       # the SSD: a file's rows, the ones being read in color
            for r in range(6):
                yy = cy + 190 + r * 17
                b.append(rect(x + 20, yy, cw - 40, 11, 4, "warn" if r in m["ssd"][3] else "soft"))
            b.append(text(x + 20, cy + 306, dlabel, 12, 500, "muted"))
    return svg(W, H, "".join(b), "How PolyStrata splits the model between your graphics card, RAM and SSD")


def guess_and_check(m):
    W, H = 960, 330
    b = [rect(24, 24, W - 48, H - 48)]
    b.append(icon("i-thinking", 48, 48, 30))
    b.append(text(92, 70, "Guess, then check - many words per step", 19, 700))
    b.append(text(92, 94, m["guess"][0], 15, 400, "muted"))
    b.append(text(92, 113, m["guess"][1], 15, 400, "muted"))
    words = ["def", "reverse", "(items", "):"]
    fixed = ", key):"                                  # the model's own word where the guess was wrong
    ok = [True, True, True, False]
    # row 1: the guesses (dashed chips)
    y1, y2 = 156, 238
    b.append(text(48, y1 + 24, m["rows"][0], 14, 600, "muted"))
    b.append(text(48, y2 + 24, m["rows"][1], 14, 600, "muted"))
    x = 176
    xs = []
    for i, w in enumerate(words[:4]):
        wd = 34 + len(w) * 9
        xs.append((x, wd))
        b.append(f'<rect x="{x}" y="{y1}" width="{wd}" height="38" rx="10" class="dash" stroke-width="1.4"/>')
        b.append(text(x + wd / 2, y1 + 25, w, 16, 500, "ink", "middle"))
        x += wd + 14
    # row 2: the verdicts
    for i, ((x, wd), good) in enumerate(zip(xs, ok)):
        tone = "acc" if good else "dan"
        b.append(rect(x, y2, wd, 38, 10, tone + "-tint"))
        b.append(text(x + wd / 2 - 8, y2 + 25, words[i], 16, 600, tone + "-t", "middle"))
        b.append(icon("i-check" if good else "i-error", x + wd - 22, y2 + 11, 16, "icon-" + tone))
        if not good:                                   # the model's own word beside the rejected guess
            fx = x + wd + 10
            fw = 26 + len(fixed) * 9
            b.append(rect(fx, y2, fw, 38, 10, "acc-tint"))
            b.append(text(fx + fw / 2, y2 + 25, fixed, 16, 600, "acc-t", "middle"))
            xs[i] = (fx, fw)
        b.append(f'<line x1="{x + wd / 2}" y1="{y1 + 40}" x2="{x + wd / 2}" y2="{y2 - 4}" class="line" stroke-width="1.4"/>')
    # the result
    rx = xs[-1][0] + xs[-1][1] + 40
    b.append(arrow(rx, y2 + 19, rx + 30, y2 + 19))
    b.append(rect(rx + 42, y2 - 6, W - 48 - (rx + 42), 50, 12, "acc"))
    b.append(icon("i-bolt", rx + 54, y2 + 6, 26, "on-acc-icon"))
    b.append(text(rx + 90, y2 + 16, "4 words in one step", 15, 700, "on-acc"))
    b.append(text(rx + 90, y2 + 34, "3 guesses kept + its own word", 12, 500, "on-acc"))
    return svg(W, H, "".join(b).replace('class="on-acc-icon"', 'class="icon" style="color:#ffffff"'),
               "Speculative decoding: a small helper guesses words, the big model checks them at once")


if __name__ == "__main__":
    for name, m in MODELS.items():
        tail = ("-" + name if name else "") + ".svg"
        made = ["how-it-works" + tail] + (["guess-and-check" + tail] if "guess" in m else [])
        (OUT / made[0]).write_text(how_it_works(m), encoding="utf-8")
        if "guess" in m:
            (OUT / made[1]).write_text(guess_and_check(m), encoding="utf-8")
        for f in made:
            print(f, (OUT / f).stat().st_size, "bytes")
