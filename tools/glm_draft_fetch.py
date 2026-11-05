"""tools/glm_draft_fetch.py - GLM 5.3 Flash's draft block, without the file it is in.

The model has one block more than its layers: it guesses the token after the next one, and the engine checks the
guesses a few at a time (docs/GLM.md).  The block is the model's, not a file's, and one serves every size.  Some
GGUFs leave it out (the 3.0BIT file), others carry it at their end.  A GGUF starts with a directory of its
tensors (name, type, shape, offset), so HTTP range requests can read the directory and then only the block.

    python tools/glm_draft_fetch.py fetch --out FILE.gguf     # the block as a GGUF of its own, resumable
    python tools/glm_draft_fetch.py verify --out FILE.gguf    # FILE against SHA256 (exit 3: missing or corrupt)

The block is copied as it is in the source, so the engine reads it like any layer (--draft-model).  Nothing here
runs a model.

A mirror or proxy that ignores the Range header answers with the file's start, so a range read needs a 206 with
the range that was asked for (mtp_fetch.check_range, #327), and the finished file is checked against SHA256.
"""
import argparse
import os
import struct
import sys
import urllib.request

from concurrent.futures import ThreadPoolExecutor

import hf_auth
from mtp_fetch import BAD, get, sha256_of

# A fixed commit of the repository, so every install reads the same bytes; STRATA_GLM_DRAFT_REVISION overrides it.
PINNED_REVISION = "621d456e93e926e4b52f85cff5f634358c1828f9"
REVISION = os.environ.get("STRATA_GLM_DRAFT_REVISION") or PINNED_REVISION
HF_ENDPOINT = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or "https://huggingface.co"
REPO = HF_ENDPOINT + "/unsloth/GLM-5.3-Flash-GGUF/resolve/%s/" % REVISION
SOURCE = "unsloth/GLM-5.3-Flash-GGUF@%s UD-IQ2_XXS" % REVISION    # written into the file: the same from any mirror
# the numbered files of this size: the block is at the model's end, and the cut between two files goes through it
SHARDS = ["UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-%05d-of-00004.gguf" % i for i in range(1, 5)]
# of the file `fetch` writes from PINNED_REVISION, 2,787,998,016 bytes
SHA256 = "91a04445b42bcd7a99980b9eab3c3b66734efb17093f20d0b6c21a320a6e3218"
ALIGN = 32
PIECE = 32 << 20              # one range request
WORKERS = 4                   # requests at a time: a mirror gives one connection a few MB/s
# the GGUF metadata value types that are numbers, as struct formats; 8 is a string, 9 an array
NUMBER = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Short(Exception):
    """The directory goes on past the bytes read so far."""


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def take(self, n):
        if self.at + n > len(self.data):
            raise Short()
        self.at += n
        return self.data[self.at - n:self.at]

    def num(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def text(self):
        return self.take(self.num("Q")).decode("utf-8")

    def value(self, kind):
        """A metadata value: a number, a string or a list of numbers.  Other lists (the vocabulary) are passed
        over and come back as None."""
        if kind == 8:
            return self.text()
        if kind == 9:
            item, count = self.num("I"), self.num("Q")
            if item in NUMBER:
                fmt = "<%d%s" % (count, NUMBER[item])
                return list(struct.unpack(fmt, self.take(struct.calcsize(fmt))))
            for _ in range(count):
                self.value(item)
            return None
        if kind in NUMBER:
            return self.num(NUMBER[kind])
        raise ValueError("GGUF metadata type %d" % kind)


def directory(data):
    """A GGUF from the bytes at its start: its tensors [(name, dims, type, offset)], where the data begins, and
    its metadata {key: value}."""
    r = Reader(data)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version, n_tensors, n_kv = r.num("I"), r.num("Q"), r.num("Q")
    if version not in (2, 3):
        raise ValueError("GGUF version %d" % version)
    meta = {}
    for _ in range(n_kv):
        key, kind = r.text(), r.num("I")
        meta[key] = r.value(kind)
    align = meta.get("general.alignment") or ALIGN
    tensors = []
    for _ in range(n_tensors):
        name = r.text()
        dims = [r.num("Q") for _ in range(r.num("I"))]
        tensors.append((name, dims, r.num("I"), r.num("Q")))
    return tensors, (r.at + align - 1) // align * align, meta


def block_of(files):
    """The draft block in `files`, [(url, a GGUF's tensors, where its data begins, its size, its metadata)]: its
    tensors [(name, dims, type, url, first byte in that file, bytes)], and what the engine needs from the model's
    metadata to run it, [(key, value)].  It is the block that has `nextn` tensors.  A tensor ends where the next
    one in its file starts."""
    blocks = {n.split(".")[1] for _, tensors, _, _, _ in files for n, _, _, _ in tensors
              if n.startswith("blk.") and ".nextn." in n}
    if len(blocks) != 1:
        return [], []
    index = int(blocks.pop())
    prefix, out = "blk.%d." % index, []
    for url, tensors, data_start, size, _ in files:
        ends = sorted(o for _, _, _, o in tensors) + [size - data_start]
        out += [(n, d, t, url, data_start + o, ends[ends.index(o) + 1] - o) for n, d, t, o in tensors
                if n.startswith(prefix)]
    # The SwiGLU limits are kept a block: the model's file has them for its own blocks, so the draft block's go
    # with it, under the names the model's file gives them.
    meta = next((m for _, _, _, _, m in files if isinstance(m.get("general.architecture"), str)), {})
    arch = meta.get("general.architecture", "")
    kv = [("general.architecture", arch)] if arch else []
    for key in sorted(meta):
        if key.startswith(arch + ".swiglu_") and meta[key] is not None:
            v = meta[key]
            kv.append((key, float(v[index] if isinstance(v, list) else v)))
    return out, kv


def header(block, kv, source):
    """The start of a GGUF that holds `block` and nothing else, and each tensor's first byte in that file.  kv:
    metadata beside its name and source, strings and floats."""
    def text(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    kv = [("general.name", "GLM 5.3 Flash draft block"), ("general.source", source)] + kv
    out = b"GGUF" + struct.pack("<IQQ", 3, len(block), len(kv))
    for key, v in kv:
        out += text(key) + (struct.pack("<I", 8) + text(v) if isinstance(v, str) else struct.pack("<If", 6, v))
    at, starts = 0, []
    for name, dims, kind, _, _, size in block:
        out += text(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        out += struct.pack("<IQ", kind, at)
        starts.append(at)
        at += (size + ALIGN - 1) // ALIGN * ALIGN
    out += b"\0" * (-len(out) % ALIGN)
    return out, [len(out) + s for s in starts], len(out) + at


def size_of(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-mtp-fetch"})
    with hf_auth.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])


def find_block(read=get, size=size_of):
    """The draft block (block_of), from the starts of SHARDS."""
    files = []
    for shard in SHARDS:
        url, n = REPO + shard, 1 << 22
        total = size(url)
        while True:
            try:
                tensors, data_start, meta = directory(read(url, 0, min(n, total) - 1))
                break
            except Short:
                if n >= total:
                    raise ValueError("%s ends inside its tensor directory" % shard)
                n *= 4
        files.append((url, tensors, data_start, total, meta))
    block, kv = block_of(files)
    if not block:
        sys.exit("the files have no draft block: " + ", ".join(SHARDS))
    return block, kv


def fetch(out, read=get, size=size_of):
    block, kv = find_block(read, size)
    head, starts, total = header(block, kv, SOURCE)
    part = out + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # The pieces are written in the file's order, so what is there already counts up to its last whole piece.
    have = os.path.getsize(part) if os.path.exists(part) else 0
    with open(part, "r+b" if have else "w+b") as f, ThreadPoolExecutor(WORKERS) as pool:
        f.seek(0)
        if f.read(len(head)) != head:
            have = 0
            f.seek(0)
            f.truncate()
            f.write(head)
        for (name, _, _, url, src, n), dst in zip(block, starts):
            todo = [(at, min(PIECE, n - at)) for at in range(0, n, PIECE)]
            todo = [(at, m) for at, m in todo if dst + at + m > have]
            for i in range(0, len(todo), WORKERS):
                group = todo[i:i + WORKERS]
                for (at, _), data in zip(group, pool.map(lambda p: read(url, src + p[0], src + p[0] + p[1] - 1),
                                                         group)):
                    f.seek(dst + at)
                    f.write(data)
                f.flush()
            print("  %-44s %8.1f MB" % (name, n / 1e6), flush=True)
        f.truncate(total)
    if SHA256 and REVISION == PINNED_REVISION and sha256_of(part) != SHA256:
        os.remove(part)
        sys.exit("the draft block that was read is not the pinned revision's (SHA256): a mirror or proxy changed "
                 "it; run the setup again")
    os.replace(part, out)
    print("draft block: %s, %.2f GB" % (out, total / 1e9))


def verify(out):
    if not os.path.exists(out):
        print("missing: %s" % out)
        return BAD
    if SHA256 and REVISION == PINNED_REVISION and sha256_of(out) != SHA256:
        print("corrupt: %s" % out)
        return BAD
    try:
        with open(out, "rb") as f:
            tensors, _, _ = directory(f.read(1 << 22))
    except (Short, ValueError) as e:
        print("corrupt: %s (%s)" % (out, e))
        return BAD
    if not any(".nextn." in n for n, _, _, _ in tensors):
        print("corrupt: %s has no draft block" % out)
        return BAD
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "verify"])
    ap.add_argument("--out", required=True, help="the GGUF to write or check")
    a = ap.parse_args()
    if a.command == "fetch":
        fetch(a.out)
        return 0
    return verify(a.out)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except hf_auth.HFAuthError as error:
        sys.exit(str(error))
