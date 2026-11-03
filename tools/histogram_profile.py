"""tools/histogram_profile.py - an expert-cache profile from counted routing: which experts a model really used.

`make_profile.py` ranks pairs from `--dump-routing` traces of one engine run.  This tool ranks them from a routing
*histogram* - counts of how often each (layer, expert) was selected, gathered over many real requests - so the
profile reflects a workload rather than one prompt.  Two inputs are read:

  * a NautiBurn histogram database (SQLite): table `histogram_counts(model_id, class_id, phase, layer, expert_id,
    count)` and table `models(model_id, n_layers, n_experts, ...)`.  Pick the model with `--model-id`; `--class` and
    `--phase` narrow which requests count.
  * a CSV with a `layer,expert,count` header (any extra columns are ignored), for counts gathered some other way.

The order: pairs by weighted count, most used first; then every pair that was never counted, interleaved across
the layers (as make_profile.py fills).  Decode is what the VRAM tier serves token after token, so by default only
decode counts; `--phase all` adds prefill at `--prefill-weight`.  Every layer routes the same number of selections
per token, so one global ranking compares layers fairly.

Two files can be written, picked by `--format` (default: from the extension):

  * `counts` (`.txt`): lines of "layer expert count", what strata-poly's `--expert-profile` reads and saves
    (src/poly/engine.cpp read_profile): it places experts in VRAM by use per byte from these counts.
  * `strp` (`.bin`): make_profile.py's format for the Qwen3.8-Flash-Next engine (STRP version 1, little-endian): the
    header, the ranked (layer, expert) pairs as u16, then the per-pair slot table as i32.

`--coverage` prints the share of counted selections the top N ranked pairs would have served - measured on the
counts themselves, so it is an upper bound for unseen requests.

    python tools/histogram_profile.py --db sidechannel.sqlite --model-id qwen3.6-35b-a3b --out profile.txt
    python tools/histogram_profile.py --csv counts.csv --n-layers 32 --n-expert 8 --out profile.bin --coverage 64
"""
import argparse
import csv
import sqlite3
import struct
from collections import defaultdict

MAGIC, VERSION = b"STRP", 1


def read_db(path, model_id, classes=None, phase="decode", prefill_weight=0.25):
    """(n_layers, n_expert, {(layer, expert): weight}) for one model in a NautiBurn histogram database."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT n_layers, n_experts FROM models WHERE model_id = ?", (model_id,)).fetchone()
        if row is None:
            known = [r[0] for r in con.execute("SELECT model_id FROM models ORDER BY model_id")]
            raise SystemExit(f"{path}: no model {model_id!r}; known: {', '.join(known)}")
        n_layers, n_expert = int(row[0]), int(row[1])
        sql = "SELECT phase, layer, expert_id, SUM(count) FROM histogram_counts WHERE model_id = ?"
        args = [model_id]
        if classes:
            sql += " AND class_id IN (%s)" % ",".join("?" * len(classes))
            args += list(classes)
        if phase != "all":
            sql += " AND phase = ?"
            args.append(phase)
        sql += " GROUP BY phase, layer, expert_id"
        freq = defaultdict(float)
        for ph, layer, e, c in con.execute(sql, args):
            freq[(int(layer), int(e))] += float(c) * (prefill_weight if phase == "all" and ph == "prefill" else 1.0)
        if not freq:
            have = con.execute("SELECT DISTINCT class_id, phase FROM histogram_counts WHERE model_id = ? "
                               "ORDER BY class_id, phase", (model_id,)).fetchall()
            raise SystemExit(f"{path}: no {phase} counts for {model_id!r}" + (f" in class {', '.join(classes)}"
                             if classes else "") + "; it has: " + (", ".join(f"{c}/{p}" for c, p in have) or "none"))
    finally:
        con.close()
    return n_layers, n_expert, freq


def read_csv(path):
    """{(layer, expert): count} from a CSV with layer, expert and count columns."""
    freq = defaultdict(float)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            freq[(int(row["layer"]), int(row["expert"]))] += float(row["count"])
    return freq


def rank_pairs(freq, n_layers, n_expert):
    """Counted pairs most used first (ties by layer, then expert), then the uncounted pairs interleaved by layer."""
    counted = sorted((p for p, c in freq.items() if c > 0 and 0 <= p[0] < n_layers and 0 <= p[1] < n_expert),
                     key=lambda p: (-freq[p], p))
    seen = set(counted)
    fill = [(layer, e) for e in range(n_expert) for layer in range(n_layers) if (layer, e) not in seen]
    return counted + fill, len(counted)


def write_profile(path, ranked, n_layers, n_expert):
    table = [[-1] * n_expert for _ in range(n_layers)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, n_layers, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(n_layers):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def write_counts(path, freq, n_layers, n_expert):
    """'layer expert count' lines, layer then expert order, for every pair counted at least once (rounded)."""
    with open(path, "w") as f:
        for layer in range(n_layers):
            for e in range(n_expert):
                n = round(freq.get((layer, e), 0.0))
                if n > 0:
                    f.write(f"{layer} {e} {n}\n")


def read_counts(path):
    """{(layer, expert): count} from a 'layer expert count' file."""
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                out[(int(parts[0]), int(parts[1]))] = int(parts[2])
    return out


def read_profile(path):
    """(n_layers, n_expert, ranked pairs) - the checks read_expert_profile in src/core/expert_cache.cpp makes."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a PolyStrata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if ver != VERSION or n > slots:
        raise SystemExit(f"{path}: version {ver}, {n} ranked of {slots} slots")
    pairs = [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]
    if any(not (0 <= l < nl and 0 <= e < ne) for l, e in pairs):
        raise SystemExit(f"{path}: a ranked pair is out of range")
    return nl, ne, pairs


def coverage(ranked, freq, slots):
    """Share of the counted selections the first `slots` ranked pairs would have served."""
    total = sum(freq.values())
    return sum(freq.get(p, 0.0) for p in ranked[:slots]) / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="NautiBurn histogram database (SQLite)")
    src.add_argument("--csv", help="CSV with layer, expert and count columns")
    ap.add_argument("--model-id", help="with --db: the model whose counts to rank")
    ap.add_argument("--class", dest="classes", action="append", help="with --db: only this request class (repeat)")
    ap.add_argument("--phase", choices=("decode", "prefill", "all"), default="decode")
    ap.add_argument("--prefill-weight", type=float, default=0.25, help="with --phase all: weight of a prefill count")
    ap.add_argument("--n-layers", type=int, help="MoE layers (required with --csv; overrides the database)")
    ap.add_argument("--n-expert", type=int, help="experts per layer (required with --csv; overrides the database)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=("counts", "strp"),
                    help="counts: strata-poly's 'layer expert count' text; strp: the Qwen3.8-Flash-Next engine's "
                         "binary (default: .txt is counts, anything else strp)")
    ap.add_argument("--coverage", type=int, action="append", default=[], help="print coverage at N slots (repeat)")
    a = ap.parse_args()

    if a.db:
        if not a.model_id:
            ap.error("--db needs --model-id")
        nl, ne, freq = read_db(a.db, a.model_id, a.classes, a.phase, a.prefill_weight)
    else:
        if not (a.n_layers and a.n_expert):
            ap.error("--csv needs --n-layers and --n-expert")
        nl, ne, freq = a.n_layers, a.n_expert, read_csv(a.csv)
    nl, ne = a.n_layers or nl, a.n_expert or ne
    if not freq:
        raise SystemExit("no counts matched: check --model-id, --class and --phase")

    ranked, n_counted = rank_pairs(freq, nl, ne)
    fmt = a.format or ("counts" if a.out.endswith(".txt") else "strp")
    if fmt == "counts":
        write_counts(a.out, freq, nl, ne)
        back = read_counts(a.out)
        assert back == {p: round(c) for p, c in freq.items() if round(c) > 0 and p[0] < nl and p[1] < ne}, \
            "the counts did not survive the round trip"
        print(f"wrote {a.out}: {nl}x{ne}, {len(back)} counted pairs (strata-poly counts) from "
              f"{sum(freq.values()):,.0f} weighted selections")
    else:
        write_profile(a.out, ranked, nl, ne)
        assert read_profile(a.out) == (nl, ne, ranked), "the profile did not survive the round trip"
        print(f"wrote {a.out}: {nl}x{ne}, {len(ranked)} ranked pairs ({n_counted} counted, "
              f"{len(ranked) - n_counted} filled in) from {sum(freq.values()):,.0f} weighted selections")
    for n in a.coverage:
        print(f"  top {n} slots: {coverage(ranked, freq, n):.1%} of counted selections")


if __name__ == "__main__":
    main()
