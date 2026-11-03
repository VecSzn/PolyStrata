"""Metal poly defaults, setup flags and header-only memory estimates (no model payload reads)."""
from __future__ import annotations

import json
import struct
from pathlib import Path

from tools.gguf_reader import GGUFFile

Q8_FAMILIES = frozenset({"qwen36", "ornith", "tiel", "minicpmv"})


def add_arguments(parser):
    parser.add_argument("--metal-optimize", action="store_true", default=None,
                        help="enable optimized strata-poly Metal kernels (default)")
    parser.add_argument("--no-metal-optimize", dest="metal_optimize", action="store_false", default=None,
                        help="disable optimized strata-poly Metal kernels")
    parser.add_argument("--metal-memory", choices=("legacy", "balanced"), default=None,
                        help="strata-poly memory policy (default balanced); omitted keeps a saved setting")
    parser.add_argument("--poly-kv", choices=("fp16", "q8_0"), default=None,
                        help="strata-poly KV override; default Q8 for compatible Qwen35MoE models, otherwise FP16")


def requested(a):
    return any(getattr(a, key, None) is not None for key in ("metal_optimize", "metal_memory", "poly_kv"))


def validate(a, poly_families):
    if requested(a) and a.family not in poly_families:
        raise ValueError("--metal-optimize, --no-metal-optimize, --metal-memory and --poly-kv require a strata-poly --family")
    if getattr(a, "poly_kv", None) == "q8_0" and a.family not in Q8_FAMILIES:
        raise ValueError("--poly-kv q8_0 requires qwen36, ornith, tiel or minicpmv (Qwen35MoE)")


def read_config(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def flag_value(args, flag, default=None):
    for i, value in enumerate(args):
        if value == flag and i + 1 < len(args):
            return args[i + 1]
        if value.startswith(flag + "="):
            return value[len(flag) + 1:]
    return default


def apply_options(args, a, old):
    """Only these new controls survive a setup rerun; other setup rules stay unchanged."""
    previous = old.get("args", [])
    if not isinstance(previous, list) or not all(isinstance(x, str) for x in previous):
        previous = []
    optimize = getattr(a, "metal_optimize", None)
    if optimize is None:
        optimize = True
        for flag in previous:
            if flag in ("--metal-optimize", "--no-metal-optimize"):
                optimize = flag == "--metal-optimize"
    args.append("--metal-optimize" if optimize else "--no-metal-optimize")
    for attr, flag in (("metal_memory", "--metal-memory"), ("poly_kv", "--kv")):
        value = getattr(a, attr, None)
        if value is None:
            value = flag_value(previous, flag, "balanced" if attr == "metal_memory" else None)
        if value is not None:
            args.extend((flag, str(value)))


def read_headers(paths):
    """Unknown geometry is reported, never replaced by a catalog-size guess."""
    try:
        return [GGUFFile(Path(path)) for path in paths], None
    except (OSError, ValueError, KeyError, struct.error, OverflowError) as e:
        return [], f"GGUF header unavailable ({type(e).__name__})"


def estimate(headers, context, *, kv="fp16", memory="legacy", parallel=1, batch=None, draft=True):
    result = {"kv_initial_bytes": None, "kv_max_bytes": None, "recurrent_bytes": None,
              "working_bytes": None, "reason": "architecture geometry not estimated"}
    if not headers:
        result["reason"] = "GGUF geometry unavailable"
        return result
    meta = headers[0].metadata
    arch = meta.get("general.architecture")
    result["architecture"] = arch
    if arch != "qwen35moe":
        return result
    if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel < 1 or (
            batch is not None and (not isinstance(batch, int) or isinstance(batch, bool) or batch < 0)):
        result["reason"] = "parallel slot count is not a valid integer"
        return result
    # The engine has one base sequence plus N batch slots when parallel > 1.
    slots = 1 + (batch if batch is not None else parallel if parallel > 1 else 0)
    names = {t.name for header in headers for t in header.tensors}
    def number(key, default=None):
        n = meta.get(arch + "." + key, default)
        return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None
    count, nextn = number("block_count"), number("nextn_predict_layers", 0)
    heads, dim = number("attention.head_count_kv"), number("attention.key_length")
    if not count or nextn is None or nextn >= count or not heads or not dim or number("attention.value_length") != dim:
        result["reason"] = "Qwen35MoE KV geometry incomplete"
        return result
    if kv == "auto":
        kv = "q8_0" if dim % 32 == 0 else "fp16"
    result["kv_format"] = kv
    base = count - nextn
    draft_name = f"blk.{base}.nextn.eh_proj.weight"
    enabled = draft and draft_name in names and not any(
        f"blk.{base}.{key}" in names for key in ("nextn.embed_tokens.weight", "nextn.shared_head_head.weight"))
    layers = range(base + int(enabled))
    attn = [i for i in layers if f"blk.{i}.attn_k.weight" in names]
    gdn = [i for i in layers if f"blk.{i}.attn_qkv.weight" in names]
    if len(attn) + len(gdn) != len(layers) or kv not in ("fp16", "q8_0") or (kv == "q8_0" and dim % 32):
        result["reason"] = "Qwen35MoE layer or KV geometry incomplete"
        return result
    row = 2 * heads * (dim * 2 if kv == "fp16" else dim + (dim // 32) * 2)
    initial = min(context, 4096) if memory == "balanced" else context
    result.update(kv_initial_bytes=len(attn) * row * initial * slots,
                  kv_max_bytes=len(attn) * row * context * slots,
                  attention_layers=len(attn), sequence_count=slots, draft_enabled=enabled, reason=None)
    state, groups, rank, inner = (number(k) for k in ("ssm.state_size", "ssm.group_count", "ssm.time_step_rank", "ssm.inner_size"))
    if not gdn:
        result["recurrent_bytes"] = 0
    elif all(v is not None and v > 0 for v in (state, groups, rank, inner)):
        result["recurrent_bytes"] = len(gdn) * (3 * (2 * groups * state + inner) + rank * state * state) * 4 * slots
    return result


def report(result, say):
    def size(value):
        return "unknown" if value is None else f"{value / (1 << 30):.3f} GiB"
    say(f"  KV allocation estimate: initially {size(result['kv_initial_bytes'])}; "
        f"at maximum context {size(result['kv_max_bytes'])}")
    if result.get("kv_format"):
        say(f"  effective KV format: {result['kv_format']}")
    if result.get("sequence_count"):
        say(f"  includes {result['sequence_count']} sequence buffer(s) and "
            f"{'an enabled' if result['draft_enabled'] else 'no'} draft block")
    say(f"  recurrent sequence state: {size(result['recurrent_bytes'])}; "
        "additional working memory (scratch, speculative snapshots, caches): unknown")
    if result.get("reason"):
        say("  " + result["reason"])
    say("  These components are not a total memory requirement; weights, expert copies, "
        "the server and macOS also need memory.")
