"""serve/models.py - the models this installation has, and what each one is.

A model is a run config: a strata-<name>.json that setup wrote, with the engine, its arguments and the tokenizer.
The registry lists them.  What a model is comes from its own file where it has one (a GGUF says its architecture,
its size, its context and whether it carries a draft block), and models.json in the same folder may add to that
or correct it: an object of model names, each with any of the keys of an entry.

    {"qwen3.6-35b-a3b": {"display_name": "Qwen3.6 35B-A3B", "notes": "the daily one"}}

Nothing here starts or stops an engine: that is the server's (Service.switch_model).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# what an architecture is run by, and what the registry calls it
BACKENDS = {
    "glm5-next": "glm5-next", "glm5next": "glm5-next",
    "qwen35moe": "qwen35moe",
    "clover_moe": "clover_moe",
    "gemma4": "gemma4",
    "deepseek4": "deepseek4",
    "lfm2moe": "lfm2moe",
    "flint": "flint",
    "kolibri1": "kolibri1",
}
# a routed expert's tensors: they stay in the file and are held in VRAM as far as it has room
EXPERT = re.compile(r"\.ffn_(gate|up|down|gate_up)_exps\.weight$")
_seen: dict = {}   # (path, size, mtime) -> what the file says


def _arg(args: list, flag: str):
    """The value after `flag` in an engine's argument list, or None."""
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def _gguf(path: Path) -> dict:
    """What a GGUF says of itself: read once per file as it is now."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _seen:
        return _seen[key]
    from gguf_reader import GGUFFile
    g = GGUFFile(path)
    m = g.metadata
    arch = str(m.get("general.architecture") or "")
    dense = experts = 0
    for t in g.tensors:
        n = t.expected_bytes() or 0
        if EXPERT.search(t.name):
            experts += n
        else:
            dense += n
    # a split file's other parts hold most of the experts: the first part's sizes are then a lower bound
    parts = int(m.get("split.count") or 1)
    out = {
        "backend": BACKENDS.get(arch, arch or "unknown"),
        "runnable": arch in BACKENDS,
        "display_name": m.get("general.name") or m.get("general.basename"),
        "size_label": m.get("general.size_label"),
        "context_trained": m.get(arch + ".context_length"),
        "layers": m.get(arch + ".block_count"),
        "experts": m.get(arch + ".expert_count"),
        "experts_used": m.get(arch + ".expert_used_count"),
        "mtp": bool(m.get(arch + ".nextn_predict_layers")),
        "dense_bytes": dense, "expert_bytes": experts, "parts": parts,
        "types": g.by_type(),
    }
    _seen[key] = out
    return out


def _params(label) -> tuple[str | None, str | None]:
    """("35B", "3B") from a size label like "35B-A3B": all the parameters, and the ones a token uses."""
    m = re.match(r"^(\d+(?:\.\d+)?[BMK])(?:-A(\d+(?:\.\d+)?[BMK]))?", str(label or ""), re.I)
    return (m.group(1), m.group(2)) if m else (None, None)


def _quant(name: str, types: dict) -> str | None:
    """The quantization a file's name states (UD-Q3_K_M, 3.0bit, ...), or else its most common tensor type."""
    m = re.search(r"((?:UD-)?I?Q\d[A-Z0-9_]*|\d(?:\.\d)?bit[\w-]*|APEX[\w-]*|BF16|F16)", name, re.I)
    if m:
        return m.group(1)
    quantized = [t for t in types if t not in ("F32", "F16", "BF16")]
    return quantized[0] if quantized else None


def _aliases(value) -> list[str]:
    """The config's other names for the model (#297: a list, or one comma-separated string); [] for anything else."""
    if isinstance(value, str):
        value = value.split(",")
    return [x.strip() for x in value if isinstance(x, str) and x.strip()] if isinstance(value, list) else []


def describe(config: Path) -> dict | None:
    """The registry's entry for a run config, or None when the file is not one."""
    try:
        cfg = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or not cfg.get("exe") or not isinstance(cfg.get("args"), list):
        return None
    args = [str(a) for a in cfg["args"]]
    cwd = Path(cfg.get("cwd") or ROOT)
    model = _arg(args, "--model")
    path = (cwd / model) if model and not Path(model).is_absolute() else Path(model) if model else None
    exe = Path(cfg["exe"]) if Path(cfg["exe"]).is_absolute() else cwd / cfg["exe"]
    ctx = _arg(args, "--max-context") or _arg(args, "--ctx")
    e = {
        "id": str(cfg.get("model_name") or Path(config).stem.replace("strata-", "", 1)),
        "config": str(config),
        "display_name": None, "backend": "qwen4exp", "path": str(path) if path else None,
        "quantization": None, "params_total": None, "params_active": None,
        "context": int(ctx) if ctx and str(ctx).isdigit() else None,
        "multimodal": bool(cfg.get("vision")),
        "aliases": _aliases(cfg.get("aliases")),
        "mtp": bool(_arg(args, "--draft-model")) and "--no-draft" not in args,
        "ram_gb": None, "vram_gb": None,
        "status": "available", "problem": None,
    }
    tok = Path(cfg.get("tokenizer") or "")
    if not exe.exists():
        e["status"], e["problem"] = "unavailable", f"the engine is missing ({exe})"
    elif not (tok / "vocab.json").exists():
        e["status"], e["problem"] = "unavailable", f"the tokenizer is missing ({tok})"
    if path is not None and path.suffix == ".gguf":
        if not path.exists():
            e["status"], e["problem"] = "unavailable", f"the model file is missing ({path})"
        else:
            try:
                g = _gguf(path)
            except (OSError, ValueError, KeyError) as err:
                e["status"], e["problem"] = "unavailable", f"the model file cannot be read ({err})"
            else:
                e["backend"] = g["backend"]
                e["display_name"] = g["display_name"]
                e["params_total"], e["params_active"] = _params(g["size_label"])
                e["quantization"] = _quant(path.name, g["types"])
                e["mtp"] = (e["mtp"] or g["mtp"]) and "--no-draft" not in args
                if g["parts"] == 1:
                    # the file is read in place: all of it is file cache at most, and VRAM holds the dense
                    # weights whole and the experts as far as the card has room
                    e["ram_gb"] = round((g["dense_bytes"] + g["expert_bytes"]) / 1e9, 1)
                    e["vram_gb"] = {"least": round(g["dense_bytes"] / 1e9 + 1.0, 1),
                                    "all": round((g["dense_bytes"] + g["expert_bytes"]) / 1e9 + 1.0, 1)}
                if not g["runnable"]:
                    e["status"], e["problem"] = "unavailable", f"no engine here runs a '{g['backend']}' model"
    # the name setup gave it, else the file's own (a file's can be a converter's leftover), else the id
    e["display_name"] = cfg.get("display_name") or e["display_name"] or e["id"]
    return e


def discover(root: Path = ROOT) -> list[dict]:
    """Every model of the installation, by name.  Two configs of one name: the newer file is the model."""
    root = Path(root)
    try:
        extra = json.loads((root / "models.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        extra = {}
    out: dict[str, dict] = {}
    for p in sorted(root.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        e = describe(p)
        if e is None or e["id"] in out:
            continue
        more = extra.get(e["id"]) if isinstance(extra, dict) else None
        if isinstance(more, dict):
            e.update({k: v for k, v in more.items() if k not in ("id", "config", "status", "problem")})
        out[e["id"]] = e
    return sorted(out.values(), key=lambda e: e["id"])


def find(models: list[dict], name: str) -> dict | None:
    """A model by its name, by a config file's name, or by the one name that contains what was typed."""
    name = name.strip()
    low = name.lower()
    for e in models:
        if low in (e["id"].lower(), Path(e["config"]).name.lower(), Path(e["config"]).stem.lower()):
            return e
    close = [e for e in models if low and low in e["id"].lower()]
    return close[0] if len(close) == 1 else None


def resolve(models: list[dict], name) -> dict | None:
    """#38: the model a request's `model` names: its name, one of its aliases or its config file's name, exactly (case
    aside).  Never a part of a name, as find() takes: a client must not start a switch by a name that only looks
    like one of them."""
    if not isinstance(name, str) or not name.strip():
        return None
    low = name.strip().lower()
    for e in models:
        config = Path(e["config"])
        if low in (e["id"].lower(), config.name.lower(), config.stem.lower(), *(a.lower() for a in e.get("aliases", ()))):
            return e
    return None


def line(e: dict, current: str | None = None) -> str:
    """One line of a listing."""
    size = "-".join(x for x in (e["params_total"], "A" + e["params_active"] if e["params_active"] else None) if x)
    vram = e["vram_gb"]
    need = f"VRAM {vram['least']:g}-{vram['all']:g} GB, file {e['ram_gb']:g} GB" if vram else ""
    mark = "*" if e["id"] == current else " "
    what = ", ".join(x for x in (size, e["quantization"], e["backend"], f"context {e['context']}" if e["context"] else "",
                                 "images" if e["multimodal"] else "", "MTP" if e["mtp"] else "", need) if x)
    tail = "" if e["status"] != "unavailable" else f"  [{e['problem']}]"
    return f"{mark} {e['id']:28s} {what}{tail}"


def main() -> int:
    for e in discover():
        print(line(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
