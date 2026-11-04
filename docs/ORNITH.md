# Ornith 1.5 (`strata-poly`)

`strata-poly`, PolyStrata's multi-architecture GGUF engine, runs Ornith 1.5: both the compact `Ornith 1.5 35B-A3B`
and the larger `Ornith 1.5 122B` (`qwen35moe` architecture), read directly from their GGUF files. The server, the web app
and the OpenAI- and Anthropic-compatible APIs are the ones the README describes.

## The models

Ornith 1.5 is a hybrid mixture-of-experts model created by Ornith AI. PolyStrata runs two sizes:

- **Ornith 1.5 35B-A3B**: 35 billion total parameters, about 3 billion active per token across 40 trunk layers and
  256 routed experts (8 active per token) plus 1 shared expert. Quantized by mudler into single-file APEX GGUFs of
  14.4 to 23.7 GB.
- **Ornith 1.5 122B**: 122 billion total parameters, about 14 billion active per token across 48 trunk layers and
  256 routed experts (8 active per token) plus 1 shared expert. Quantized by Unsloth into GGUF files of 44.5 to 132.6 GB.

Both models include multi-token prediction (MTP): a speculative draft block embedded directly inside the model file
that guesses the next token ahead, increasing generation speed without needing an auxiliary draft model file.

## Architecture

Ornith 1.5 combines linear recurrent attention, full causal attention, routed mixture-of-experts feedforward networks,
and speculative drafting in a unified block structure:

### The 4-layer cycle (3 GDN : 1 full attention)

Unlike standard transformer architectures that run quadratic self-attention at every layer, Ornith 1.5 interleaves
linear recurrent attention and full causal attention in a repeating 4-layer cycle:

- **Layers 0, 1, 2**: Gated Delta Net (GDN) linear attention.
- **Layer 3**: Full causal self-attention.
- **Layers 4, 5, 6**: Gated Delta Net (GDN) linear attention.
- **Layer 7**: Full causal self-attention.
- ...and so on.

In every 4 layers, exactly 3 layers are linear recurrent attention and 1 layer is full attention.

- In **Ornith 1.5 35B-A3B** (40 trunk layers + 1 draft layer): 30 layers are GDN linear recurrent attention, and 10
  layers are full attention (plus 1 full attention draft layer, 11 total).
- In **Ornith 1.5 122B** (48 trunk layers + 1 draft layer): 36 layers are GDN linear recurrent attention, and 12
  layers are full attention (plus 1 full attention draft layer, 13 total).

Because 3 out of every 4 layers use the recurrent GDN mechanism instead of full attention, the memory required for the
attention cache is dramatically reduced:

1. **Recurrent state**: The GDN recurrent state maintains a fixed-size hidden state per sequence ($s_{\text{dim}} = 128$,
   $v_{\text{heads}} = 64$ on 122B, $v_{\text{heads}} = 32$ on 35B) with a short 1D convolution of 4 tokens (`conv4_silu`).
   This takes about 156 MB of RAM per active conversation session on 122B (and about 130 MB on 35B), regardless of context length.
2. **KV cache**: Full key-value cache rows are stored only for the 13 full attention layers (12 trunk + 1 draft). At fp16,
   one context token takes 26,624 bytes of VRAM on 122B (compared to over 100,000 bytes for a dense 49-layer model of similar width).

### Mixture-of-Experts (MoE)

Each layer routes token features through an MoE block:
- **Routed experts**: 256 total experts. A softmax router selects the top 8 experts for each token.
  On 122B, each expert has an intermediate SwiGLU feedforward dimension of 1024.
- **Shared expert**: 1 shared expert (intermediate dimension 1024) evaluated for every token and combined with the
  routed experts via a learned sigmoid gate.

### Multi-Token Prediction (MTP) draft block

Both models carry their own draft block for speculative token guessing:
- Integrated as the final block in the GGUF file (block index 40 on 35B, block index 48 on 122B).
- Identified by `qwen35moe.nextn_predict_layers = 1` and `blk.48.nextn.eh_proj.weight`.
- `strata-poly` automatically detects the embedded draft tensors and activates speculative decoding (`draft_ = true`).
  The model guesses candidate tokens from its draft layer while computing the trunk layers, accepting verified tokens
  at zero quality loss. `--no-draft` turns speculative drafting off if desired.

### Multimodal rotary embeddings (M-RoPE) and vision

- **RoPE**: Uses multimodal rotary position embeddings with sections `[11, 11, 10, 0]` (summing to 32 pairs / 64 dimensions),
  with base frequency $10,000,000$ and support for up to 262,144 tokens of context.
- **Vision encoder**: `mmproj-F16.gguf` (908 MB) provides vision understanding through 27 vision transformer blocks,
  projecting image patches into the 3072-dimensional text embedding space.

## What you need

For **Ornith 1.5 35B-A3B**:
- An NVIDIA graphics card with 8 GB or more of VRAM (or an AMD card on Linux).
- 18 to 28 GB of system RAM beside an 8 GB card.

For **Ornith 1.5 122B**:
- An NVIDIA graphics card with 16 GB or more of VRAM (RTX 3090, 4080, 4090, 5080, 5090, or professional cards;
  or an AMD card on Linux). Dense weights and buffers require about 8.5 GB of VRAM before caching experts, establishing
  a minimum card floor of 10.5 GB. 8 GB cards do not fit.
- System RAM:
  - Beside a 24 GB card: about 47 GB of free RAM for `UD-Q3_K_XL` (32 GB for `UD-Q2_K_XL`).
  - Beside a 16 GB card: about 55 GB of free RAM for `UD-Q3_K_XL` (41 GB for `UD-Q2_K_XL`).
- Disk space: 50 to 140 GB on a fast SSD.

## Available quantizations

Files for Ornith 1.5 122B are pinned to revision `3842d003341507c216295c6f28dfcd30349925c6` of
[`unsloth/Ornith-1.5-122B-GGUF`](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF):

| Size | Shards | Download | RAM beside a 24 GB card | Description |
| --- | :---: | :---: | :---: | --- |
| **UD-Q3_K_XL** | 3 | 59.8 GB | ~47 GB | The daily choice: highest quality under 60 GB |
| **UD-Q2_K_XL** | 1 | 44.5 GB | ~32 GB | The least-RAM choice: single file, fits 32 GB PCs |
| **UD-IQ3_XXS** | 1 | 49.5 GB | ~37 GB | Compact i-matrix quantization: single file |
| **UD-Q4_K_XL** | 3 | 79.7 GB | ~66 GB | 4-bit Unsloth Dynamic: needs 64 GB+ RAM |
| **Q8_0** | 4 | 132.6 GB | ~119 GB | 8-bit throughout: the closest to the model as trained, needs 128 GB+ RAM |
| **mmproj-F16.gguf** | 1 | 0.9 GB | — | Vision multimodal projector (optional, `--vision yes`) |

Quantizations larger than 50 GB are split across multiple files (`UD-Q3_K_XL` and `UD-Q4_K_XL` into 3 shards, `Q8_0` into 4 shards). Shard 1 is a 10.9 MB metadata
and tokenizer header with 0 weight tensors; the remaining shards carry the quantized model weights. Setup handles downloading,
verifying and configuring all shards automatically.

## Install and run

### Quick start with setup

To install Ornith 1.5 122B interactively:

```bash
# Windows
python setup.py --family ornith122

# Linux
./setup.sh --family ornith122
```

To install non-interactively with defaults:

```bash
python setup.py --family ornith122 --model UD-Q3_K_XL --yes --no-start
```

To install using files you already downloaded:

```bash
python setup.py --family ornith122 --gguf-dir /path/to/gguf/folder
```

### Running with PolyStrata

Once setup completes, start the model using the PolyStrata launcher:

```bash
python polystrata.py --model ornith122-ud-q3_k_xl
```

Or run the generated start script:

```bash
# Windows
run-ornith122-ud-q3_k_xl.bat

# Linux
./run-ornith122-ud-q3_k_xl.sh
```

### Switching models

In the web interface at `http://127.0.0.1:8080`, select **Ornith 1.5 122B** from the model dropdown.
Or switch via the API:

```bash
curl http://127.0.0.1:8080/model -d '{"model": "ornith122-ud-q3_k_xl"}'
```

## Chat format and tool calling

Ornith 1.5 uses the standard Qwen / Ornith chat template with reasoning tags and structured XML tool calling:

- **Reasoning**: The model outputs reasoning inside `<think>...</think>` tags. The server streams these as a separate
  reasoning channel in the API response.
- **Tool calls**: Functions are defined in the system prompt under a `# Tools` header. The model invokes a tool using:

```xml
<tool_call>
<function=function_name>
<parameter=parameter_name>
parameter_value
</parameter>
</function>
</tool_call>
```

- **Tool responses**: The client responds with the tool result wrapped in `<tool_response>` tags:

```xml
<tool_response>
{"status": "ok", "result": 42}
</tool_response>
```

The server frontend parses these tags incrementally so partial XML markers are never exposed to the client.
Quoted tags within messages are escaped using Unicode private-use markers to prevent prompt injection.
