# AGENTS.md

## 铁律第一条：测试前必查驻留进程，测试一律串行（2026-10-16 官家指定）

**跑任何测试前，先查看之前的程序是否还在驻留；有驻留就先停掉，再运行。所有测试一律串行执行，严禁并行。**

- 两台引擎 / 两个 `serve/server.py` / 两个测试进程同时驻留 = **并行**，即使请求是交替发的也不算串行 —— 单张卡上会互抢显存与算力，读数失真。
- 查驻留：`tasklist | grep -i strata-poly` + `netstat -ano | grep ":8081" | grep LISTEN` + `nvidia-smi --query-gpu=memory.used --format=csv,noheader`；属主不明时用 `Get-CimInstance Win32_Process` 看 `CommandLine`/`ParentProcessId` 定位整条进程链。
- 起新测试前**先杀旧**：引擎 → server → bash 包装，逐条 `taskkill //PID <pid> //F`，然后复查 GPU 回落到基线、端口已释放。
- **A/B 对比同样串行**：跑完 A 组彻底停掉（引擎 + server），再起 B 组；不许两个配置的引擎同时驻留轮换发请求。
- 踩坑背景：2026-10-16 A/B 期间两次并行（本机 8081 + 本移植树 8082 同时驻留），被官家两次纠正。

PolyStrata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

A second engine, `strata-poly` (`src/poly/`), runs GLM 5.3 Flash and three smaller models from their GGUF files, on
Linux with one NVIDIA card: [docs/GLM.md](docs/GLM.md), [docs/BACKENDS.md](docs/BACKENDS.md). The Qwen engine, the installer, the server and the web app come from
[Strata](https://github.com/Niko1221/Strata).

## Installing PolyStrata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, PolyStrata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Text taken from the Strata project names that project. After merging from it,
  `python tools/polystrata_names.py` writes PolyStrata again; its first lines say what it leaves alone.
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
