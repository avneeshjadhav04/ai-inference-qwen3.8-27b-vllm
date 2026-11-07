# Qwen3.8-27B-FP8 on 2x RTX 3090 (vLLM) — reproducible deploy + bench

Self-hosted inference for `Qwen/Qwen3.8-27B-FP8` with vLLM (tensor-parallel 2),
a no-dependency chat UI, and a stdlib-only benchmark. Benched on 2x RTX 3090
(no P2P/NVLink), i7-10700F, 78GB RAM, Ubuntu 24.04.

## Prereqs

- Ubuntu 24.04, root (`sudo`), 2x 24GB VRAM (script falls back to TP=N if fewer)
- NVIDIA driver + `nvidia-smi` working (Docker + nvidia-container-toolkit are auto-installed)
- ~60GB free disk (~30GB HF weights on first run + ~10GB image + vLLM compile cache)
- Ports 8000 (API) / 8080 (UI) free
- Benched image: `vllm/vllm-openai:v0.32.0`
  `sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`

## Quickstart

```bash
# 1. Deploy (from the VM, as root). Copy deploy/ to the VM first.
sudo bash deploy/deploy.sh
# LAN-only (no public tunnels):
# ENABLE_TUNNELS=0 sudo bash deploy.sh

# 2. Benchmark (run ON the VM, measures serving not networking)
python3 bench/bench_qwen.py all
# or single suites: latency | prefill | concurrency | cache | quality
# python3 bench/bench_qwen.py latency quality --base=http://127.0.0.1:8000
```

Endpoints: API `http://127.0.0.1:8000/v1` (model `Qwen3.8-27B-FP8`),
UI `http://127.0.0.1:8080/`. With tunnels on (default), the script prints
public `trycloudflare.com` URLs and pre-fills the UI backend field.

Heavy-data dirs default to `/var/lib/qwen/hf-cache`, `/var/lib/qwen/vllm-cache`,
`/opt/qwen-ui` and are overridable:
`HF_CACHE=/mnt/data/hf-cache VLLM_CACHE=/mnt/data/vllm-cache sudo -E bash deploy/deploy.sh`.
Use a native Linux filesystem with space — encrypted/overlay mounts fail as
Docker bind-mounts.

## What good numbers look like

From `bench/results/results-2026-11-01.json` (vLLM v0.32.0, TP=2,
`max-len 32768, max-seqs 128, gpu-util 0.93, --reasoning-parser qwen3`):

| Check | Expected |
|---|---|
| Decode (TPOT flat) | ~38ms (~26 tok/s) |
| TTFT short / 1.1k ctx | ~0.9s / ~3.6–5.2s warm |
| Prefill 8k/16k warm (prefix-cache hit) | 12.4x / 8.4x vs cold |
| Concurrency N=8 | ~30 agg tok/s, 10.5–12.4s E2E |
| Quality (frozen, temp=0) | 6/6 pass |
| Cost @ $0.38/hr | $4.06/1M out, $0.42 in-short, $1.06 in-long, ~$0.01 cached |

Full methodology: `bench/README.md`.

## Security warning

Default tunnels (`cloudflared quick`) expose API + UI on public URLs with
**no auth — anyone with the URL burns your GPU**. Fine for a demo; otherwise:

- Run `ENABLE_TUNNELS=0 sudo bash deploy.sh` and use
  `ssh -L 8000:127.0.0.1:8000 -L 8080:127.0.0.1:8080 <vm>`, or
- Front the tunnels with Cloudflare Access / Tailscale Serve.

## Layout

```
bench/bench_qwen.py    stdlib-only harness (latency/prefill/concurrency/cache/quality)
bench/README.md        methodology + 2026-11-01 results table
bench/results/         example run (keep one, gitignore the rest)
deploy/deploy.sh       idempotent deploy (pinned vLLM digest)
deploy/ui.html         chat UI (streaming, thinking toggle, images, export .md)
```
