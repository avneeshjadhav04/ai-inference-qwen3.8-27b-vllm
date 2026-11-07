# Benchmark methodology — Qwen3.8-27B-FP8 on 2× RTX 3090 (vLLM, TP=2)

## Setup under test
- Model: `Qwen/Qwen3.8-27B-FP8`, served as `Qwen3.8-27B-FP8`
- Engine: `vllm/vllm-openai:latest` (v0.32.0), `--tensor-parallel-size 2`
  `--max-model-len 32768 --max-num-seqs 128 --gpu-memory-utilization 0.93`
  `--trust-remote-code --reasoning-parser qwen3`
- Host: 2× RTX 3090 24GB (no P2P/NVLink), i7-10700F, 78GB RAM, Ubuntu 24.04
- Client: this script, run **on the VM** against `http://127.0.0.1:8000`
  (tunnel latency excluded on purpose — we measure serving, not networking)

## Metrics (per request, from SSE timestamps + `usage`)
- **TTFT-answer / TTFT-reason**: time to first `content` / `reasoning_content` delta.
- **TPOT / ITL p50/p95**: gaps between consecutive token-bearing SSE events.
- **E2E**: wall time from request send to stream end.
- **Throughput**: `completion_tokens / E2E` (single) or total / wall (concurrent).
- Token counts come from server `usage`; reasoning split from
  `completion_tokens_details.reasoning_tokens` (qwen3 parser).

## Suites (`python3 bench_qwen.py [suite ...]`, default `all`)
| Suite | What | Reps | ~Runtime |
|---|---|---|---|
| `latency` | short / math-think / 1.5k-context summary | 5 | ~5 min |
| `prefill` | fixed 16 out-tokens × 0.5k/2k/8k/16k in-tokens → TTFT curve | 2 | ~15 min |
| `concurrency` | N = 1/2/4/8 parallel shorts, 64 out-tokens | 1 | ~5 min |
| `cache` | same 1.5k prompt 3× back-to-back (cold/warm1/warm2) | 1 | ~1 min |
| `quality` | 6 frozen checks (strict format ×2, math ×3, keyword summary) | 1 | ~3 min |

## Reading the results
- `latency`: steady-state decode speed (TPOT flat ≈ 38ms ⇒ ~26 tok/s) and
  prefill scaling via TTFT.
- `prefill`: TTFT vs prompt tokens → prefill tok/s per length.
- `concurrency`: aggregate tok/s vs N (expect sub-linear: no GPU P2P on this box).
- `cache`: warm TTFT collapse ⇒ prefix-cache working; input $/1M warm ≈ $0.01.
- `quality`: `pass/fail` per check — strict-format checks validate the
  reasoning-parser fix (thinking must NOT leak into `content`).

## Limitations (stated openly)
- Single box, single engine version, temp=0–1.0 fixed per case.
- Prefill sweep uses synthetic repeated text (worst case for compression,
  best case for prefix cache — cache suite isolates that effect).
- No power-draw measurement: $/token uses flat $0.38/hr rental.
- Results JSON: `bench/results/<timestamp>.json` (this machine).

## Results — 2026-11-01 (`results/results-2026-11-01.json`)

### Latency (n=5; mean [min–max])
| Case | In / Out tok | TTFT-reason | TTFT-answer | TPOT | ITL p95 | E2E | tok/s |
|---|---|---|---|---|---|---|---|
| short, no-think | 57 / 16 | 0.94 [0.93–0.98] | —¹ | 37.5ms | 37.7ms | 1.50 [1.49–1.54] | 10.7 |
| math, think-xhigh | 69 / 64 | 1.07 [1.05–1.08] | 3.12 [2.40–3.66] | 38.4ms | 38.0ms | 3.44 [2.78–3.96] | 18.4 |
| summary, 1.1k ctx | 1127 / 135 | 5.19 [3.60–11.47]² | 7.67 [5.41–13.27] | 38.0ms | 38.0ms | 10.26 [7.90–15.65] | 14.0 |

¹ 16-token budget consumed by reasoning trace (see lesson below) — no answer tokens.
² First-rep cold effect; min shows warm steady state.

### Prefill curve (fixed 16 out-tokens, n=2; rep1 = cold, rep2 = warm — same prompt)
| Prompt tok | TTFT cold | TTFT warm | Speedup |
|---|---|---|---|
| 394 | 4.08s | 4.10s | — |
| 1,514 | 7.40s | 7.40s | — |
| 5,882 | 50.95s | 4.12s | **12.4×** |
| 11,706 | 62.32s | 7.42s | **8.4×** |

Warm = prefix-cache hit. Prefill is the entire TTFT story; decode is flat 38ms.

### Concurrency (64 out-tokens each)
| N | wall | agg tok/s | per-req E2E min–max |
|---|---|---|---|
| 1 | 3.31s | 19.3 | 3.31s |
| 2 | 5.04s | 21.3 | 4.13–5.04s |
| 4 | 8.27s | 19.8 | 7.07–8.27s |
| 8 | 12.40s | 30.4 | 10.51–12.40s |

No P2P between the 3090s caps TP scaling; N=8 wins on aggregate at the cost
of per-request latency. Single-stream decode (~26 tok/s) is the ceiling per request.

### Cache (same 1.1k prompt ×3)
Cold 3.55s → warm 3.60/3.67s (flat: the prompt was already cached by the
latency suite — itself evidence of cross-request prefix reuse).

### Quality (frozen set, temp=0) — 6/6 pass AFTER `--reasoning-parser qwen3`
| Check | Result | Detail |
|---|---|---|
| `Reply with exactly: OK` | pass | `OK` |
| ONLY 3-bullet list | pass | 3 lines |
| 17×23 / 15% of 200 / √1764 | pass | 391 / 30 / 42 |
| summary keywords | pass | 3/4 (chlorophyll, atp, stomata) |

**Before the parser:** strict-format checks failed (thinking leaked into `content`).
**Lesson logged:** with the parser, reasoning tokens count toward `max_tokens` —
first quality run starved answers (16-token budget, 0 answer tokens). Budgets raised.

### Cost at $0.38/hr (from measured rates)
| Stream | $/1M |
|---|---|
| Output (26 tok/s) | **$4.06** |
| Input, short ctx (~250 tok/s) | **$0.42** |
| Input, 8–16k ctx (~100 tok/s) | **$1.06** |
| Input, prefix-cache hit | **~$0.01** |
