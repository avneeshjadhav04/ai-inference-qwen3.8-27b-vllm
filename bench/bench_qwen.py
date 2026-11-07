#!/usr/bin/env python3
"""Portfolio-grade benchmark for Qwen3.8-27B-FP8 on vLLM (2x RTX 3090, TP=2).

Suites: latency | prefill | concurrency | cache | quality | all
Usage:  python3 bench_qwen.py [--base URL] [suite ...]
Stdlib only. Each latency case runs N reps; reports mean/min/max.
Results JSON written to stdout between markers + to results file.
"""
import json, sys, time, threading, urllib.request, re, statistics

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen3.8-27B-FP8"

# ---------------- HTTP / SSE ----------------
def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)

def stream_chat(messages, max_tokens=256, temperature=0.7, thinking=False,
                effort="xhigh", timeout=900):
    payload = {
        "model": MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
    }
    if thinking:
        payload["reasoning_effort"] = effort
    t_send = time.perf_counter()
    arrivals, usage = [], None
    first_reason = first_answer = None
    text, reas = "", ""
    resp = post("/v1/chat/completions", payload, timeout)
    event_lines = []
    while True:
        raw = resp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", "replace").strip()
        if line == "":
            if not event_lines:
                continue
            data = "\n".join(
                ln[5:].strip() for ln in event_lines if ln.startswith("data:")).strip()
            event_lines = []
            if data in ("[DONE]", ""):
                continue
            try:
                j = json.loads(data)
            except Exception:
                continue
            if j.get("usage"):
                usage = j["usage"]
            ch = (j.get("choices") or [{}])[0]
            d = ch.get("delta") or {}
            now = time.perf_counter()
            if d.get("reasoning_content") or d.get("reasoning"):
                arrivals.append((now, "reason"))
                reas += d.get("reasoning_content") or d.get("reasoning") or ""
                if first_reason is None:
                    first_reason = now
            if d.get("content"):
                arrivals.append((now, "answer"))
                text += d["content"]
                if first_answer is None:
                    first_answer = now
        elif line.startswith(":"):
            continue
        else:
            event_lines.append(line)
    t_end = time.perf_counter()
    return {"t_send": t_send, "t_end": t_end, "first_reason": first_reason,
            "first_answer": first_answer, "arrivals": arrivals,
            "usage": usage, "text": text, "reason": reas}

def summarize(name, r, rep=None):
    u = r["usage"] or {}
    comp = u.get("completion_tokens") or len(r["arrivals"])
    e2e = r["t_end"] - r["t_send"]
    gaps = sorted(b[0] - a[0] for a, b in zip(r["arrivals"], r["arrivals"][1:]))
    t_first = r["first_reason"] or r["first_answer"]
    n = len(r["arrivals"])
    out = {
        "test": name, "rep": rep,
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": comp,
        "reasoning_tokens": ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")),
        "e2e_s": round(e2e, 3),
        "ttft_reason_s": round(r["first_reason"] - r["t_send"], 3) if r["first_reason"] else None,
        "ttft_answer_s": round(r["first_answer"] - r["t_send"], 3) if r["first_answer"] else None,
        "tpot_s": round((r["arrivals"][-1][0] - t_first) / (n - 1), 4) if (t_first and n > 1) else None,
        "itl_p50_s": round(gaps[int(0.50 * (len(gaps) - 1))], 4) if gaps else None,
        "itl_p95_s": round(gaps[int(0.95 * (len(gaps) - 1))], 4) if gaps else None,
        "out_tps": round(comp / e2e, 2) if (comp and e2e) else None,
    }
    return out

def agg(name, runs):
    keys = ["e2e_s", "ttft_reason_s", "ttft_answer_s", "tpot_s",
            "itl_p50_s", "itl_p95_s", "out_tps"]
    d = {"test": name, "n": len(runs)}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        if vals:
            d[k] = {"mean": round(statistics.mean(vals), 4),
                    "min": round(min(vals), 4), "max": round(max(vals), 4)}
        else:
            d[k] = None
    d["prompt_tokens"] = runs[0]["prompt_tokens"]
    d["completion_tokens_mean"] = round(statistics.mean([r["completion_tokens"] for r in runs]), 1)
    return d

# ---------------- frozen prompts ----------------
PARA = ("Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs photons, "
        "splitting water and releasing oxygen while producing ATP and NADPH. The Calvin cycle then "
        "fixes carbon dioxide into glucose. Stomata regulate gas exchange, balancing CO2 intake against "
        "water loss through transpiration. ")

def sized_prompt(target_tokens):
    reps = max(1, int(target_tokens * 4 / len(PARA)))
    return PARA * reps

QUALITY = [
    {"id": "strict_ok", "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
     "max_tokens": 96, "temperature": 0.0, "thinking": False,
     "check": "exact_ok"},
    {"id": "bullets_only", "messages": [{"role": "user", "content": "Reply with ONLY a 3-bullet list of the primary colors, no other text."}],
     "max_tokens": 160, "temperature": 0.0, "thinking": False,
     "check": "three_bullets"},
    {"id": "math_17x23", "messages": [{"role": "user", "content": "Reply with ONLY the number: what is 17*23?"}],
     "max_tokens": 128, "temperature": 0.0, "thinking": False,
     "check": "number_391"},
    {"id": "math_pct", "messages": [{"role": "user", "content": "Reply with ONLY the number: what is 15% of 200?"}],
     "max_tokens": 128, "temperature": 0.0, "thinking": False,
     "check": "number_30"},
    {"id": "math_sqrt", "messages": [{"role": "user", "content": "Reply with ONLY the number: what is the square root of 1764?"}],
     "max_tokens": 128, "temperature": 0.0, "thinking": False,
     "check": "number_42"},
    {"id": "summary_keys", "messages": [{"role": "user", "content": PARA * 6 + "\nSummarize the above in one sentence."}],
     "max_tokens": 128, "temperature": 0.0, "thinking": False,
     "check": "keywords"},
]

def grade(item, text):
    t = (text or "").strip()
    c = item["check"]
    if c == "exact_ok":
        return ("pass" if t == "OK" else "fail", t[:80])
    if c == "three_bullets":
        lines = [ln for ln in t.splitlines() if ln.strip()]
        ok = len(lines) == 3 and all(ln.strip().startswith("-") for ln in lines)
        return ("pass" if ok else "fail", f"{len(lines)} lines: {t[:100]}")
    if c.startswith("number_"):
        want = c.split("_", 1)[1]
        nums = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
        got = nums[-1] if nums else None
        ok = got is not None and abs(float(got) - float(want)) < 1e-9
        return ("pass" if ok else "fail", f"got={got} want={want} raw={t[:80]}")
    if c == "keywords":
        low = t.lower()
        need = ["chlorophyll", "atp", "calvin", "stomata"]
        hit = [w for w in need if w in low]
        return ("pass" if len(hit) >= 3 else "fail", f"{len(hit)}/4 keywords {hit}")
    return ("info", t[:120])

CONC_POOL = [
    "Write a haiku about GPUs.", "Name 3 oceans.",
    "What is the capital of Japan? Reply with one word.",
    "Write a one-line Python hello world.",
    "Name the largest planet. One word.",
    "What is 9*9? Reply with ONLY the number.",
    "Write a haiku about rain.", "Name 2 prime numbers.",
]

# ---------------- suites ----------------
def suite_latency(reps=5):
    out = []
    cases = [
        ("short-nothink", [{"role": "user", "content": "Reply with exactly: OK"}],
         dict(max_tokens=16, temperature=0.0, thinking=False)),
        ("math-think", [{"role": "user", "content": "What is 17*23? Give the final number in a box."}],
         dict(max_tokens=256, temperature=1.0, thinking=True, effort="xhigh")),
        ("medium-1.5k", [{"role": "user", "content": sized_prompt(1500) + "\nSummarize the above in 3 bullets."}],
         dict(max_tokens=256, temperature=0.7, thinking=False)),
    ]
    for name, msgs, kw in cases:
        runs = []
        for i in range(reps):
            print(f"  {name} rep {i+1}/{reps}", flush=True)
            runs.append(summarize(name, stream_chat(msgs, **kw), rep=i))
        a = agg(name, runs)
        print(json.dumps(a, indent=1), flush=True)
        out.append(a)
    return out

def suite_prefill(reps=2):
    out = []
    for target in (500, 2000, 8000, 16000):
        runs = []
        for i in range(reps):
            print(f"  prefill-{target} rep {i+1}/{reps}", flush=True)
            msgs = [{"role": "user", "content": sized_prompt(target) + "\nReply with exactly: DONE"}]
            runs.append(summarize(f"prefill-{target}",
                                  stream_chat(msgs, max_tokens=16, temperature=0.0, thinking=False), rep=i))
        a = agg(f"prefill-{target}", runs)
        print(json.dumps(a, indent=1), flush=True)
        out.append(a)
    return out

def suite_concurrency():
    out = []
    for n in (1, 2, 4, 8):
        print(f"  concurrency-{n}", flush=True)
        res, errs = [], []
        t0 = time.perf_counter()
        def worker(p):
            try:
                res.append(stream_chat([{"role": "user", "content": p}],
                                       max_tokens=64, temperature=0.7, thinking=False))
            except Exception as e:
                errs.append(str(e))
        ths = [threading.Thread(target=worker, args=(CONC_POOL[i % len(CONC_POOL)],)) for i in range(n)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        wall = time.perf_counter() - t0
        tot_c = sum(((o["usage"] or {}).get("completion_tokens") or len(o["arrivals"])) for o in res)
        tot_p = sum(((o["usage"] or {}).get("prompt_tokens") or 0) for o in res)
        e2es = sorted(o["t_end"] - o["t_send"] for o in res)
        d = {"test": f"concurrency-{n}", "ok": len(res), "errors": errs,
             "wall_s": round(wall, 2), "total_prompt_tokens": tot_p,
             "total_completion_tokens": tot_c,
             "agg_out_tps": round(tot_c / wall, 2) if wall else None,
             "e2e_min_s": round(e2es[0], 3), "e2e_max_s": round(e2es[-1], 3)}
        print(json.dumps(d, indent=1), flush=True)
        out.append(d)
    return out

def suite_cache():
    out = []
    msg = [{"role": "user", "content": sized_prompt(1500) + "\nSummarize the above in 3 bullets."}]
    for i, label in enumerate(("cold", "warm1", "warm2")):
        print(f"  cache-{label}", flush=True)
        r = summarize(f"cache-{label}",
                      stream_chat(msg, max_tokens=256, temperature=0.7, thinking=False), rep=i)
        print(json.dumps(r, indent=1), flush=True)
        out.append(r)
    return out

def suite_quality():
    out = []
    for item in QUALITY:
        print(f"  quality-{item['id']}", flush=True)
        r = stream_chat(item["messages"], max_tokens=item["max_tokens"],
                        temperature=item["temperature"], thinking=item["thinking"])
        verdict, detail = grade(item, r["text"])
        d = {"test": "quality-" + item["id"], "verdict": verdict, "detail": detail,
             "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
             "completion_tokens": (r["usage"] or {}).get("completion_tokens"),
             "e2e_s": round(r["t_end"] - r["t_send"], 2)}
        print(json.dumps(d, indent=1), flush=True)
        out.append(d)
    return out

def main():
    global BASE
    args = [a for a in sys.argv[1:] if not a.startswith("--base")]
    for a in sys.argv[1:]:
        if a.startswith("--base="):
            BASE = a.split("=", 1)[1]
    if not args:
        args = ["all"]
    want = lambda *names: ("all" in args) or any(n in args for n in names)
    allres = {}
    if want("latency"):
        print("== latency ==", flush=True)
        allres["latency"] = suite_latency()
    if want("prefill"):
        print("== prefill ==", flush=True)
        allres["prefill"] = suite_prefill()
    if want("concurrency"):
        print("== concurrency ==", flush=True)
        allres["concurrency"] = suite_concurrency()
    if want("cache"):
        print("== cache ==", flush=True)
        allres["cache"] = suite_cache()
    if want("quality"):
        print("== quality ==", flush=True)
        allres["quality"] = suite_quality()
    print("RESULTS_JSON_START")
    print(json.dumps(allres))
    print("RESULTS_JSON_END")

if __name__ == "__main__":
    main()
