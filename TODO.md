# TODO / scratchpad

Working notes for the assignment. Not graded — my own tracker.

## Ideas to try

- [ ] **Structured output for `verify_node`.** Use `llm().with_structured_output(Verdict)`
      with a Pydantic `Verdict(ok: bool, issue: str)`. Kills the `_extract_json`
      defensive parsing — verdict JSON becomes guaranteed valid at the decoder level.
      - vLLM: `guided_json` / `response_format` json_schema. Ollama: `format` param.
      - LangChain `with_structured_output` hides the backend difference → ports clean.
      - Only verify, NOT generate/revise (SQL is free text, fenced block is enough).
      - **Phase 6 lever:** guided decoding costs a bit of throughput/latency in vLLM.
        Measure "verify with structured output vs without" as one tuning iteration.

## Phase 1 / vLLM bring-up notes (H100)

- **OOM on default config.** Out of the box vLLM picks `max_model_len 262144`
  (256K). 30B weights eat ~60GB of the 80GB H100, KV cache for 256K context
  blows the remaining ~20GB → crash on startup. Fix in `start_vllm.sh`:
  - `--max-model-len 32768` — BIRD schema + question fit easily; shorter context
    buys KV room. (REPORT justification: less max-len → less KV → no OOM.)
  - `--gpu-memory-utilization 0.90` — give vLLM 90% of the card.
- **HF_TOKEN** wasn't reaching vLLM — start_vllm.sh runs vLLM directly and does
  NOT load .env. Script now sources .env (set -a) so HF_TOKEN exports. Without it:
  unauthenticated HF downloads (slower / rate-limited).
- **Docker perm denied** (`docker.sock permission denied`) — user not in docker
  group. Fix: `sudo usermod -aG docker $USER && newgrp docker`.
- Flags above are a STARTING point. Phase 6: tune them (max-num-seqs, max-len,
  thinking on/off, quant) and log what moved which metric.
- **transformers 5.x breaks vLLM 0.10.2** (`Qwen2Tokenizer has no attribute
  all_special_tokens_extended`). Must pin `transformers>=4.56,<5` in pyproject
  AND `uv lock && uv sync` — `uv run` re-syncs to lock, so pip install alone
  gets reverted. The pin must be in the VM's pyproject (was missing there).
- **FlashInfer not installed** → vLLM falls back to PyTorch-native top-p/top-k
  sampling. Installing flashinfer speeds sampling = a Phase 6 perf lever worth
  measuring (latency/throughput before vs after).

## Local → VM carry-over (don't forget on H100)

- [ ] Revert `.env` to defaults: `VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`,
      drop the ollama/openai override block (comment it back out).
- [ ] `.env` is currently messy — both openai AND ollama overrides uncommented,
      duplicate keys. Clean it: keep only the active backend.
- [ ] SSH port-forward 5 ports: 3000 Grafana, 9090 Prometheus, 3001 Langfuse,
      8000 vLLM, 8001 agent. (Or VSCode Remote-SSH Ports panel.)
- [ ] **Prompts need re-tuning on real 30B** — `/no_think` prompts tuned on 8B are
      a v1 starting point, not final. Final tuning = Phase 6 `eval_after_tuning`.
- [ ] **Parsing may chip on 30B** — `_extract_sql` / `_extract_json` are tuned to
      ollama's reply shape. 30B answers slightly differently → expect a small fix.
- [ ] Disable thinking on 30B too (Qwen3 thinking model). on/off is a P1 latency
      tradeoff AND a P6 lever — worth measuring.

## Local eval finding (qwen3:8b, NOT a deliverable)

Full 30-q run on ollama: overall 0.233 (7/30), per-iteration [0.233, 0.233, 0.233].
- 12 questions fired revise (>1 iter), **0 changed correctness** → loop adds zero
  value on 8B. 18 stopped at iter1 (verify ok), 11 hit the cap (3 iters), 1 at iter2.
- Harness verified correct. Real verify→revise value verdict must come from 30B.
- If 30B also shows flat per-iteration: dig into whether verify is too lenient
  (passes wrong answers) or revise just can't fix on this model. That analysis IS
  the Phase 5/7 "did the loop earn its keep" writeup.
- Saved to results/eval_local_qwen3-8b.json (delete or keep as a local reference;
  real baseline = results/eval_baseline.json from 30B).

## Phase progress

- [x] Phase 0 — setup, docker stack up (killed stale `mlops-hw-2` stack on :9090)
- [x] Phase 3 — graph wired (verify/revise/router + prompts), smoke-tested on ollama
- [x] Phase 5 (harness) — eval_one + summarize done, end-to-end tested on ollama.
      Real numbers must come from 30B on H100. Note: verify rarely fires revise on
      8B (max_iters=1 in mini run) — loop value shows on hard Qs / real model.
- [ ] Phase 1 — vLLM serving 30B on H100, pick flags + justify in REPORT (15%)
- [ ] Phase 2 — Grafana: latency/throughput/KV-cache panels reacting to load (15%)
      - local option: CPU-vLLM + Qwen3-0.6B (ollama gives no /metrics)
- [ ] Phase 4 — Langfuse traces, generate/verify/revise waterfall + metadata tags (5%)
- [ ] Phase 5 — finish `run_eval.py` (`eval_one`, `summarize`), baseline numbers (15%)
- [ ] Phase 6 — load test, find bottleneck on dashboard, fix, prove quality survived (25%)
- [ ] Phase 7 — REPORT.md ≤3 pages + all screenshots/results artifacts (15%)

## Deliverables checklist (Final table, README:406)

- [ ] REPORT.md (≤3 pages)
- [ ] infra/grafana/.../serving.json (all required panels)
- [x] agent/graph.py, agent/prompts.py
- [x] evals/run_eval.py
- [ ] results/eval_baseline.json, results/eval_after_tuning.json
- [ ] screenshots: vllm_manual_query, grafana_serving, langfuse_trace,
      langfuse_tags, grafana_eval_run, grafana_before, grafana_after
