"""FastAPI wrapper exposing the agent over HTTP.

Run (single worker — asyncio handles concurrency, not processes):
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

Concurrency model
-----------------
Incoming requests are async (non-blocking). The sync LangGraph graph runs in a
ThreadPoolExecutor so it never blocks the event loop. An asyncio.Semaphore caps
how many threads are actually calling vLLM at once, keeping the vLLM queue
short and P95 stable.

Adaptive regulator
------------------
A background task polls vLLM /metrics every ADAPT_INTERVAL_S seconds and reads
vllm:num_requests_waiting. If the queue is growing it shrinks the semaphore
(backpressure); if it's empty it grows it (more throughput). The semaphore
value stays in [CONCURRENCY_MIN, CONCURRENCY_MAX].

To disable adaptation and use a fixed concurrency set ADAPT_INTERVAL_S=0.
"""
from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402

# ── concurrency knobs (tune via env, no code change needed) ──────────────────
CONCURRENCY_INIT = int(os.environ.get("AGENT_CONCURRENCY_INIT", "3"))
CONCURRENCY_MIN  = int(os.environ.get("AGENT_CONCURRENCY_MIN",  "2"))
CONCURRENCY_MAX  = int(os.environ.get("AGENT_CONCURRENCY_MAX",  "5"))
ADAPT_INTERVAL_S = float(os.environ.get("ADAPT_INTERVAL_S",     "5"))
VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://localhost:8000/metrics")
# grow when vLLM running < LOW_RUNNING, shrink when >= HIGH_RUNNING
VLLM_RUNNING_LOW  = int(os.environ.get("VLLM_RUNNING_LOW",  "3"))
VLLM_RUNNING_HIGH = int(os.environ.get("VLLM_RUNNING_HIGH", "6"))

# ── Langfuse: one handler per thread, created lazily ─────────────────────────
_lf_local = threading.local()
_lf_enabled = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)


def _get_lf_handler() -> Any:
    if not _lf_enabled:
        return None
    if not hasattr(_lf_local, "handler"):
        try:
            from langfuse.langchain import CallbackHandler
            _lf_local.handler = CallbackHandler()
        except Exception:
            _lf_local.handler = None
    return _lf_local.handler


# ── adaptive semaphore ────────────────────────────────────────────────────────
class AdaptiveSemaphore:
    """asyncio.Semaphore whose capacity can grow/shrink at runtime."""

    def __init__(self, initial: int) -> None:
        self._capacity = initial
        self._sem = asyncio.Semaphore(initial)

    async def acquire(self) -> None:
        await self._sem.acquire()

    def release(self) -> None:
        self._sem.release()

    async def __aenter__(self) -> "AdaptiveSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.release()

    def grow(self) -> None:
        if self._capacity < CONCURRENCY_MAX:
            self._capacity += 1
            self._sem.release()  # adds one extra permit

    def shrink(self) -> None:
        if self._capacity > CONCURRENCY_MIN:
            self._capacity -= 1
            # acquiring without await: only works if a permit is available;
            # if not, just skip — the natural drain will reduce in-flight count
            if self._sem._value > 0:  # noqa: SLF001
                self._sem._value -= 1  # noqa: SLF001

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_flight(self) -> int:
        return self._capacity - self._sem._value  # noqa: SLF001


_sem: AdaptiveSemaphore  # initialised in lifespan
_executor: ThreadPoolExecutor


def _parse_metric(text: str, name: str) -> int:
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            try:
                return int(float(line.split()[-1]))
            except ValueError:
                pass
    return 0


async def _adapt_loop() -> None:
    """Adjust semaphore based on vllm:num_requests_running.

    waiting stays 0 with continuous batching — vLLM accepts everything
    immediately. running is the real signal: if few requests are in flight
    on the GPU we can afford more concurrency; if many are running latency
    is already climbing and we should back off.
    """
    if ADAPT_INTERVAL_S <= 0:
        return
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            await asyncio.sleep(ADAPT_INTERVAL_S)
            try:
                r = await client.get(VLLM_METRICS_URL)
                running = _parse_metric(r.text, "vllm:num_requests_running")
                if running < VLLM_RUNNING_LOW and _sem.in_flight > 0:
                    _sem.grow()
                elif running >= VLLM_RUNNING_HIGH:
                    _sem.shrink()
            except Exception:
                pass  # vLLM unreachable → keep current capacity


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI()


@app.on_event("startup")
async def startup() -> None:
    global _sem, _executor
    _sem = AdaptiveSemaphore(CONCURRENCY_INIT)
    _executor = ThreadPoolExecutor(max_workers=CONCURRENCY_MAX)
    if ADAPT_INTERVAL_S > 0:
        asyncio.create_task(_adapt_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    _executor.shutdown(wait=False)


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/concurrency")
async def concurrency_status() -> dict[str, int]:
    """Expose current adaptive semaphore state for debugging."""
    return {"capacity": _sem.capacity, "in_flight": _sem.in_flight}


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    state = AgentState(question=req.question, db_id=req.db)
    handler = _get_lf_handler()
    config: dict[str, Any] = {
        "callbacks": [handler] if handler is not None else [],
        "metadata": req.tags,
    }

    loop = asyncio.get_event_loop()
    async with _sem:
        try:
            final = await loop.run_in_executor(
                _executor,
                lambda: graph.invoke(state, config=config),
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    if execution is None:
        return AnswerResponse(
            sql=sql, rows=None, iterations=iteration, ok=False,
            error="agent produced no execution result", history=history,
        )
    if not execution.ok:
        return AnswerResponse(
            sql=sql, rows=None, iterations=iteration, ok=False,
            error=execution.error, history=history,
        )
    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
