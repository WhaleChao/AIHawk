"""TEST HARNESS ONLY: an OpenAI-compatible endpoint in front of OpenRouter that records what each request cost.

The memory systems the LongMemEval bench weighs (Mem0, Hindsight, Letta) call the model with their own clients; they
are pointed here instead of at OpenRouter, under a path that names the question:

    http://127.0.0.1:<port>/q/<question id>/api/v1/...   ->   https://openrouter.ai/api/v1/...

Every chat completion and embedding request is forwarded with OpenRouter's usage accounting asked for, and its cost,
tokens and kind are appended to <log>, one JSON line a request, so a system's spend per question is the sum of its
lines. A streamed answer is passed through as it comes and its cost read from the last chunk.

    python tests/bench/openrouter_meter.py --port 8790 --log /work/bench-jobs/meter.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

UPSTREAM = "https://openrouter.ai"


def record(log: Path, question: str, path: str, usage: dict | None, status: int) -> None:
    usage = usage or {}
    with log.open("a", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    "t": time.time(),
                    "question": question,
                    "kind": path.rsplit("/", 1)[-1],
                    "status": status,
                    "cost": usage.get("cost"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )
            + "\n"
        )


def make_app(log: Path, key: str) -> web.Application:
    session: dict[str, ClientSession] = {}

    async def start(app: web.Application) -> None:
        session["client"] = ClientSession(timeout=ClientTimeout(total=1800))

    async def stop(app: web.Application) -> None:
        await session["client"].close()

    async def forward(request: web.Request) -> web.StreamResponse:
        question = request.match_info["question"]
        path = "/" + request.match_info["rest"]
        body = await request.read()
        payload = None
        if body and request.method == "POST":
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
        if isinstance(payload, dict) and path.endswith(("/chat/completions", "/completions")):
            payload["usage"] = {"include": True}
            if payload.get("stream"):
                payload.setdefault("stream_options", {})["include_usage"] = True
            body = json.dumps(payload).encode()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": request.headers.get("Content-Type", "application/json")}
        async with session["client"].request(request.method, UPSTREAM + path, data=body or None, headers=headers, params=request.query) as upstream:
            if isinstance(payload, dict) and payload.get("stream"):
                response = web.StreamResponse(status=upstream.status, headers={"Content-Type": upstream.headers.get("Content-Type", "text/event-stream")})
                await response.prepare(request)
                usage = None
                async for line in upstream.content:
                    await response.write(line)
                    text = line.decode(errors="replace").strip()
                    if text.startswith("data:") and '"usage"' in text:
                        try:
                            usage = json.loads(text[5:]).get("usage") or usage
                        except ValueError:
                            pass
                await response.write_eof()
                record(log, question, path, usage, upstream.status)
                return response
            data = await upstream.read()
            usage = None
            try:
                usage = json.loads(data).get("usage")
            except (ValueError, AttributeError):
                pass
            if request.method == "POST":
                record(log, question, path, usage, upstream.status)
            return web.Response(status=upstream.status, body=data, content_type=upstream.headers.get("Content-Type", "application/json").split(";")[0])

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app.router.add_route("*", "/q/{question}/{rest:.*}", forward)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    key = Path(os.environ["E2E_OPENROUTER_KEY_FILE"]).read_text(encoding="utf-8").strip()
    web.run_app(make_app(Path(args.log), key), host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
