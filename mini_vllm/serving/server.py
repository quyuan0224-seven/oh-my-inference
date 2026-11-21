#!/usr/bin/env python3
"""mini_vllm 的最小 HTTP Server。

职责边界：
    HTTP 请求 -> Server -> EngineCore.generate() -> JSON 响应

mini_vllm 刻意做成单线程、同步、非流式：
- 一个请求处理完才处理下一个；
- 不做 token streaming；
- 不做 OpenAI 兼容协议；
- 不做并发调度。
这些是 V1 的扩展点，但 Server 这一层已经存在。
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from ..engine import CharTokenizer, EngineCore, SamplingParams, VOCAB_SIZE
from ..model import TinyAttentionModel


class InferenceHTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], engine: EngineCore) -> None:
        super().__init__(server_address, InferenceHandler)
        self.engine = engine


class InferenceHandler(BaseHTTPRequestHandler):
    server_version = "mini-inference-v0.1"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if urlparse(self.path).path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/generate":
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("empty request body")
            payload = json.loads(self.rfile.read(length))
            prompt = str(payload["prompt"])
            params = SamplingParams(
                max_tokens=int(payload.get("max_tokens", 20)),
                temperature=float(payload.get("temperature", 0.0)),
                stop_chars=str(payload.get("stop_chars", ".")),
            )
            result = self.server.engine.generate(prompt, params)  # type: ignore[attr-defined]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - demo-level safety net
            self._send_json(500, {"error": f"internal error: {exc}"})
            return

        self._send_json(
            200,
            {
                "request_id": result.request_id,
                "prompt": result.prompt,
                "text": result.output_text,
                "output_token_ids": result.output_token_ids,
                "finish_reason": result.finish_reason,
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        print(f"[server] {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini_vllm minimal HTTP inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = CharTokenizer()
    model = TinyAttentionModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=16,
        max_model_len=args.max_model_len,
        seed=0,
    )
    engine = EngineCore(tokenizer=tokenizer, model=model, seed=0)
    server = InferenceHTTPServer((args.host, args.port), engine)

    print(f"mini_vllm server listening on http://{args.host}:{args.port}")
    print("  GET  /health")
    print("  POST /generate  {\"prompt\": \"hello\", \"max_tokens\": 8}")
    print("note: single-threaded, blocking, non-streaming; outputs are random gibberish.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
