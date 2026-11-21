#!/usr/bin/env python3
"""mini_vllm 实验：观察 Scheduler 的 prefill/decode 步进和 KV cache。"""

from __future__ import annotations

import argparse

from ..engine import CharTokenizer, EngineCore, SamplingParams, VOCAB_SIZE
from ..model import TinyAttentionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini_vllm prefill/decode/KV-cache trace experiment")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="prompt; can be repeated. Defaults to two sample prompts.",
    )
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompts or ["the quick brown", "hello"]

    tokenizer = CharTokenizer()
    model = TinyAttentionModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=16,
        max_model_len=32,
        seed=0,
    )
    engine = EngineCore(tokenizer=tokenizer, model=model, seed=0)

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop_chars=".",
    )
    for prompt in prompts:
        engine.add_request(prompt, params)

    print("architecture: Server -> EngineCore -> [Scheduler, KVCacheManager, ModelRunner]")
    print("this experiment: EngineCore directly (HTTP Server is exercised by mini_vllm/serving/server.py)")
    print(f"queued: {len(prompts)} requests")
    print()
    print(
        f"{'step':>4} | {'request':<7} | {'phase':<7} | "
        f"{'token':^7} | {'computed':>8} | {'cached':>6} | output"
    )
    print("-" * 92)

    step = 0
    while engine.has_requests():
        results = engine.step()
        if not results:
            break
        for result in results:
            step += 1
            token = (result.token_char or "").replace("\n", "\\n")
            print(
                f"{step:>4} | {result.request_id:<7} | {result.phase:<7} | "
                f"{token:^7} | {result.num_computed_tokens:>8} | "
                f"{engine.kv_cache_manager.cached_tokens:>6} | "
                f"{result.output_text!r}"
            )

    print()
    print("final outputs:")
    for output in engine.finished_outputs:
        print(
            f"  {output.request_id}: prompt={output.prompt!r} "
            f"output={output.output_text!r} "
            f"finish={output.finish_reason} "
            f"tokens={output.output_token_ids}"
        )
    print()
    print(
        "note: outputs are random-weight gibberish; "
        "the point is the prefill/decode/KV-cache flow."
    )


if __name__ == "__main__":
    main()
