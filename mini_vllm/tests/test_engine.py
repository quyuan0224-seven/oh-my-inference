"""mini_vllm 的最小回归测试。

重点不是覆盖率，而是验证三件事：
1. prefill + decode 的 KV cache 结果与 full forward 一致；
2. EngineCore 的 step loop 能生成并正确结束；
3. Scheduler 能按 FCFS 串行处理多个请求，并释放 KV cache。
"""

from __future__ import annotations

import unittest

from ..engine import (
    CharTokenizer,
    EngineCore,
    Request,
    SamplingParams,
    ScheduledRequest,
    SchedulerOutput,
    VOCAB_SIZE,
)
from ..model import KVCache, TinyAttentionModel


def make_model() -> TinyAttentionModel:
    return TinyAttentionModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=16,
        max_model_len=32,
        seed=0,
    )


class TestKVCache(unittest.TestCase):
    def test_prefill_and_decode_match_full_forward(self) -> None:
        tokenizer = CharTokenizer()
        model = make_model()
        prompt = tokenizer.encode("hello")

        cache = KVCache(hidden_size=model.hidden_size)
        prefill_logits = model.prefill(prompt, cache)
        full_logits = model.forward_full(prompt)

        for a, b in zip(prefill_logits, full_logits):
            self.assertAlmostEqual(a, b, places=10)

        next_token = prompt[0]
        decode_logits = model.decode(next_token, len(prompt), cache)
        full_next_logits = model.forward_full(prompt + [next_token])

        for a, b in zip(decode_logits, full_next_logits):
            self.assertAlmostEqual(a, b, places=10)


class TestContracts(unittest.TestCase):
    def test_empty_scheduler_output(self) -> None:
        output = SchedulerOutput(scheduled=[])

        self.assertTrue(output.is_empty)
        self.assertEqual(output.total_num_scheduled_tokens, 0)

    def test_scheduler_output_counts_scheduled_tokens(self) -> None:
        request = Request(
            request_id="req-x",
            prompt="hi",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=1, stop_chars=""),
        )
        cache = KVCache(hidden_size=16)
        scheduled = ScheduledRequest(
            request=request,
            phase="prefill",
            token_ids=[1, 2],
            start_pos=0,
            cache=cache,
            needs_logits=True,
        )

        output = SchedulerOutput(scheduled=[scheduled])

        self.assertFalse(output.is_empty)
        self.assertEqual(output.total_num_scheduled_tokens, 2)
        self.assertEqual(scheduled.num_new_tokens, 2)


class TestEngineCore(unittest.TestCase):
    def test_generate_is_deterministic_and_respects_max_tokens(self) -> None:
        tokenizer = CharTokenizer()
        params = SamplingParams(max_tokens=4, temperature=0.0, stop_chars="")

        engine_a = EngineCore(tokenizer=tokenizer, model=make_model(), seed=7)
        engine_b = EngineCore(tokenizer=tokenizer, model=make_model(), seed=7)

        out_a = engine_a.generate("hello", params)
        out_b = engine_b.generate("hello", params)

        self.assertEqual(out_a.output_token_ids, out_b.output_token_ids)
        self.assertEqual(len(out_a.output_token_ids), 4)
        self.assertEqual(out_a.finish_reason, "length")
        self.assertEqual(engine_a.kv_cache_manager.active_requests, 0)

    def test_generate_requires_an_idle_engine(self) -> None:
        tokenizer = CharTokenizer()
        engine = EngineCore(tokenizer=tokenizer, model=make_model(), seed=0)
        params = SamplingParams(max_tokens=2, temperature=0.0, stop_chars="")

        engine.add_request("the quick brown", params)

        with self.assertRaises(RuntimeError):
            engine.generate("hello", params)

    def test_scheduler_runs_queued_requests_serially(self) -> None:
        tokenizer = CharTokenizer()
        engine = EngineCore(tokenizer=tokenizer, model=make_model(), seed=0)
        params = SamplingParams(max_tokens=2, temperature=0.0, stop_chars="")

        engine.add_request("the quick brown", params)
        engine.add_request("hello", params)

        steps = []
        while engine.has_requests():
            results = engine.step()
            self.assertTrue(results)
            steps.extend(results)

        self.assertEqual(len(engine.finished_outputs), 2)
        self.assertTrue(all(o.finish_reason == "length" for o in engine.finished_outputs))
        self.assertEqual(steps[0].request_id, "req-000")
        self.assertEqual(steps[0].phase, "prefill")
        self.assertTrue(any(s.phase == "decode" for s in steps))
        self.assertEqual(engine.kv_cache_manager.active_requests, 0)

    def test_idle_step_returns_empty_list(self) -> None:
        tokenizer = CharTokenizer()
        engine = EngineCore(tokenizer=tokenizer, model=make_model(), seed=0)

        self.assertEqual(engine.step(), [])


if __name__ == "__main__":
    unittest.main()
