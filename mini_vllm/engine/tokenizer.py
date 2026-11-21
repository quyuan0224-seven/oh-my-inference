"""字符级 tokenizer。

mini_vllm 用一个字符一个 token 的最简实现，避免引入 BPE/SentencePiece。
它的职责和真实 tokenizer 一样：文本 <-> token id。
"""

from __future__ import annotations

from typing import Sequence

VOCAB = "abcdefghijklmnopqrstuvwxyz .,!?"
STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for i, ch in enumerate(VOCAB)}
SPACE_ID = STOI[" "]
VOCAB_SIZE = len(VOCAB)


class CharTokenizer:
    def encode(self, text: str) -> list[int]:
        # 未知字符统一映射为空格，保证 demo 不会因为输入字符而崩掉。
        ids = [STOI.get(ch, SPACE_ID) for ch in text]
        return ids or [SPACE_ID]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(ITOS.get(int(i), " ") for i in token_ids)

    def stop_token_ids(self, stop_chars: str) -> set[int]:
        return {STOI[ch] for ch in stop_chars if ch in STOI}
