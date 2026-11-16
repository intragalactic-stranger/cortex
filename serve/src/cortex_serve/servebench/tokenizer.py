"""Model tokenizers for engine-independent token counting.

Output tokens are re-counted with the model's own Hugging Face tokenizer so that every engine is
measured with the same ruler (spec §8 rule 2); engine-reported counts are logged alongside.
"""

from __future__ import annotations

from typing import Protocol

from tokenizers import Tokenizer as HFTokenizer

from cortex_serve.memcalc import hub


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class HubTokenizer:
    def __init__(self, repo: str, revision: str) -> None:
        cache = hub.CACHE_DIR.parent / "tokenizers" / f"{repo.replace('/', '__')}@{revision}.json"
        if not cache.exists():
            with hub.client() as client:
                data = hub._get(client, f"{hub.HF}/{repo}/resolve/{revision}/tokenizer.json")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        self._tok = HFTokenizer.from_file(str(cache))
        self.name = f"{repo}@{revision[:8]}"

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(ids))


class WhitespaceTokenizer:
    """Deterministic toy tokenizer for tests: one token per whitespace-separated word."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab)
                self._inv[self._vocab[word]] = word
            ids.append(self._vocab[word])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._inv[i] for i in ids)
