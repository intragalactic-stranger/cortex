"""Seeded synthetic workloads with exact token lengths in the target model's tokenizer.

Prompts are built from a fixed English word list (realistic token statistics, unlike random token
ids) and trimmed to the exact token count. The chat template adds a small, constant number of
tokens on top; the engine-reported prompt length is recorded per request.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from cortex_serve.servebench.config import WorkloadSpec
from cortex_serve.servebench.tokenizer import Tokenizer

_WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "was",
    "for",
    "on",
    "that",
    "with",
    "as",
    "by",
    "at",
    "from",
    "it",
    "his",
    "an",
    "were",
    "are",
    "which",
    "this",
    "be",
    "or",
    "had",
    "not",
    "but",
    "have",
    "one",
    "their",
    "they",
    "has",
    "new",
    "first",
    "all",
    "been",
    "more",
    "who",
    "after",
    "two",
    "other",
    "also",
    "its",
    "time",
    "some",
    "into",
    "would",
    "over",
    "years",
    "only",
    "most",
    "many",
    "can",
    "year",
    "then",
    "these",
    "may",
    "such",
    "made",
    "used",
    "about",
    "between",
    "states",
    "when",
    "during",
    "under",
    "city",
    "world",
    "system",
    "part",
    "known",
    "later",
    "both",
    "three",
    "where",
    "through",
    "government",
    "like",
    "people",
    "work",
    "well",
    "while",
    "called",
    "before",
    "number",
    "state",
    "school",
    "since",
    "group",
    "team",
    "being",
    "war",
    "however",
    "state",
    "family",
    "early",
    "local",
    "because",
    "national",
    "water",
    "power",
    "music",
    "history",
    "series",
    "several",
    "each",
    "among",
    "market",
    "model",
    "data",
    "memory",
    "server",
    "request",
    "network",
    "energy",
    "design",
    "report",
    "value",
    "price",
    "research",
    "early",
    "river",
    "village",
    "station",
    "church",
    "court",
    "season",
    "league",
    "album",
    "film",
    "company",
    "university",
    "building",
    "army",
    "language",
    "support",
    "process",
    "program",
    "control",
    "service",
    "public",
    "result",
    "policy",
    "level",
    "region",
    "period",
    "form",
]


ULTRACHAT = (
    "HuggingFaceH4/ultrachat_200k",
    "8049631c405ae6576f93f445c6b8166f76f5505a",
    "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
)


@dataclass(frozen=True)
class Request:
    id: int
    workload: str
    prompt: str
    input_tokens: int
    max_tokens: int


def _text(rng: random.Random, n_words: int) -> str:
    words = [rng.choice(_WORDS) for _ in range(n_words)]
    # sentence-ish punctuation so tokenization looks like prose
    out = []
    for i, w in enumerate(words):
        out.append(w.capitalize() if i % 14 == 0 else w)
        if i % 14 == 13:
            out[-1] += "."
    return " ".join(out)


def exact_tokens(tok: Tokenizer, rng: random.Random, n_tokens: int) -> str:
    """Return text that encodes to exactly ``n_tokens`` tokens (or the closest achievable below)."""
    if n_tokens <= 0:
        return ""
    ids = tok.encode(_text(rng, n_tokens * 2 + 16))
    while len(ids) < n_tokens:
        ids += tok.encode(" " + _text(rng, n_tokens))
    text = tok.decode(ids[:n_tokens])
    # decode/encode can merge across the cut; trim until it fits
    while len(tok.encode(text)) > n_tokens:
        text = text[: text.rfind(" ")] if " " in text else text[:-1]
    return text


def real_prompts(rows: list[str], spec: WorkloadSpec, tok: Tokenizer, n: int, seed: int) -> list[Request]:
    """First-turn user prompts, shuffled with the seed; prompts longer than ``input_tokens`` are skipped."""
    order = list(range(len(rows)))
    random.Random(f"{seed}:{spec.name}").shuffle(order)
    out: list[Request] = []
    for i in order:
        n_in = len(tok.encode(rows[i]))
        if 0 < n_in <= spec.input_tokens:
            out.append(Request(id=len(out), workload=spec.name, prompt=rows[i], input_tokens=n_in,
                               max_tokens=spec.output_tokens))  # fmt: skip
            if len(out) == n:
                return out
    raise ValueError(f"only {len(out)} prompts <= {spec.input_tokens} tokens, need {n}")


def ultrachat_rows() -> list[str]:
    from cortex_serve.servebench.quality.datasets import fetch_parquet

    df = fetch_parquet(*ULTRACHAT)
    return [str(p) for p in df["prompt"].to_list()]


def generate(spec: WorkloadSpec, tok: Tokenizer, n: int, seed: int) -> list[Request]:
    if spec.source == "ultrachat":
        return real_prompts(ultrachat_rows(), spec, tok, n, seed)
    rng = random.Random(f"{seed}:{spec.name}")
    prefix = exact_tokens(tok, random.Random(f"{seed}:{spec.name}:prefix"), spec.shared_prefix_tokens)
    instr_tokens = len(tok.encode(spec.instruction))
    requests = []
    for i in range(n):
        target = spec.input_tokens
        if spec.input_jitter:
            target = round(target * rng.uniform(1 - spec.input_jitter, 1 + spec.input_jitter))
        body_tokens = max(1, target - instr_tokens - spec.shared_prefix_tokens)
        body = exact_tokens(tok, rng, body_tokens)
        # Instruction goes AFTER the unique body: a shared leading instruction would be reused from the
        # engine's prompt cache and bias TTFT. Only the explicit shared-prefix workload shares a prefix.
        prompt = "\n\n".join(p for p in (prefix, body, spec.instruction) if p)
        requests.append(
            Request(
                id=i,
                workload=spec.name,
                prompt=prompt,
                input_tokens=len(tok.encode(prompt)),
                max_tokens=spec.output_tokens,
            )
        )
    return requests
