# Invalid: Gemma 4 loglikelihood tasks run without <bos> (2026-11-24)

lm-eval 0.4.13's `local-completions` tokenizes loglikelihood requests with `add_special_tokens=False` unless
`add_bos_token=True` is passed. Gemma requires `<bos>`; without it ARC-Challenge fell to 24.7% (chance is 25%) and
WikiText word perplexity rose to 86. Generative tasks (chat template adds `<bos>`) and the BFSI perplexity (vLLM's
completions endpoint adds special tokens) are unaffected and stay in `quality-gpu/`. The loglikelihood tasks were
re-run with `--add-bos` (eval_app.py, commit 57ce02c). Qwen3 has no BOS requirement; its results are unaffected.
