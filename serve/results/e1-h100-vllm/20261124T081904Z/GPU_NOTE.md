# This run executed on an NVIDIA H200 (143,771 MiB), not an H100

Modal served the `gpu="H100"` request on an H200 (automatic upgrade; billed at the H100 rate, see
`03_COMPUTE_AND_BUDGET.md`). The manifest's `host.gpu` records the real device. The data is valid **as an H200 point**
but must not be compared with `e1-h100-sglang` (which ran on a real H100 80GB HBM3). Configs now request `H100!`
(exact type, no upgrade) and vLLM was re-run on a real H100.
