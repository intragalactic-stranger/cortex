| Path | Expected | Actual | Result |
|---|---|---|---|
| unlabelled pod in ns cortex -> primary :8080/health | expect blocked | got 000 | PASS |
| pod in ns default -> primary :8080/health | expect blocked | got 000 | PASS |
| unlabelled pod in ns cortex -> gateway DB :5432 | expect blocked | got 000 | PASS |
| gateway -> primary /health | expect 200 | got 200 | PASS |
| gateway -> primary /v1/models without backend key | expect 401 | got 401 | PASS |
| gateway -> primary /metrics without backend key | expect 401 | got 401 | PASS |
| gateway -> primary /v1/models with backend key | expect 200 | got 200 | PASS |
| gateway -> internet (https://huggingface.co) | expect blocked | got 000 | PASS |
| primary pod -> internet (egress denied) | expect blocked | got 000 | PASS |
| primary pod -> gateway :4000 (egress denied) | expect blocked | got 000 | PASS |
failures: 0
