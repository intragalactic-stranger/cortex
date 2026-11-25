# GPU check: OK (2 x H100 80GB HBM3)

The automatic mismatch note written by the run was a false positive: `bench_app._check_gpu` did not parse the
multi-GPU request `H100!:2` (fixed in the next commit). The manifest lists two H100 80GB HBM3 devices and the engine
log shows `tensor_parallel_size=2`, `world_size=2`.
