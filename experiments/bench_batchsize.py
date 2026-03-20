#!/usr/bin/env python3
"""Quick benchmark: measure throughput at different batch sizes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
from lir.gflownet.tb_normalize import (
    CONFIG, _set_runtime_device, _build_env, _build_estimators,
    _instantiate_gflownet, _build_optimizer,
)
from gfn.utils.common import set_seed

_set_runtime_device("cuda")
CONFIG["height"] = 24
CONFIG["ndim"] = 4
set_seed(42)

env = _build_env("original")
pf, pb = _build_estimators(env)
gflownet = _instantiate_gflownet("TBGFlowNet", pf, pb)
optimizer = _build_optimizer(gflownet.pf_pb_parameters(), lr=1e-3)
logz_params = list(gflownet.logz_parameters())
if logz_params:
    optimizer.add_param_group({"params": logz_params, "lr": 1e-2})

# Warmup.
for _ in range(3):
    trajs = gflownet.sample_trajectories(env, n=256, save_logprobs=True)
    loss = gflownet.loss_from_trajectories(env, trajs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.cuda.synchronize()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'bs':>6}  {'s/iter':>8}  {'samples/s':>10}  {'peak_MB':>8}  {'equiv_iters_for_256x2500':>26}")
print("-" * 75)

for bs in [256, 512, 1024, 2048, 4096, 8192]:
    try:
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            trajs = gflownet.sample_trajectories(env, n=bs, save_logprobs=True)
            loss = gflownet.loss_from_trajectories(env, trajs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gflownet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        avg = sum(times) / len(times)
        samples_per_sec = bs / avg
        # How many iters at this bs to see same total samples as 256 * 2500?
        equiv_iters = int(256 * 2500 / bs)
        equiv_time = equiv_iters * avg
        print(
            f"{bs:6d}  {avg:8.3f}  {samples_per_sec:10.0f}  {peak_mb:8.0f}  "
            f"{equiv_iters:5d} iters = {equiv_time/60:5.1f} min"
        )
    except torch.cuda.OutOfMemoryError:
        print(f"{bs:6d}  OOM")
        torch.cuda.empty_cache()
