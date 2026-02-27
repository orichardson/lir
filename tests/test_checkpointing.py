from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lir.gflownet import checkpoint
from lir.gflownet import tb_normalize as tb_module


@pytest.mark.slow
def test_checkpoint_resume(tmp_path, monkeypatch):
    """Ensure checkpointed runs resume without duplicating or missing work."""
    tb = importlib.reload(tb_module)
    monkeypatch.setattr(tb, "RESULTS_DIR", tmp_path)

    original_config = copy.deepcopy(tb.CONFIG)
    try:
        # Force a tiny, deterministic benchmark configuration on CPU to keep the
        # checkpointing loop lightweight and device-stable during the test.
        tb.CONFIG.update(
            {
                "height": 16,
                "ndim": 2,
                "n_iterations": 3,
                "batch_size": 2,
                "validation_interval": 10,
                "validation_samples": 16,
                "n_seeds": 1,
                "show_progress": False,
                "device": "cpu",
            }
        )
        tb._set_runtime_device(tb.CONFIG["device"])
        envs = ("original",)
        # Snapshot the effective config and materialize an empty run state so we
        # can simulate a partially completed benchmark run.
        expected_snapshot = checkpoint.build_effective_config(tb.CONFIG, envs)

        run_state = checkpoint.prepare_run_state(
            tb.RESULTS_DIR,
            tb.CONFIG,
            envs,
        )
        first_algo = tb.ALGORITHM_ORDER[0]
        seed = 7

        # Execute one (env, algo, seed) combo, persist the intermediate records,
        # and pretend that the sweep was interrupted at this checkpoint.
        records = tb._train_single_run(envs[0], first_algo, seed)
        checkpoint.append_records(run_state.partial_csv_path, records)
        run_state.completed.add((envs[0], first_algo, seed))
        run_state.persist_checkpoint()

        # Resume the benchmark from disk and ensure the remaining combinations
        # run exactly once, then finalize artifacts.
        results_df, csv_path, plot_path = tb.run_benchmark(
            env_names=envs, resume_from=run_state.checkpoint_path
        )

        # Config/metadata files should reflect the snapshot we expect as of the
        # resumed run.
        config_payload = json.loads(Path(run_state.config_path).read_text())
        assert config_payload["config_snapshot"] == expected_snapshot

        checkpoint_payload = json.loads(
            Path(run_state.checkpoint_path).read_text()
        )
        assert checkpoint_payload["status"] == "completed"

        expected_rows = (
            len(tb.ALGORITHM_ORDER) * tb.CONFIG["n_seeds"] * tb.CONFIG["n_iterations"]
        )
        assert len(results_df) == expected_rows
        per_scenario = results_df.groupby(
            ["environment", "algorithm", "seed"]
        ).size()
        assert len(per_scenario) == len(tb.ALGORITHM_ORDER) * tb.CONFIG["n_seeds"]
        assert (per_scenario == tb.CONFIG["n_iterations"]).all()

        # CSV/plot outputs should exist, and the final CSV should match the
        # in-memory dataframe byte-for-byte.
        assert csv_path.exists()
        assert plot_path.exists()

        csv_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(results_df, csv_df)

    finally:
        tb.CONFIG.clear()
        tb.CONFIG.update(original_config)
