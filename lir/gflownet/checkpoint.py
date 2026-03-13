from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "RunState",
    "build_effective_config",
    "prepare_run_state",
    "write_completed_checkpoint",
]


@dataclass
class RunState:
    """Container for bookkeeping files associated with a benchmark run."""

    run_id: str
    config_snapshot: dict[str, Any]
    config_hash: str
    config_path: Path
    checkpoint_path: Path
    csv_path: Path
    plot_path: Path
    created_at: str


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Serialize JSON to a temporary file then atomically move into place.

    Retries the rename a few times to tolerate transient shared-filesystem
    errors (e.g. NFS/Lustre ESTALE or cross-device rename failures).
    """
    import os
    import time

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            tmp_path.replace(path)
            return
        except OSError:
            if attempt == max_attempts - 1:
                # Last resort: non-atomic write directly to the target.
                with path.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, sort_keys=True)
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
                return
            time.sleep(0.1 * (2 ** attempt))


def build_effective_config(
    base_config: dict[str, Any],
    env_names: tuple[str, ...],
    algo_names: tuple[str, ...],
) -> dict[str, Any]:
    """Snapshot the configuration that uniquely identifies a benchmark run."""
    return {
        "config": copy.deepcopy(base_config),
        "envs": list(env_names),
        "algos": list(algo_names),
        "created_at": datetime.now().isoformat(),
    }


def _config_hash(snapshot: dict[str, Any]) -> str:
    serialized = json.dumps(snapshot, sort_keys=True).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()


def prepare_run_state(
    results_dir: Path,
    config: dict[str, Any],
    env_names: tuple[str, ...],
    algo_names: tuple[str, ...],
) -> RunState:
    """Create a fresh RunState for a benchmark run — no resume logic."""
    config_snapshot = build_effective_config(config, env_names, algo_names)
    config_hash = _config_hash(config_snapshot)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    config_path = results_dir / f"{run_id}_config.json"
    checkpoint_path = results_dir / f"{run_id}_checkpoint.json"
    csv_path = results_dir / f"{run_id}_results.csv"
    plot_path = results_dir / f"{run_id}_results.png"
    created_at = datetime.now().isoformat()

    config_payload = {
        "run_id": run_id,
        "config_hash": config_hash,
        "config_snapshot": config_snapshot,
        "created_at": created_at,
    }
    _write_json_atomic(config_path, config_payload)

    return RunState(
        run_id=run_id,
        config_snapshot=config_snapshot,
        config_hash=config_hash,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
        plot_path=plot_path,
        created_at=created_at,
    )


def write_completed_checkpoint(run_state: RunState) -> None:
    """Write the checkpoint JSON once at the end as a completion marker."""
    payload = {
        "run_id": run_state.run_id,
        "config_hash": run_state.config_hash,
        "config_snapshot": run_state.config_snapshot,
        "created_at": run_state.created_at,
        "status": "completed",
        "updated_at": datetime.now().isoformat(),
    }
    _write_json_atomic(run_state.checkpoint_path, payload)
