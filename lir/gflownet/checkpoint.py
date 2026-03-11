from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "RunState",
    "append_records",
    "build_effective_config",
    "prepare_run_state",
]


@dataclass
class RunState:
    """Container for bookkeeping files associated with a benchmark run."""

    run_id: str
    config_snapshot: dict[str, Any]
    config_hash: str
    config_path: Path
    checkpoint_path: Path
    partial_csv_path: Path
    csv_path: Path
    plot_path: Path
    completed: set[tuple[str, str, int]]
    created_at: str
    resumed: bool
    status: str = "in_progress"

    def persist_checkpoint(self, status: str | None = None) -> None:
        """Write the current checkpoint metadata to disk."""
        payload = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "config_snapshot": self.config_snapshot,
            "partial_csv": str(self.partial_csv_path),
            "completed": [
                {
                    "environment": env,
                    "algorithm": algo,
                    "seed": seed,
                }
                for env, algo, seed in sorted(self.completed)
            ],
            "created_at": self.created_at,
            "status": status or self.status,
            "updated_at": datetime.now().isoformat(),
        }
        _write_json_atomic(self.checkpoint_path, payload)
        self.status = payload["status"]


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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_effective_config(
    base_config: dict[str, Any],
    env_names: tuple[str, ...],
) -> dict[str, Any]:
    """Snapshot the configuration that uniquely identifies a benchmark run."""
    return {
        "config": copy.deepcopy(base_config),
        "envs": list(env_names),
    }


def _config_hash(snapshot: dict[str, Any]) -> str:
    serialized = json.dumps(snapshot, sort_keys=True).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()


def _deserialize_completed(
    combos: list[dict[str, Any]],
) -> set[tuple[str, str, int]]:
    completed: set[tuple[str, str, int]] = set()
    for entry in combos:
        completed.add(
            (
                str(entry["environment"]),
                str(entry["algorithm"]),
                int(entry["seed"]),
            )
        )
    return completed


def _auto_discover_checkpoint(
    results_dir: Path,
    config_hash: str,
) -> Path | None:
    """Find the most recent incomplete checkpoint matching the config hash."""
    candidates: list[tuple[float, Path]] = []
    if not results_dir.exists():
        return None
    for checkpoint_path in results_dir.glob("*_checkpoint.json"):
        try:
            data = _read_json(checkpoint_path)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("config_hash") != config_hash:
            continue
        if data.get("status") == "completed":
            continue
        try:
            mtime = checkpoint_path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, checkpoint_path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _resolve_path(candidate: str, fallback_dir: Path) -> Path:
    path = Path(candidate)
    if not path.is_absolute():
        path = (fallback_dir / path).resolve()
    return path


def _infer_completed_from_partial(
    partial_csv_path: Path,
    expected_iterations: int,
) -> set[tuple[str, str, int]]:
    if expected_iterations <= 0 or not partial_csv_path.exists():
        return set()
    try:
        df = pd.read_csv(
            partial_csv_path,
            usecols=["environment", "algorithm", "seed", "iteration"],
        )
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        return set()
    completed: set[tuple[str, str, int]] = set()
    grouped = (
        df.groupby(["environment", "algorithm", "seed"])["iteration"].max()
    )
    for (env, algo, seed), max_iter in grouped.items():
        if int(max_iter) >= expected_iterations:
            completed.add((str(env), str(algo), int(seed)))
    return completed


def _hydrate_completed_from_partial(
    state: RunState,
    expected_iterations: int,
) -> RunState:
    partial_completed = _infer_completed_from_partial(
        state.partial_csv_path,
        expected_iterations,
    )
    if partial_completed:
        state.completed |= partial_completed
    return state


def append_records(
    partial_csv_path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Append training records to the partial CSV."""
    if not records:
        return
    df = pd.DataFrame(records)
    file_exists = partial_csv_path.exists()
    df.to_csv(
        partial_csv_path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )


def prepare_run_state(
    results_dir: Path,
    config: dict[str, Any],
    env_names: tuple[str, ...],
    resume_from: Path | None = None,
) -> RunState:
    """Create or resume a RunState for the benchmark loop."""
    config_snapshot = build_effective_config(config, env_names)
    config_hash = _config_hash(config_snapshot)
    checkpoint_path = resume_from or _auto_discover_checkpoint(
        results_dir, config_hash
    )
    expected_iterations = int(
        config_snapshot.get("config", {}).get("n_iterations", 0)
    )

    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
        data = _read_json(checkpoint_path)
        if data.get("config_hash") != config_hash:
            raise ValueError(
                "Checkpoint configuration does not match the current settings."
            )
        run_id = str(data["run_id"])
        partial_csv = _resolve_path(
            str(data["partial_csv"]),
            checkpoint_path.parent,
        )
        csv_path = results_dir / f"{run_id}_results.csv"
        plot_path = results_dir / f"{run_id}_results.png"
        config_path = results_dir / f"{run_id}_config.json"
        created_at = str(data.get("created_at", datetime.now().isoformat()))
        completed = _deserialize_completed(data.get("completed", []))
        state = RunState(
            run_id=run_id,
            config_snapshot=data.get("config_snapshot", config_snapshot),
            config_hash=config_hash,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            partial_csv_path=partial_csv,
            csv_path=csv_path,
            plot_path=plot_path,
            completed=completed,
            created_at=created_at,
            resumed=True,
            status=str(data.get("status", "in_progress")),
        )
        return _hydrate_completed_from_partial(
            state,
            expected_iterations,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / f"{run_id}_config.json"
    checkpoint_path = results_dir / f"{run_id}_checkpoint.json"
    partial_csv_path = (results_dir / f"{run_id}_partial_results.csv").resolve()
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
    state = RunState(
        run_id=run_id,
        config_snapshot=config_snapshot,
        config_hash=config_hash,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        partial_csv_path=partial_csv_path,
        csv_path=csv_path,
        plot_path=plot_path,
        completed=set(),
        created_at=created_at,
        resumed=False,
    )
    state.persist_checkpoint()
    return _hydrate_completed_from_partial(
        state,
        expected_iterations,
    )


