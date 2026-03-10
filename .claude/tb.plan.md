<!-- 22597266-99b0-4fac-b1e9-61faefa8ee9c a7c1497f-af6a-444b-9cf8-dcfdcc4503b5 -->
# Plan: TB Normalize Enhancements

## Goals

- Persist the effective training configuration for each benchmark run alongside existing CSV/PNG outputs so experiments are reproducible.
- Introduce checkpointing that records completed `(environment, algorithm, seed)` combinations and partial metrics so interrupted runs can resume without redoing finished scenarios.

## Steps

1. **Config serialization** (`lir/gflownet/tb_normalize.py`)

- After CLI overrides are applied (before running benchmarks), capture the resolved CONFIG plus selected envs and timestamp.
- Save this dict as JSON next to the CSV/PNG outputs (matching timestamp prefix) whenever `run_benchmark` executes.

2. **Checkpoint data model**

- Decide on a checkpoint file (e.g., `<timestamp>_checkpoint.json`) containing:
- List/set of completed `(env, algorithm, seed)` tuples
- Accumulated records already written to disk so far (or path to aggregated CSV fragment)
- Ensure format supports fast lookup for resume.

3. **Resume logic** (`run_benchmark` loop)

- On startup, check for an existing checkpoint for the current timestamp (passed via CLI or auto-discovered); load completed tuples and skip them when iterating seeds.
- After each finished scenario, append records to disk (either incremental CSV append or in-memory to checkpoint) and update checkpoint file atomically so progress is durable.

4. **Finalization**

- Once all combinations finish, consolidate accumulated records into the final CSV/PNG (preserving current behavior) and mark checkpoint as complete (delete or flag), ensuring plots/CSV include all data collected during incremental saves.

5. **CLI adjustments & docs**

- Provide a way to target an existing checkpoint run (e.g., detect timestamp or accept `--resume-from` path) and document the new workflow in module docstring or argparse help message.

### To-dos

- [ ] Write effective config JSON next to outputs
- [ ] Design checkpoint file schema
- [ ] Skip completed scenarios via checkpoint
- [ ] Flush accumulated data and clean checkpoint
- [ ] Add CLI args/help for resume mode