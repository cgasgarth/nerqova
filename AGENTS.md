# Nerqova

This repository builds a model-specific Apple Silicon inference runtime for the released Kev-4B checkpoint. The primary benchmark uses unchanged checkpoint weights. A smaller student model is separate research and cannot establish a runtime speedup.

## Commands

- Environment: `uv sync --extra serve --group dev` on Apple Silicon.
- Unit tests: `uv run python -m pytest tests -q -m 'not model'`.
- Model tests: `NERQOVA_TEST_RUN=<checkpoint> uv run python -m pytest tests -q -m model`.
- Stock timing: `uv run python scripts/bench_decisions.py --engine kev-mlx --out runs/kev-mlx-latency.json`.
- Nerqova timing: `uv run python scripts/bench_decisions.py --engine nerqova --out runs/nerqova-latency.json`.
- Compare: `uv run python scripts/compare_engines.py runs/kev-mlx-latency.json runs/nerqova-latency.json`.
- Development suites: `uv run python scripts/evaluate.py --engine <kev-mlx|nerqova> --out runs/<name>`.
- Serve: `uv run python -m nerqova.serve --run jaredpalmer/kev-4b --port 8009`.

## Evidence rules

- Compare the same resolved checkpoint, base revision, adapter/head hashes, temperature, and encoded request.
- Keep model-only, complete local decision, and HTTP timing separate. Benchmark both engines in separate quiet windows on the same Mac.
- Check every choice and maximum probability difference against stock MLX. Verify repeated prefix hits, same-length different states, reordered questions, unequal branches, and cache reuse after another state.
- Read only development partitions while selecting runtime changes. Keep the locked test closed until the final path is selected.
- Keep generated reports, model files, and traces under ignored `runs/`. Commit small verified evidence and explain its limits.
- Do not report a speedup from a smaller model as a same-weight runtime gain.
