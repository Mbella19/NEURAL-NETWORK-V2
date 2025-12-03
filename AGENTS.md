# Repository Guidelines

## Project Structure & Module Organization
- `config/settings.py` holds paths, hyperparameters, and device helpers; update `PathConfig.training_data_dir` if your raw EURUSD CSV lives elsewhere and run `ensure_dirs()` when adding new outputs.
- `scripts/run_pipeline.py` is the end-to-end entry point (data prep → analyst training → agent training → backtest). Intermediate artifacts land in `data/processed`, `models/analyst`, `models/agent`, and reports in `results/`.
- Core code lives in `src`: `data` (loading/resampling/features/normalization), `models` (analyst + fusion encoders), `training` (analyst/agent trainers and resume helpers), `agents/sniper_agent.py`, `environments/trading_env.py`, `evaluation` (backtesting/metrics), and `utils` (logging, visualization, metrics).
- `notebooks/` is for exploratory work; keep derived plots in `results/` rather than committing notebook outputs.

## Setup, Build, and Development Commands
- Python 3.10+ recommended. Create an env and install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Place the raw 1-minute CSV at the configured `training_data_dir` (defaults to `/Users/gervaciusjr/Desktop/Market data/Data`); adjust in `config/settings.py` if needed.
- Run the full workflow: `python scripts/run_pipeline.py`. This will process data, engineer features, train the analyst, freeze it, train the PPO agent, and backtest.
- Targeted runs: `python -m src.training.train_analyst` to retrain the analyst only, `python -m src.training.train_agent` to retrain the PPO agent using a saved analyst. Use smaller date slices for quick iterations.

## Coding Style & Naming Conventions
- Follow PEP8 with 4-space indentation and type hints. Use snake_case for functions/variables and PascalCase for classes. Keep configs as dataclasses to mirror existing patterns.
- Prefer structured logging (`src.utils.logging_config.get_logger`) over print statements; surface key metrics (loss, accuracy, win rate, PnL) at INFO level.
- Keep modules narrowly scoped (data/, models/, training/, evaluation/). Name files by role (`train_*.py`, `*_env.py`, `*_agent.py`).

## Testing & Validation
- No dedicated unit test suite yet; rely on integration runs. After changes, run at least `python scripts/run_pipeline.py` on a trimmed dataset to verify the full loop.
- Inspect outputs in `results/` (backtest metrics, plots) and model checkpoints in `models/` for regressions. Compare against `evaluation.backtest.compare_with_baseline` outputs to ensure the agent still beats buy-and-hold.
- When modifying feature engineering or data splits, spot-check `data/processed/*.parquet` to confirm alignment across timeframes.

## Commit & Pull Request Guidelines
- Commit messages follow the existing short, imperative pattern with a leading qualifier: e.g., `Fix: Pass correct AgentConfig to create_agent factory` or `Update Analyst Config: Shorter windows`.
- For PRs, include: what changed, key files touched, data range used for validation, before/after metrics (accuracy, win rate, PnL), and how to reproduce (`run_pipeline` args or script invocations). Attach plots/log snippets from `results/` when available.
- Do not commit raw market data or credentials; keep paths configurable via `config/settings.py` and document any local assumptions in the PR description.
