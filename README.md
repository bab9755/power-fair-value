# European Power Fair Value

Forecasting Day-Ahead electricity prices in the German (DE/LU) market and translating hourly forecasts into a prompt curve view for trading desk use.

## What This Project Does

This prototype builds an end-to-end pipeline that:

1. **Ingests** publicly available hourly data from SMARD (Bundesnetzagentur) — day-ahead prices, load, and generation by fuel type
2. **Cleans** the data using an AI-orchestrated QA pipeline (LangChain ReAct agent backed by Claude) that validates, remediates, and produces a training-ready dataset
3. **Forecasts** next-day hourly DA prices using LightGBM (24 separate hour-of-day models), benchmarked against ElasticNet and two seasonal naive baselines
4. **Translates** hourly forecasts into daily block prices (Baseload, Peakload, Offpeak) following EPEX conventions, producing a fair-value curve view with directional signals and invalidation criteria

The final output is a trading guidance report showing how the forecast informs prompt curve positioning and when to distrust the model.

---

## Prerequisites

- **Python >= 3.12**
- **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- **Anthropic API key** (for the AI-driven QA pipeline — Claude)

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

After installation, restart your terminal or run `source ~/.bashrc` (or `~/.zshrc`).

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url> cobblestone
cd cobblestone
uv sync
```

This installs all dependencies from `pyproject.toml` + `uv.lock` into a local `.venv`.

### 2. Set your API key

Create a `.env` file in the project root (or export the variable):

```bash
echo 'ANTHROPIC_API_KEY=your-key-here' > .env
```

The QA and remediation pipelines use Claude via LangChain. The forecasting and DA-to-curve stages do **not** require an API key.

---

## Running the Pipeline

Execute each step in order. All commands assume you are in the project root.

### Step 1 — Data Ingestion

```bash
uv run python ingestion/ingest_smard_pandas.py
```

**What it does:** Reads raw SMARD CSVs from `data/raw/smard/` (market prices, consumption, generation — covering Oct 2018 to Feb 2026), merges them on timestamp, and produces a unified dataset.

**Outputs:**
- `data/processed/smard_unified.csv` — 64,609 hourly rows, 11 columns
- `data/processed/smard_unified.parquet` — same data in Parquet format

### Step 2 — QA Pipeline (requires API key)

```bash
uv run qa-pipeline run --input data/processed/smard_unified.csv
```

**What it does:** A LangChain ReAct agent orchestrates automated QA — schema validation, duplicate detection, DST handling (23/25-hour days), missing-data checks, outlier detection (bounds: -600 to +1500 EUR/MWh), and consistency checks. Produces a structured report and diagnostic plots.

**Outputs:**
- `data/processed/smard_clean.csv` — cleaned dataset
- `artifacts/{run_id}/qa_report.md` — full QA findings report
- `artifacts/{run_id}/qa_results.json` — structured QA results
- `artifacts/{run_id}/plots/` — diagnostic plots (missingness heatmap, price distribution, hourly seasonality, correlation heatmap)
- `artifacts/{run_id}/run_log.jsonl` — step-by-step LLM execution log (all prompts and outputs)

### Step 3 — Remediation Pipeline (requires API key)

```bash
uv run remediation-pipeline auto-pipeline --input data/processed/smard_unified.csv
```

**What it does:** Runs a full Clean → QA → Fix → QA loop: parses the dataset, applies cleaning, runs QA checks, plans fixes (interpolation for small gaps, forward-fill for DST hours, capping for extremes), applies them, and re-validates until the dataset passes all checks.

**Outputs:**
- `data/processed/training_ready.csv` — zero missing prices, verified temporal integrity
- `data/processed/training_ready.parquet` — same in Parquet format
- `artifacts/auto_pipeline_{timestamp}/pipeline_report.md` — full remediation report (before/after comparison, all fixes applied)
- `artifacts/auto_pipeline_{timestamp}/pipeline_results.json` — structured pipeline results
- `context/qa_remediation_report.md` — copy for downstream consumption

### Step 4 — Model Training

```bash
uv run python training/train.py --data data/processed/training_ready.csv --use_lgbm
```

**What it does:** Trains 24 hour-of-day models (one per delivery hour) using a strict chronological split — training on all data before Jan 2024, testing on Jan 2024 through Feb 2026 (18,575 hours). Models trained: LightGBM, ElasticNet, and two Seasonal Naive baselines. Features include price lags, rolling statistics, lagged fundamentals, and cyclic time encodings. All features use `shift(1)` to prevent look-ahead.

**Outputs:**
- `outputs/predictions.csv` — hourly out-of-sample predictions (y_true + y_pred per model)
- `outputs/model_metrics.csv` — MAE, RMSE, SMAPE, rMAE for each model
- `outputs/model_comparison.csv` — side-by-side model comparison

### Step 5 — Block Aggregation

```bash
uv run python blocks/blocks.py --mode combined
```

**What it does:** Aggregates hourly DA prices and forecasts into daily block products using EPEX conventions:
- **Baseload** (hours 1-24): flat average, all hours
- **Peakload** (hours 9-20): business hours, higher demand
- **Offpeak** (hours 1-8, 21-24): complement of peak

All timestamps aligned to CET/CEST delivery days with correct DST handling.

**Outputs:**
- `data/processed/daily_blocks.csv` — daily actuals + forecasts per block per model

### Step 6 — DA-to-Curve Translation

```bash
uv run python DA-to-curve/DA-to-curve.py
```

**What it does:** Translates block-level forecasts into a prompt curve fair-value view. Computes spreads between the model's fair value and a reference curve, generates directional signals (BUY/SELL/HOLD), evaluates invalidation criteria, and produces a trading guidance report.

**Outputs:**
- `outputs/da_to_curve/trading_guidance.md` — methodology, monthly fair-value view, usage guidance, and invalidation rules
- `outputs/da_to_curve/monthly_fair_value_curve.png` — monthly curve: model FV vs realised settlement (base + peak)
- `outputs/da_to_curve/spread_and_signals.png` — daily spread with directional signals
- `outputs/da_to_curve/block_prices_overview.png` — daily block prices: actual vs forecast vs reference
- `outputs/da_to_curve/daily_signals.csv` — full daily data with spreads, signals, conviction
- `outputs/da_to_curve/monthly_curve.csv` — monthly aggregation
- `outputs/da_to_curve/invalidation.json` — current invalidation criteria status

### (Optional) Step 7 — Interpretation Agent

```bash
uv run interpret run
```

**What it does:** Runs an AI interpretation agent that reads the DA-to-curve outputs (trading guidance, signals, backtest summary, invalidation status, and key plots), pulls in recent market context, and synthesizes everything into a research-style narrative.

**Outputs:**
- `outputs/interpretation_report.md` — human-readable interpretation of the strategy results, risks, and recommendations

---

## Results

### Forecasting Performance

| Model | MAE | RMSE | SMAPE | rMAE vs D-1 | rMAE vs D-7 |
|-------|-----|------|-------|-------------|-------------|
| **LightGBM** | **9.63** | **15.44** | **25.0%** | **0.36** | **0.30** |
| ElasticNet | 10.29 | 16.51 | 25.3% | 0.39 | 0.32 |
| Seasonal Naive D-1 | 26.59 | 42.00 | 48.8% | 1.00 | 0.82 |
| Seasonal Naive D-7 | 32.37 | 51.78 | 55.2% | 1.22 | 1.00 |

LightGBM reduces error by 64% versus the day-ahead naive baseline. Evaluated on 18,575 out-of-sample hours (Jan 2024 – Feb 2026).

### Prompt Curve View

The monthly fair-value view shows the model's baseload forecast averages within ~3 EUR/MWh of realised settlement across 26 out-of-sample months. This fair value maps directly to the German Power Baseload Calendar Month Future (CME/EEX), which settles to the average of hourly DA prices over the delivery month.

The trading guidance report (`outputs/da_to_curve/trading_guidance.md`) provides:
- How to use the forecast for prompt positioning (compare model FV against EEX forward prices)
- Three positioning strategies (DA-to-prompt, calendar spread, week-ahead)
- Four quantitative invalidation criteria with current status
- Practical considerations (transaction costs, position frequency, conviction scaling)

---

## Project Structure

```
cobblestone/
├── ingestion/
│   └── ingest_smard_pandas.py        # Step 1: raw SMARD → unified dataset
├── src/
│   ├── qa_pipeline/                  # Step 2: AI-orchestrated QA (LangChain + Claude)
│   ├── remediation_pipeline/         # Step 3: automated data remediation
│   └── interpretation_agent/         # AI agent: synthesizes results into research report
├── training/
│   └── train.py                      # Step 4: model training (LightGBM, ElasticNet, baselines)
├── blocks/
│   └── blocks.py                     # Step 5: hourly → daily block aggregation
├── DA-to-curve/
│   └── DA-to-curve.py                # Step 6: prompt curve fair-value view + guidance
├── data/
│   ├── raw/smard/                    # Raw SMARD downloads (market, consumption, generation)
│   └── processed/                    # Cleaned and processed datasets
├── outputs/
│   ├── predictions.csv               # Hourly out-of-sample forecasts
│   ├── model_metrics.csv             # Model performance metrics
│   ├── model_comparison.csv          # Side-by-side model comparison
│   └── da_to_curve/                  # Trading guidance, plots, signals
│       ├── trading_guidance.md       #   Main report (methodology, view, guidance, invalidation)
│       ├── monthly_fair_value_curve.png  #   Monthly curve: FV vs realised (base + peak)
│       ├── spread_and_signals.png    #   Daily spread with directional signals
│       ├── block_prices_overview.png #   Block prices: actual vs forecast vs reference
│       ├── daily_signals.csv         #   Full daily data
│       ├── monthly_curve.csv         #   Monthly aggregation
│       └── invalidation.json         #   Invalidation criteria status
├── artifacts/                        # QA pipeline outputs (created on run)
│   └── {run_id}/
│       ├── qa_report.md              #   QA findings report
│       ├── qa_results.json           #   Structured QA results
│       ├── run_log.jsonl             #   LLM execution log (prompts + outputs)
│       └── plots/                    #   Diagnostic plots
├── tests/                            # Unit tests
├── config.yaml                       # QA pipeline configuration
├── pyproject.toml                    # Dependencies (pinned with uv.lock)
├── SUBMISSION.md                     # 1-3 page project summary
└── project.txt                       # Original case study requirements
```

---

## Submission Checklist

| Requirement | Location |
|-------------|----------|
| Pipeline code | `ingestion/`, `src/`, `training/`, `blocks/`, `DA-to-curve/` |
| README | This file |
| Requirements | `pyproject.toml` + `uv.lock` |
| QA output | `artifacts/` (generated by Steps 2-3), `data/processed/smard_clean.csv` |
| Figures/tables | `outputs/da_to_curve/*.png`, `outputs/model_metrics.csv` |
| AI component | `src/qa_pipeline/`, `src/remediation_pipeline/`, `src/interpretation_agent/` (LLM-powered QA, remediation, and interpretation) |
| Submission document | `SUBMISSION.md` |
| Out-of-sample predictions | `outputs/predictions.csv` |

---

## AI/LLM Integration

This project uses **LangGraph ReAct agents for QA and remediation**, plus a separate LLM-based interpretation agent:

1. **QA Pipeline Agent** (`src/qa_pipeline/`) — orchestrates the data-quality workflow over deterministic tools. Rather than hard-coding a fixed script, the agent decides which tool to invoke next — load data, clean, run QA checks, generate plots, write report — based on intermediate results. A single CLI command (`qa-pipeline run`) replaces what would otherwise be a multi-step manual inspection process. All prompts and LLM outputs can be logged in JSONL format (`artifacts/*/run_log.jsonl`) for auditability.

2. **Remediation Pipeline Agent** (`src/remediation_pipeline/`) — when used via its LangGraph entry point, parses QA results, plans fixes, applies remediation strategies (gap filling, imputations, etc.), re-validates the dataset, and writes a remediation report. The `remediation-pipeline auto-pipeline` CLI exposes this logic in a reproducible, mostly deterministic way, while still allowing LLM-guided planning where appropriate.

3. **Interpretation Agent** (`src/interpretation_agent/`) — a senior energy-market analyst agent that reads pipeline outputs (trading guidance, signals, metrics, plots), analyzes the DA-to-curve results, and synthesizes them into a research-style interpretation report (`outputs/interpretation_report.md`). Run with:

   ```bash
   uv run interpret run
   ```

   The interpretation agent focuses on narrative synthesis; all core numeric work (ingestion, QA checks, remediation, training, backtests, plots) is done by deterministic Python code.

---

## Running Without an API Key

If you don't have an Anthropic API key, you can skip Steps 2-3 and use the pre-generated cleaned data:

```bash
uv sync
uv run python ingestion/ingest_smard_pandas.py
# Skip qa-pipeline and remediation-pipeline
uv run python training/train.py --data data/processed/training_ready.csv --use_lgbm
uv run python blocks/blocks.py --mode combined
uv run python DA-to-curve/DA-to-curve.py
```

The cleaned dataset (`training_ready.csv`) and all outputs are already committed to the repository.

---

## Tests

```bash
uv run pytest
```

Runs unit tests for QA checks, cleaning logic, configuration, and remediation strategies.
