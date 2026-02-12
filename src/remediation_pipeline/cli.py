
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .pipeline import RemediationPipeline
from .config import RemediationConfig

console = Console()


@click.group()
def cli():

    pass


@cli.command()
@click.option(
    "--qa-results",
    "-q",
    required=True,
    type=click.Path(exists=True),
    help="Path to QA results JSON file",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to dataset to fix (Parquet or CSV)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for fixed dataset (auto-generated if not provided)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to remediation config YAML file",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Run identifier (auto-generated if not provided)",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(),
    default="artifacts/remediation",
    help="Base directory for artifacts",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be fixed without applying changes",
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Ask before applying each fix",
)
def fix(
    qa_results: str,
    dataset: str,
    output: str | None,
    config: str | None,
    run_id: str | None,
    artifacts_dir: str,
    dry_run: bool,
    interactive: bool,
):

    console.print("\n[bold blue]🔧 Remediation Pipeline Starting...[/bold blue]\n")

    try:
        if config:
            console.print(f"Loading config from: [cyan]{config}[/cyan]")
            remediation_config = RemediationConfig.from_yaml(config)
        else:
            console.print("Using default configuration")
            remediation_config = RemediationConfig.load()

        if dry_run:
            remediation_config.dry_run = True
        if interactive:
            remediation_config.interactive = True

    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[dim]Dry run: {remediation_config.dry_run}[/dim]")
    console.print(f"[dim]Interactive: {remediation_config.interactive}[/dim]")
    console.print(
        f"[dim]LLM: {remediation_config.llm_provider}/{remediation_config.llm_model}[/dim]\n"
    )

    try:
        pipeline = RemediationPipeline(
            config=remediation_config,
            artifacts_dir=Path(artifacts_dir),
            run_id=run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    except Exception as e:
        console.print(f"[bold red]Error initializing pipeline:[/bold red] {e}")
        sys.exit(1)

    console.print(
        f"[bold green]Running remediation on:[/bold green] {dataset}\n"
    )
    console.print(f"[bold cyan]Using QA results:[/bold cyan] {qa_results}\n")

    try:
        with console.status("[bold yellow]Processing...", spinner="dots"):
            result = pipeline.run(qa_results_path=qa_results, dataset_path=dataset)

        if not result.get("success"):
            console.print(
                f"\n[bold red]Remediation failed:[/bold red] {result.get('error')}\n"
            )
            if "traceback" in result:
                console.print(f"[dim]{result['traceback']}[/dim]")
            sys.exit(1)

        console.print("\n[bold green] Remediation completed![/bold green]\n")

        table = Table(title="Remediation Results")
        table.add_column("Item", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Run ID", result["run_id"])
        table.add_row("Fixes Applied", str(result["fixes_applied"]))
        table.add_row(
            "Training Ready",
            " YES" if result["training_ready"] else " NO",
        )
        table.add_row("Fixed Dataset", result.get("fixed_dataset_path", "N/A"))
        table.add_row("Remediation Report", result["remediation_report_path"])
        table.add_row("Artifacts Directory", result["artifacts_dir"])

        console.print(table)

        if result.get("pipeline_output"):
            console.print(f"\n[bold]Pipeline Summary:[/bold]")
            console.print(result["pipeline_output"])

        if result["training_ready"]:
            console.print(
                "\n[bold green]✨ Dataset is now training-ready![/bold green]\n"
            )
            sys.exit(0)
        else:
            console.print(
                "\n[bold yellow]⚠️  Dataset may require additional manual fixes. Check remediation report.[/bold yellow]\n"
            )
            sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n[bold red]Remediation interrupted by user[/bold red]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to raw dataset",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/processed/training_ready.parquet",
    help="Output path for training-ready dataset",
)
@click.option(
    "--max-iterations",
    type=int,
    default=3,
    help="Maximum QA → Fix → QA iterations",
)
@click.option(
    "--qa-config",
    type=click.Path(exists=True),
    default=None,
    help="Path to QA config YAML",
)
@click.option(
    "--remediation-config",
    type=click.Path(exists=True),
    default=None,
    help="Path to remediation config YAML",
)
@click.option(
    "--ai-summary",
    is_flag=True,
    help="Generate AI-powered executive summary and training recommendations",
)
def auto_pipeline(
    input_path: str,
    output: str,
    max_iterations: int,
    qa_config: str | None,
    remediation_config: str | None,
    ai_summary: bool,
):

    console.print(
        "\n[bold blue]🚀 Auto-Pipeline: Clean → QA → Remediation → QA[/bold blue]\n"
    )

    try:
        from qa_pipeline.checks import run_all_checks
        from qa_pipeline.cleaning import clean_dataset
        from qa_pipeline.config import QAConfig
        from qa_pipeline.io import load_raw_dataset, save_dataset
        from qa_pipeline.profiling import compare_profiles, profile_dataset
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] QA pipeline not found. Make sure it's installed."
        )
        sys.exit(1)

    from .issue_parser import IssueParser
    from .strategies import StrategyFactory
    from .tools.missing_data import MissingDataTools
    from .tools.time_series import TimeSeriesTools
    from .tools.validation import ValidationTools

    import json

    qa_cfg = QAConfig.load(qa_config) if qa_config else QAConfig()
    rem_cfg = (
        RemediationConfig.from_yaml(remediation_config)
        if remediation_config
        else RemediationConfig.load()
    )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(f"artifacts/auto_pipeline_{run_timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]QA timezone policy: {qa_cfg.timezone_policy}[/dim]")
    console.print(f"[dim]QA DST policy: {qa_cfg.dst_policy}[/dim]")

    report_data = {
        "run_timestamp": run_timestamp,
        "input_path": input_path,
        "output_path": output,
        "qa_config": {
            "timezone_policy": qa_cfg.timezone_policy,
            "dst_policy": qa_cfg.dst_policy,
            "frequency": qa_cfg.frequency,
        },
        "cleaning_actions": [],
        "raw_profile": {},
        "clean_profile": {},
        "pre_fix_checks": [],
        "remediation_actions": [],
        "post_fix_checks": [],
        "iterations": [],
    }

    console.print(f"\n[bold]Step 1:[/bold] Loading raw dataset: [cyan]{input_path}[/cyan]")
    try:
        df_raw = load_raw_dataset(input_path)
        console.print(f"  Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        sys.exit(1)

    report_data["raw_profile"] = profile_dataset(df_raw)

    console.print(f"\n[bold]Step 2:[/bold] Cleaning dataset...")
    try:
        df_clean, cleaning_log = clean_dataset(df_raw, qa_cfg)
        console.print(f"  Cleaned: {len(df_raw)} → {len(df_clean)} rows")
        console.print(f"  Cleaning actions: {len(cleaning_log.to_dict())}")
    except Exception as e:
        console.print(f"[bold red]Error cleaning dataset:[/bold red] {e}")
        sys.exit(1)

    report_data["cleaning_actions"] = cleaning_log.to_dict()
    report_data["clean_profile"] = profile_dataset(df_clean)

    clean_dir = Path("data/processed")
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_parquet = clean_dir / "smard_clean.parquet"
    clean_csv = clean_dir / "smard_clean.csv"
    save_dataset(df_clean, clean_parquet, format="parquet")
    save_dataset(df_clean, clean_csv, format="csv")
    console.print(f"  Saved cleaned data: [cyan]{clean_parquet}[/cyan]")

    missing_tools = MissingDataTools()
    ts_tools = TimeSeriesTools()
    validation_tools = ValidationTools()
    strategy_factory = StrategyFactory(rem_cfg, missing_tools, ts_tools, validation_tools)
    issue_parser = IssueParser()

    df_current = df_clean.copy()
    iteration = 0
    first_qa_checks = None

    while iteration < max_iterations:
        iteration += 1
        console.print(f"\n[bold cyan]═══ QA/Fix Iteration {iteration}/{max_iterations} ═══[/bold cyan]\n")

        iter_data = {"iteration": iteration, "fixes": [], "checks": []}

        console.print("[bold]Step 3:[/bold] Running QA checks...")
        try:
            check_results = run_all_checks(df_current, qa_cfg)
        except Exception as e:
            console.print(f"[bold red]Error running QA checks:[/bold red] {e}")
            sys.exit(1)

        if first_qa_checks is None:
            first_qa_checks = check_results
            report_data["pre_fix_checks"] = [r.to_dict() for r in check_results]

        iter_data["checks"] = [r.to_dict() for r in check_results]

        pass_count = sum(1 for r in check_results if r.status == "PASS")
        warn_count = sum(1 for r in check_results if r.status == "WARN")
        fail_count = sum(1 for r in check_results if r.status == "FAIL")

        for r in check_results:
            icon = {"PASS": "", "WARN": "⚠️ ", "FAIL": ""}[r.status]
            console.print(f"  {icon} {r.name}: {r.message}")

        console.print(f"\n  Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")

        training_ready = fail_count == 0
        if training_ready:
            console.print("\n[bold green] Dataset is training-ready![/bold green]")
            report_data["post_fix_checks"] = [r.to_dict() for r in check_results]
            report_data["iterations"].append(iter_data)
            break

        console.print("\n[bold]Step 4:[/bold] Analyzing issues and creating fix plan...")

        artifacts_dir = Path(f"artifacts/auto_pipeline_iter_{iteration}")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        qa_results_dict = {
            "checks": [r.to_dict() for r in check_results],
            "cleaning_actions": cleaning_log.to_dict(),
            "dataset_profile": profile_dataset(df_current),
        }
        qa_results_path = artifacts_dir / "qa_results.json"
        with open(qa_results_path, "w") as f:
            json.dump(qa_results_dict, f, indent=2, default=str)

        try:
            issues = issue_parser.parse_qa_results(qa_results_path)
        except Exception as e:
            console.print(f"[bold red]Error parsing issues:[/bold red] {e}")
            sys.exit(1)

        fixable_issues = [i for i in issues if i.fixable]
        console.print(f"  Found {len(issues)} issues, {len(fixable_issues)} fixable")

        if not fixable_issues:
            console.print("\n[bold yellow]⚠️  No fixable issues found. Remaining issues need manual review.[/bold yellow]")
            for issue in issues:
                console.print(f"  - {issue.name}: {issue.description} (fixable: {issue.fixable})")
            report_data["iterations"].append(iter_data)
            break

        plan = issue_parser.create_fix_plan(issues)
        execution_order = plan["execution_order"]
        console.print(f"  Fix plan: {len(execution_order)} strategies to apply")
        for strategy_name in execution_order:
            group_issues = plan["fix_groups"][strategy_name]
            console.print(f"    - {strategy_name}: {len(group_issues)} issue(s)")

        console.print(f"\n[bold]Step 5:[/bold] Applying fixes...")
        fixes_applied = 0
        fixes_failed = 0

        for strategy_name in execution_order:
            group_issues = plan["fix_groups"][strategy_name]
            for issue in group_issues:
                try:
                    df_after, fix_result = strategy_factory.apply_strategy(df_current, issue)

                    if "error" in fix_result:
                        console.print(f"  [yellow]⚠️  {issue.name}:[/yellow] {fix_result['error']}")
                        fixes_failed += 1
                        iter_data["fixes"].append({
                            "issue": issue.name,
                            "strategy": strategy_name,
                            "success": False,
                            "error": fix_result["error"],
                        })
                    else:
                        df_current = df_after
                        fixes_applied += 1
                        fix_summary = fix_result.get("fix_summary", {})
                        method = fix_summary.get("method", strategy_name)
                        imputed = fix_summary.get("imputed_count", fix_summary.get("rows_added", "N/A"))
                        console.print(f"  [green] {issue.name}:[/green] {method} (changed: {imputed})")
                        iter_data["fixes"].append({
                            "issue": issue.name,
                            "strategy": strategy_name,
                            "success": True,
                            "method": method,
                            "details": fix_summary,
                        })
                        report_data["remediation_actions"].append({
                            "issue": issue.name,
                            "strategy": strategy_name,
                            "method": method,
                            "details": fix_summary,
                            "iteration": iteration,
                        })
                except Exception as e:
                    console.print(f"  [red] {issue.name}:[/red] {e}")
                    fixes_failed += 1
                    iter_data["fixes"].append({
                        "issue": issue.name,
                        "strategy": strategy_name,
                        "success": False,
                        "error": str(e),
                    })

        console.print(f"\n  Applied {fixes_applied} fixes, {fixes_failed} failed")
        report_data["iterations"].append(iter_data)

        if fixes_applied == 0:
            console.print("\n[bold yellow]⚠️  No fixes could be applied. Breaking loop.[/bold yellow]")
            break

        save_dataset(df_current, clean_parquet, format="parquet")
        save_dataset(df_current, clean_csv, format="csv")

    if not training_ready:
        console.print(f"\n[bold cyan]═══ Final QA Check ═══[/bold cyan]\n")
        try:
            final_checks = run_all_checks(df_current, qa_cfg)
        except Exception as e:
            console.print(f"[bold red]Error in final QA:[/bold red] {e}")
            sys.exit(1)

        report_data["post_fix_checks"] = [r.to_dict() for r in final_checks]

        final_fail = sum(1 for r in final_checks if r.status == "FAIL")
        final_warn = sum(1 for r in final_checks if r.status == "WARN")
        final_pass = sum(1 for r in final_checks if r.status == "PASS")

        for r in final_checks:
            icon = {"PASS": "", "WARN": "⚠️ ", "FAIL": ""}[r.status]
            console.print(f"  {icon} {r.name}: {r.message}")

        console.print(f"\n  Final: {final_pass} PASS, {final_warn} WARN, {final_fail} FAIL")
        training_ready = final_fail == 0

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        csv_output = output_path.with_suffix(".csv")
        parquet_output = output_path
    else:
        csv_output = output_path
        parquet_output = output_path.with_suffix(".parquet")

    save_dataset(df_current, parquet_output, format="parquet")
    save_dataset(df_current, csv_output, format="csv")

    console.print(f"\n[bold]Training dataset saved:[/bold]")
    console.print(f"  Parquet: [cyan]{parquet_output}[/cyan]")
    console.print(f"  CSV:     [cyan]{csv_output}[/cyan]")

    final_profile = profile_dataset(df_current)
    report_data["final_profile"] = final_profile
    report_data["training_ready"] = training_ready

    report_md = _generate_pipeline_report(report_data)
    report_path = report_dir / "pipeline_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)

    with open(report_dir / "pipeline_results.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    console.print(f"\n[bold]Pipeline report:[/bold] [cyan]{report_path}[/cyan]")
    console.print(f"[bold]Results JSON:[/bold]  [cyan]{report_dir / 'pipeline_results.json'}[/cyan]")

    context_dir = Path("context")
    context_dir.mkdir(parents=True, exist_ok=True)

    with open(context_dir / "qa_remediation_report.md", "w") as f:
        f.write(report_md)

    with open(context_dir / "qa_remediation_results.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    with open(context_dir / "data_profile.json", "w") as f:
        json.dump(final_profile, f, indent=2, default=str)

    console.print(f"[bold]Context saved:[/bold]  [cyan]{context_dir.resolve()}[/cyan]")

    if ai_summary:
        console.print(f"\n[bold cyan]═══ AI Analysis ═══[/bold cyan]\n")
        console.print("[dim]Generating AI-powered executive summary...[/dim]")
        try:
            ai_section = _generate_ai_analysis(report_data, rem_cfg)
            if ai_section:
                with open(report_path, "a") as f:
                    f.write("\n" + ai_section)
                with open(context_dir / "qa_remediation_report.md", "a") as f:
                    f.write("\n" + ai_section)
                console.print("[bold green] AI analysis added to report[/bold green]")
                console.print(f"\n{ai_section}")
            else:
                console.print("[yellow]AI analysis returned empty - skipped[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠️  AI analysis failed (non-blocking): {e}[/yellow]")
            console.print("[dim]The deterministic report is still complete.[/dim]")

    if training_ready:
        console.print(f"\n[bold green] Dataset is training-ready![/bold green]\n")
        sys.exit(0)
    else:
        console.print(f"\n[bold yellow]⚠️  Dataset saved but has remaining FAIL check(s).[/bold yellow]\n")
        sys.exit(1)


def _generate_pipeline_report(data: dict) -> str:

    ts = data["run_timestamp"]
    training_ready = data.get("training_ready", False)

    report = f"""

**Run:** {ts}
**Input:** `{data['input_path']}`
**Output:** `{data['output_path']}`
**Status:** {'TRAINING READY' if training_ready else 'NOT READY'}

---

"""

    if training_ready:
        report += """The dataset has passed all QA checks and is ready for training.

**VERDICT: READY**
"""
    else:
        report += """Some QA checks still fail. The dataset **requires further review**.

**VERDICT: NOT READY**

Review the remaining FAIL checks above and address them manually.
"""

    import json
    import os

    from dotenv import load_dotenv

    load_dotenv()

    raw_profile = data.get("raw_profile", {})
    clean_profile = data.get("clean_profile", {})
    final_profile = data.get("final_profile", {})
    pre_checks = data.get("pre_fix_checks", [])
    post_checks = data.get("post_fix_checks", [])
    remediation_actions = data.get("remediation_actions", [])
    cleaning_actions = data.get("cleaning_actions", [])
    training_ready = data.get("training_ready", False)

    context = json.dumps(
        {
            "raw_dataset": {
                "rows": raw_profile.get("shape", {}).get("rows"),
                "columns": raw_profile.get("shape", {}).get("columns"),
                "date_range": raw_profile.get("date_range"),
                "missing_data_summary": {
                    col: info.get("missing_pct", 0)
                    for col, info in raw_profile.get("columns_detail", {}).items()
                    if info.get("missing_pct", 0) > 0
                },
            },
            "cleaning_actions": [a["action"] for a in cleaning_actions],
            "pre_fix_qa_checks": [
                {"name": c["name"], "status": c["status"], "message": c["message"]}
                for c in pre_checks
            ],
            "remediation_applied": [
                {
                    "issue": a["issue"],
                    "method": a["method"],
                    "details": {
                        k: v
                        for k, v in a.get("details", {}).items()
                        if k
                        in (
                            "before_missing",
                            "after_missing",
                            "imputed_count",
                            "calculated_count",
                            "rows_added",
                            "correlation",
                        )
                    },
                }
                for a in remediation_actions
            ],
            "post_fix_qa_checks": [
                {"name": c["name"], "status": c["status"], "message": c["message"]}
                for c in post_checks
            ],
            "final_dataset": {
                "rows": final_profile.get("shape", {}).get("rows"),
                "columns": final_profile.get("shape", {}).get("columns"),
                "remaining_missing": {
                    col: {
                        "count": info.get("missing_count", 0),
                        "pct": round(info.get("missing_pct", 0), 2),
                    }
                    for col, info in final_profile.get("columns_detail", {}).items()
                    if info.get("missing_count", 0) > 0
                },
                "column_stats": {
                    col: {
                        "min": info.get("min"),
                        "max": info.get("max"),
                        "mean": round(info.get("mean", 0), 2) if info.get("mean") is not None else None,
                    }
                    for col, info in final_profile.get("columns_detail", {}).items()
                    if col != "timestamp"
                },
            },
            "training_ready": training_ready,
        },
        indent=2,
        default=str,
    )

    prompt = f"""You are an expert data analyst specialising in European electricity
market data (day-ahead prices, generation mix, load). You have been given the
results of an automated data-quality pipeline that cleaned, checked, and
remediated a SMARD dataset.

Write a concise analysis in markdown (no top-level heading - it will be appended
to an existing report). Include EXACTLY these sections:

A 3-5 sentence narrative overview of the dataset quality journey: what came in,
what was wrong, what was fixed, and the final state. Contextualise the numbers
with domain knowledge (e.g. why negative prices exist, what drives missingness
patterns, why Q4-2018 data may be sparse).

Bullet-point observations that a data scientist training a price-forecasting
model should know. For example:
- Seasonal patterns and their implications
- Correlation between price and neighbouring-country prices
- Renewable generation composition
- Any remaining data quality caveats (remaining NaNs, imputation quality)

Actionable advice for using this dataset in an ML pipeline:
- Suggested train/test split strategy (time-based, walk-forward, etc.)
- Feature engineering ideas (lags, rolling stats, hour/weekday dummies, etc.)
- Potential pitfalls to watch for (data leakage, concept drift, regime changes)
- Whether remaining missing values need further treatment

Keep the tone professional but accessible. Be specific - reference actual column
names and numbers from the data.

Pipeline results:
{context}"""

    # Try to get LLM config from data, use defaults if not available
    llm_provider = data.get("llm_provider", "anthropic")
    llm_model = data.get("llm_model", "claude-sonnet-4-5-20250929")

    if llm_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return report  # Return report without AI summary if no API key
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=llm_model,
            temperature=0.3,
            api_key=api_key,
            max_tokens=2000,
        )
    elif llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return report  # Return report without AI summary if no API key
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=llm_model,
            temperature=0.3,
            api_key=api_key,
            max_tokens=2000,
        )
    else:
        return report  # Return report without AI summary if provider not supported

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="remediation_config.yaml",
    help="Output path for config file",
)
def init_config(output: str):

    output_path = Path(output)

    if output_path.exists():
        console.print(
            f"[bold yellow]Config file already exists:[/bold yellow] {output}"
        )
        if not click.confirm("Overwrite?"):
            console.print("Aborted.")
            return

    try:
        config = RemediationConfig()
        config.to_yaml(output_path)
        console.print(f"[bold green] Config file created:[/bold green] {output}")
    except Exception as e:
        console.print(f"[bold red]Error creating config:[/bold red] {e}")
        sys.exit(1)


@cli.command()
def version():

    from . import __version__

    console.print(
        f"Remediation Pipeline version: [bold cyan]{__version__}[/bold cyan]"
    )


def main():

    cli()


if __name__ == "__main__":
    main()
