
import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .checks import run_all_checks
from .cleaning import clean_dataset
from .config import QAConfig
from .io import ArtifactManager, load_raw_dataset, save_dataset
from .plots import generate_all_plots
from .profiling import compare_profiles, profile_dataset
from .report import generate_qa_report


class QAPipeline:

    def __init__(self, config: QAConfig, artifacts_dir: Path, run_id: str | None = None):

        self.config = config
        self.artifact_manager = ArtifactManager(artifacts_dir, run_id)

        self.state: dict[str, Any] = {
            "config": config,
            "artifact_manager": self.artifact_manager,
        }

        self.llm = self._init_llm()

        self.pipeline_executor = self._create_pipeline()

    def _init_llm(self):

        system_message = """You are a data quality pipeline for electricity market data.
Your job is to orchestrate a deterministic QA pipeline:

1. Load and profile the raw dataset
2. Clean the dataset according to policy
3. Run QA checks on the cleaned dataset
4. Generate visualization plots
5. Write a comprehensive QA report

You must call tools in this order. Do NOT invent data - all metrics must come from tool calls.
Your role is to explain results and provide recommendations, not to compute statistics directly.

Be concise and systematic. Focus on actionable insights."""

        if self.config.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                api_key=api_key,
            ).bind(system=system_message)
        elif self.config.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def _create_pipeline(self):

        tools = [
            self._create_load_tool(),
            self._create_clean_tool(),
            self._create_qa_tool(),
            self._create_plots_tool(),
            self._create_report_tool(),
        ]

        return create_react_agent(self.llm, tools)

    def _create_load_tool(self):

        state = self.state

        @tool
        def load_raw_dataset_tool(input_path: str) -> str:
            """Load and profile a raw dataset from the given file path.
            
            Args:
                input_path: Path to the dataset file (CSV or Parquet)
                
            Returns:
                JSON string containing the dataset profile with shape, columns, date range, and memory usage.
            """
            state["artifact_manager"].log_step(
                "load_raw_dataset", input_path=input_path
            )

            try:
                df = load_raw_dataset(input_path)
                profile = profile_dataset(df)

                state["raw_df"] = df
                state["raw_profile"] = profile
                state["input_path"] = input_path

                return json.dumps(profile, indent=2, default=str)
            except Exception as e:
                return f"Error loading dataset: {str(e)}"

        return load_raw_dataset_tool

    def _create_clean_tool(self):

        state = self.state

        @tool
        def clean_dataset_tool() -> str:
            """Clean the raw dataset according to the configured cleaning policy.
            
            This tool applies timestamp cleaning, handles missing timestamps, coerces numeric types,
            and cleans column values. Must be called after load_raw_dataset_tool.
            
            Returns:
                JSON string containing cleaning summary with rows before/after, cleaning actions count,
                output paths, and profile comparison.
            """
            if "raw_df" not in state:
                return "Error: Must load raw dataset first"

            state["artifact_manager"].log_step("clean_dataset")

            try:
                config = state["config"]
                df_raw = state["raw_df"]

                df_clean, cleaning_log = clean_dataset(df_raw, config)

                clean_profile = profile_dataset(df_clean)

                comparison = compare_profiles(state["raw_profile"], clean_profile)

                output_dir = Path("data/processed")
                output_dir.mkdir(parents=True, exist_ok=True)

                clean_path_parquet = output_dir / "smard_clean.parquet"
                clean_path_csv = output_dir / "smard_clean.csv"

                save_dataset(df_clean, clean_path_parquet, format="parquet")
                save_dataset(df_clean, clean_path_csv, format="csv")

                state["clean_df"] = df_clean
                state["clean_profile"] = clean_profile
                state["cleaning_log"] = cleaning_log
                state["clean_path"] = str(clean_path_parquet)

                summary = {
                    "rows_before": state["raw_profile"]["shape"]["rows"],
                    "rows_after": clean_profile["shape"]["rows"],
                    "cleaning_actions": len(cleaning_log.to_dict()),
                    "output_paths": {
                        "parquet": str(clean_path_parquet),
                        "csv": str(clean_path_csv),
                    },
                    "comparison": comparison,
                }

                return json.dumps(summary, indent=2, default=str)
            except Exception as e:
                return f"Error cleaning dataset: {str(e)}"

        return clean_dataset_tool

    def _create_qa_tool(self):

        state = self.state

        @tool
        def run_qa_checks_tool() -> str:
            """Run all QA checks on the cleaned dataset.
            
            Performs schema validation, duplicate detection, DST handling checks, missing data checks,
            outlier detection, and consistency checks. Must be called after clean_dataset_tool.
            
            Returns:
                JSON string containing QA check results summary with total checks, pass/warn/fail counts,
                and details for each check.
            """
            if "clean_df" not in state:
                return "Error: Must clean dataset first"

            state["artifact_manager"].log_step("run_qa_checks")

            try:
                config = state["config"]
                df_clean = state["clean_df"]

                check_results = run_all_checks(df_clean, config)

                state["qa_results"] = check_results

                qa_results_dict = {
                    "checks": [r.to_dict() for r in check_results],
                    "cleaning_actions": state["cleaning_log"].to_dict(),
                    "dataset_profile": state["clean_profile"],
                }

                state["artifact_manager"].save_json(
                    qa_results_dict, "qa_results.json"
                )

                summary = {
                    "total_checks": len(check_results),
                    "pass": sum(1 for r in check_results if r.status == "PASS"),
                    "warn": sum(1 for r in check_results if r.status == "WARN"),
                    "fail": sum(1 for r in check_results if r.status == "FAIL"),
                    "checks": [
                        {
                            "id": r.id,
                            "name": r.name,
                            "status": r.status,
                            "message": r.message,
                        }
                        for r in check_results
                    ],
                }

                return json.dumps(summary, indent=2, default=str)
            except Exception as e:
                return f"Error running QA checks: {str(e)}"

        return run_qa_checks_tool

    def _create_plots_tool(self):

        state = self.state

        @tool
        def generate_plots_tool() -> str:
            """Generate diagnostic plots for the cleaned dataset.
            
            Creates visualization plots including missingness heatmap, price distribution,
            hourly seasonality, and correlation heatmap. Must be called after clean_dataset_tool.
            
            Returns:
                JSON string containing list of generated plot file paths.
            """
            if "clean_df" not in state:
                return "Error: Must clean dataset first"

            state["artifact_manager"].log_step("generate_plots")

            try:
                df_clean = state["clean_df"]
                artifacts_dir = state["artifact_manager"].run_dir

                plot_paths = generate_all_plots(df_clean, artifacts_dir)

                state["plot_paths"] = plot_paths

                return json.dumps({"plots": plot_paths}, indent=2)
            except Exception as e:
                return f"Error generating plots: {str(e)}"

        return generate_plots_tool

    def _create_report_tool(self):

        state = self.state

        @tool
        def write_report_tool() -> str:
            """Generate and save a comprehensive QA report.
            
            Creates a markdown report with dataset overview, check details, cleaning actions,
            recommendations, plots, and final verdict. Must be called after run_qa_checks_tool.
            
            Returns:
                String message indicating the report file path.
            """
            if "qa_results" not in state:
                return "Error: Must run QA checks first"

            state["artifact_manager"].log_step("write_report")

            try:
                report_content = generate_qa_report(
                    dataset_profile=state["clean_profile"],
                    qa_results=state["qa_results"],
                    cleaning_log=state["cleaning_log"],
                    plot_paths=state.get("plot_paths", []),
                )

                report_path = state["artifact_manager"].save_markdown(
                    report_content, "qa_report.md"
                )

                context_dir = Path("context")
                context_dir.mkdir(parents=True, exist_ok=True)
                with open(context_dir / "qa_report.md", "w") as ctx_f:
                    ctx_f.write(report_content)

                qa_context = {
                    "checks": [r.to_dict() for r in state["qa_results"]],
                    "cleaning_actions": state["cleaning_log"].to_dict(),
                    "dataset_profile": state["clean_profile"],
                }
                with open(context_dir / "qa_results.json", "w") as ctx_f:
                    json.dump(qa_context, ctx_f, indent=2, default=str)

                return f"Report generated at: {report_path}"
            except Exception as e:
                return f"Error writing report: {str(e)}"

        return write_report_tool

    def run(self, input_path: str) -> dict[str, Any]:

        self.artifact_manager.log_step("qa_pipeline_start", input_path=input_path)

        message = f"""Run the complete QA pipeline on the dataset at: {input_path}

Execute these steps in order:
1. Load and profile the raw dataset
2. Clean the dataset
3. Run QA checks
4. Generate plots (optional, skip if there are errors)
5. Write the QA report

Provide a final summary of the results."""

        try:
            result = self.pipeline_executor.invoke({"messages": [("user", message)]})

            training_ready = all(
                r.status != "FAIL" for r in self.state.get("qa_results", [])
            )

            final_message = ""
            if "messages" in result:
                final_message = result["messages"][-1].content if result["messages"] else ""

            final_result = {
                "success": True,
                "run_id": self.artifact_manager.run_id,
                "artifacts_dir": str(self.artifact_manager.run_dir),
                "training_ready": training_ready,
                "clean_dataset_path": self.state.get("clean_path"),
                "qa_report_path": str(
                    self.artifact_manager.get_path("qa_report.md")
                ),
                "qa_results_path": str(
                    self.artifact_manager.get_path("qa_results.json")
                ),
                "pipeline_output": final_message,
            }

            self.artifact_manager.log_step(
                "qa_pipeline_complete",
                **final_result,
            )

            return final_result

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "run_id": self.artifact_manager.run_id,
            }

            self.artifact_manager.log_step("qa_pipeline_error", error=str(e))

            return error_result
