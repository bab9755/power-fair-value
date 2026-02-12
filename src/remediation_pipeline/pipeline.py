
import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .config import RemediationConfig
from .issue_parser import Issue, IssueParser
from .strategies import StrategyFactory
from .tools.missing_data import MissingDataTools
from .tools.time_series import TimeSeriesTools
from .tools.validation import ValidationTools

load_dotenv()


class RemediationPipeline:

    def __init__(
        self,
        config: RemediationConfig,
        artifacts_dir: Path,
        run_id: Optional[str] = None,
    ):

        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.run_id = run_id or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.artifacts_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.missing_tools = MissingDataTools()
        self.ts_tools = TimeSeriesTools()
        self.validation_tools = ValidationTools()

        self.strategy_factory = StrategyFactory(
            config,
            self.missing_tools,
            self.ts_tools,
            self.validation_tools,
        )

        self.issue_parser = IssueParser()

        self.state: dict[str, Any] = {
            "config": config,
            "artifacts_dir": self.run_dir,
            "strategy_factory": self.strategy_factory,
            "issue_parser": self.issue_parser,
            "remediation_log": [],
        }

        self.llm = self._init_llm()

        self.pipeline_executor = self._create_pipeline()

    def _init_llm(self):

        system_message = """You are a data remediation pipeline for electricity market data.
Your job is to automatically fix data quality issues identified by the QA pipeline.

Your workflow:
1. Parse QA results to identify issues
2. Create a prioritized fix plan
3. Apply fixes sequentially using appropriate strategies
4. Validate each fix to ensure no new issues are introduced
5. Generate a comprehensive remediation report

You have access to specialized tools for:
- Missing data imputation (correlation, interpolation, seasonal patterns)
- Time series gap filling
- DST transition handling
- Duplicate removal
- Validation

Be systematic and always validate after each fix. Provide clear explanations of what was fixed and how."""

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
            self._create_parse_tool(),
            self._create_plan_tool(),
            self._create_fix_tool(),
            self._create_validate_tool(),
            self._create_report_tool(),
        ]

        return create_react_agent(self.llm, tools)

    def _create_parse_tool(self):

        state = self.state

        @tool
        def parse_qa_results_tool(qa_results_path: str) -> str:

            try:
                issues = state["issue_parser"].parse_qa_results(qa_results_path)
                state["issues"] = issues

                summary = {
                    "total_issues": len(issues),
                    "by_severity": {},
                    "by_category": {},
                    "fixable_count": sum(1 for i in issues if i.fixable),
                    "issues": [
                        {
                            "id": issue.id,
                            "name": issue.name,
                            "severity": issue.severity.value,
                            "category": issue.category.value,
                            "fixable": issue.fixable,
                            "fix_strategy": issue.fix_strategy,
                            "description": issue.description,
                        }
                        for issue in issues
                    ],
                }

                return json.dumps(summary, indent=2)
            except Exception as e:
                return f"Error parsing QA results: {str(e)}"

        return parse_qa_results_tool

    def _create_plan_tool(self):

        state = self.state

        @tool
        def create_fix_plan_tool() -> str:

            if "issues" not in state:
                return "Error: Must parse QA results first"

            try:
                issues = state["issues"]
                plan = state["issue_parser"].create_fix_plan(issues)
                state["fix_plan"] = plan

                plan_path = state["artifacts_dir"] / "fix_plan.json"
                with open(plan_path, "w") as f:
                    serializable_plan = {
                        **plan,
                        "fix_groups": {
                            strategy: [
                                {
                                    "id": issue.id,
                                    "name": issue.name,
                                    "column": issue.metadata.get("column"),
                                }
                                for issue in issues_list
                            ]
                            for strategy, issues_list in plan["fix_groups"].items()
                        },
                        "unfixable": [
                            {
                                "id": issue.id,
                                "name": issue.name,
                                "description": issue.description,
                            }
                            for issue in plan["unfixable"]
                        ],
                    }
                    json.dump(serializable_plan, f, indent=2)

                return json.dumps(serializable_plan, indent=2)
            except Exception as e:
                return f"Error creating fix plan: {str(e)}"

        return create_fix_plan_tool

    def _create_fix_tool(self):

        state = self.state

        @tool
        def apply_fixes_tool(dataset_path: str) -> str:

            if "fix_plan" not in state:
                return "Error: Must create fix plan first"

            try:
                dataset_p = Path(dataset_path)
                if dataset_p.suffix == ".csv":
                    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
                elif dataset_p.suffix == ".parquet":
                    df = pd.read_parquet(dataset_path)
                else:
                    return f"Error: Unsupported file format: {dataset_p.suffix}"
                state["original_df"] = df.copy()
                state["current_df"] = df.copy()

                plan = state["fix_plan"]
                execution_order = plan["execution_order"]

                results = []
                
                if not execution_order:
                    output_path = state["artifacts_dir"] / "dataset_fixed.parquet"
                    state["current_df"].to_parquet(output_path, index=False)
                    state["fixed_dataset_path"] = str(output_path)
                    
                    summary = {
                        "total_fixes_attempted": 0,
                        "successful_fixes": 0,
                        "failed_fixes": 0,
                        "output_path": str(output_path),
                        "results": [],
                        "note": "No fixes needed - dataset already clean"
                    }
                    return json.dumps(summary, indent=2, default=str)

                for strategy_name in execution_order:
                    issues = plan["fix_groups"][strategy_name]

                    for issue in issues:
                        df_before = state["current_df"].copy()
                        df_after, fix_result = state["strategy_factory"].apply_strategy(
                            state["current_df"], issue
                        )

                        result = {
                            "strategy": strategy_name,
                            "issue_id": issue.id,
                            "issue_name": issue.name,
                            "success": "error" not in fix_result,
                            "result": fix_result,
                        }
                        results.append(result)

                        if result["success"]:
                            state["current_df"] = df_after
                            state["remediation_log"].append(result)

                output_path = state["artifacts_dir"] / "dataset_fixed.parquet"
                state["current_df"].to_parquet(output_path, index=False)
                state["fixed_dataset_path"] = str(output_path)

                summary = {
                    "total_fixes_attempted": len(results),
                    "successful_fixes": sum(1 for r in results if r["success"]),
                    "failed_fixes": sum(1 for r in results if not r["success"]),
                    "output_path": str(output_path),
                    "results": results,
                }

                return json.dumps(summary, indent=2, default=str)
            except Exception as e:
                import traceback

                return f"Error applying fixes: {str(e)}\n{traceback.format_exc()}"

        return apply_fixes_tool

    def _create_validate_tool(self):

        state = self.state

        @tool
        def validate_fixes_tool() -> str:

            if "original_df" not in state or "current_df" not in state:
                return "Error: Must apply fixes first"

            try:
                validation_results = state[
                    "validation_tools"
                ].run_all_validations(
                    state["original_df"], state["current_df"]
                )

                state["validation_results"] = validation_results

                validation_path = (
                    state["artifacts_dir"] / "validation_results.json"
                )
                with open(validation_path, "w") as f:
                    json.dump(validation_results, f, indent=2, default=str)

                return json.dumps(validation_results, indent=2, default=str)
            except Exception as e:
                return f"Error validating fixes: {str(e)}"

        return validate_fixes_tool

    def _create_report_tool(self):

        state = self.state

        @tool
        def generate_report_tool() -> str:

            if "remediation_log" not in state:
                return "Error: No remediation actions to report"

            try:
                report = self._generate_markdown_report(state)

                report_path = state["artifacts_dir"] / "remediation_report.md"
                with open(report_path, "w") as f:
                    f.write(report)

                log_path = state["artifacts_dir"] / "remediation_log.json"
                with open(log_path, "w") as f:
                    json.dump(state["remediation_log"], f, indent=2, default=str)

                return f"Report generated at: {report_path}"
            except Exception as e:
                return f"Error generating report: {str(e)}"

        return generate_report_tool

    def _generate_markdown_report(self, state: dict) -> str:

        report = f"""
        
**Run ID**: {self.run_id}
**Timestamp**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        message = f"""Run the complete remediation pipeline:

1. Parse QA results from: {qa_results_path}
2. Create a prioritized fix plan
3. Apply fixes to dataset: {dataset_path}
4. Validate that fixes didn't introduce new issues
5. Generate a comprehensive remediation report

Provide a final summary of the remediation results."""

        try:
            result = self.pipeline_executor.invoke({"messages": [("user", message)]})

            final_message = ""
            if "messages" in result:
                final_message = (
                    result["messages"][-1].content if result["messages"] else ""
                )

            training_ready = False
            if "validation_results" in self.state:
                validation = self.state["validation_results"]
                training_ready = validation.get("all_validations_passed", False)

            final_result = {
                "success": True,
                "run_id": self.run_id,
                "artifacts_dir": str(self.run_dir),
                "training_ready": training_ready,
                "fixed_dataset_path": self.state.get("fixed_dataset_path"),
                "remediation_report_path": str(
                    self.run_dir / "remediation_report.md"
                ),
                "fixes_applied": len(self.state.get("remediation_log", [])),
                "pipeline_output": final_message,
            }

            return final_result

        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "run_id": self.run_id,
            }
