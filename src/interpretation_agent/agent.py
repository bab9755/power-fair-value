import os
import time
from typing import Any, Generator, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .config import InterpretationConfig

load_dotenv()

SYSTEM_PROMPT = """\
You are an energy analyst interpreting DA-to-curve trading strategy results 
for German electricity markets.


1. Discover artifacts (focus on trading outputs in da_to_curve/)
2. Read key files: trading_guidance.md, backtest_summary.json, 
   daily_signals.csv, invalidation.json
3. Analyze 2-3 key images (backtest P&L, spreads)
4. Do 2-3 web searches for market context (German power prices, gas, renewables)
5. Compute 3-4 key stats (monthly P&L, losing days, signal distribution)
6. Write concise report with sections:
   - Executive Summary (3-4 sentences)
   - Market Context (brief)
   - Strategy Results (P&L, win rate, Sharpe)
   - Key Insights (what worked, what didn't)
   - Risks & Recommendations


- Be concise. Target 800-1200 words total.
- Use only data from tool calls.
- Focus on actionable insights.
- Be honest about limitations (proxy curve vs real forwards).
"""


class InterpretationAgent:
    """Simple agent that reads DA-to-curve outputs and writes an interpretation report."""

    def __init__(self, config: InterpretationConfig):
        self.config = config
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initialise the underlying chat model."""
        if self.config.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                api_key=api_key,
            ).bind(system=SYSTEM_PROMPT)
        elif self.config.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def _build_context(self) -> tuple[str, int]:
        """Collect key artifacts from DA-to-curve outputs to feed the LLM."""
        from pathlib import Path
        import json
        import textwrap

        base = Path(self.config.da_to_curve_dir)
        context_base = Path("context")

        parts: list[str] = []
        artifacts_loaded = 0

        def _read_text(path: Path, label: str, max_chars: int = 4000):
            nonlocal artifacts_loaded
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")
                except Exception:
                    return
                artifacts_loaded += 1
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n... [truncated]"
                parts.append(f"\n\n# {label}\n\n")
                parts.append(content)

        def _read_json(path: Path, label: str, max_chars: int = 4000):
            nonlocal artifacts_loaded
            if path.exists():
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    content = json.dumps(obj, indent=2, default=str)
                except Exception:
                    return
                artifacts_loaded += 1
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n... [truncated]"
                parts.append(f"\n\n# {label}\n\n")
                parts.append(content)

        # Core DA-to-curve artifacts
        _read_text(base / "trading_guidance.md", "Trading Guidance")
        _read_json(base / "backtest_summary.json", "Backtest Summary")
        _read_json(base / "invalidation.json", "Invalidation Status")

        # For daily_signals, include only a small preview to stay within token limits
        signals_path = base / "daily_signals.csv"
        if signals_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(signals_path)
                preview = df.tail(30).to_string(index=False)
                artifacts_loaded += 1
                parts.append("\n\n# Daily Signals (last 30 rows)\n\n")
                parts.append(preview)
            except Exception:
                # If pandas fails for any reason, fall back to raw text head
                try:
                    text_preview = "".join(signals_path.open("r", encoding="utf-8").readlines()[:50])
                    artifacts_loaded += 1
                    parts.append("\n\n# Daily Signals (text preview)\n\n")
                    parts.append(textwrap.shorten(text_preview, width=4000, placeholder="... [truncated]"))
                except Exception:
                    pass

        # Context artifacts from QA / remediation / training
        _read_text(context_base / "qa_remediation_report.md", "QA + Remediation Report")
        _read_json(context_base / "qa_remediation_results.json", "QA + Remediation Results")
        _read_json(context_base / "training_summary.json", "Training Summary")

        # Optional: include a compact view of model metrics by hour if present
        metrics_by_hour_path = context_base / "model_metrics_by_hour.csv"
        if metrics_by_hour_path.exists():
            try:
                import pandas as pd

                df_mh = pd.read_csv(metrics_by_hour_path)
                preview_mh = df_mh.head(48).to_string(index=False)
                artifacts_loaded += 1
                parts.append("\n\n# Model Metrics by Hour (preview)\n\n")
                parts.append(preview_mh)
            except Exception:
                pass

        context = "".join(parts) if parts else "No DA-to-curve artifacts found."
        return context, artifacts_loaded

    def run_streaming(self) -> Generator[dict[str, Any], None, None]:
        """Yield progress events compatible with the CLI's streaming display."""

        start = time.time()
        # Simple progress: one thinking event, then done/error
        yield {"event": "thinking"}

        try:
            result = self._run_once()
            elapsed = round(time.time() - start, 1)
            yield {
                "event": "done",
                "result": {
                    "success": True,
                    "report_path": result["report_path"],
                    "artifacts_loaded": result["artifacts_loaded"],
                    # This simplified implementation does not yet analyse images or hit the web.
                    "images_analyzed": 0,
                    "web_searches": 0,
                    "agent_output": result.get("agent_output", ""),
                },
                "elapsed": elapsed,
            }
        except Exception as e:
            yield {
                "event": "error",
                "error": str(e),
            }

    def _run_once(self) -> dict[str, Any]:
        """Run a single interpretation pass and write the report."""
        from pathlib import Path

        context, artifacts_loaded = self._build_context()

        user_prompt = (
            "Using the DA-to-curve strategy outputs and context below, write the full "
            "interpretation report as described in the system prompt.\n\n"
            "Focus on:\n"
            "- Overall performance of the DA-to-curve strategy\n"
            "- Key drivers of P&L and risk\n"
            "- How to use the signals in practice\n"
            "- Limitations and when to distrust the model\n\n"
            "=== PIPELINE CONTEXT START ===\n"
            f"{context}\n"
            "=== PIPELINE CONTEXT END ===\n"
        )

        response = self.llm.invoke(user_prompt)
        content = response.content if hasattr(response, "content") else str(response)

        output_path = Path(self.config.report_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(content), encoding="utf-8")

        return {
            "report_path": str(output_path),
            "artifacts_loaded": artifacts_loaded,
            "agent_output": str(content),
        }