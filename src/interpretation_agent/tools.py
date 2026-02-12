import base64
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import tool


def create_discover_artifacts_tool(state: dict[str, Any], config: Any):
    @tool
    def discover_artifacts() -> str:
        scan_dirs = {
            "outputs": config.outputs_dir,
            "da_to_curve": config.da_to_curve_dir,
            "artifacts": config.artifacts_dir,
            "data": config.data_dir,
        }

        manifest: dict[str, list[dict[str, Any]]] = {
            "trading": [],
            "training": [],
            "qa": [],
            "data": [],
            "other": [],
        }

        qa_keywords = ["qa", "remediation", "cleaning", "validation", "check"]
        training_keywords = ["metric", "model", "prediction", "training", "comparison"]
        trading_keywords = [
            "signal", "backtest", "block", "curve", "spread", "guidance",
            "invalidation", "forward", "monthly", "trading", "pnl",
        ]
        data_keywords = ["smard", "processed", "clean", "unified", "ready"]

        for _dir_label, dir_path in scan_dirs.items():
            p = Path(dir_path)
            if not p.exists():
                continue

            for item in sorted(p.rglob("*")):
                if not item.is_file():
                    continue
                if any(part.startswith(".") for part in item.parts):
                    continue

                name_lower = item.name.lower()
                rel_path = str(item)
                size_kb = round(item.stat().st_size / 1024, 1)
                ext = item.suffix.lower()

                file_info = {
                    "path": rel_path,
                    "name": item.name,
                    "size_kb": size_kb,
                    "extension": ext,
                }

                if any(kw in name_lower for kw in trading_keywords):
                    manifest["trading"].append(file_info)
                elif any(kw in name_lower for kw in qa_keywords):
                    manifest["qa"].append(file_info)
                elif any(kw in name_lower for kw in training_keywords):
                    manifest["training"].append(file_info)
                elif any(kw in name_lower for kw in data_keywords):
                    manifest["data"].append(file_info)
                else:
                    manifest["other"].append(file_info)

        state["manifest"] = manifest

        summary = {cat: len(files) for cat, files in manifest.items()}
        summary["total"] = sum(summary.values())

        return json.dumps({"summary": summary, "artifacts": manifest}, indent=2)

    return discover_artifacts


def create_read_artifact_tool(state: dict[str, Any]):
    @tool
    def read_artifact(path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"

        ext = p.suffix.lower()

        if "loaded_artifacts" not in state:
            state["loaded_artifacts"] = {}

        try:
            if ext in (".csv", ".parquet"):
                df = pd.read_csv(p) if ext == ".csv" else pd.read_parquet(p)
                artifact_key = p.stem
                state["loaded_artifacts"][artifact_key] = df

                head_str = df.head(3).to_string(index=False)

                return (
                    f"**Shape**: {df.shape[0]} rows x {df.shape[1]} columns\n"
                    f"**Columns**: {', '.join(df.columns.tolist())}\n\n"
                    f"**Preview**:\n{head_str}\n\n"
                    f"Use compute_stats for detailed analysis."
                )

            elif ext == ".json":
                with open(p, "r") as f:
                    data = json.load(f)
                state["loaded_artifacts"][p.stem] = data
                content = json.dumps(data, indent=2, default=str)
                if len(content) > 2000:
                    content = content[:2000] + "\n... [truncated]"
                return (
                    "**Content (JSON):**\n```json\n"
                    + content
                    + "\n```"
                )

            elif ext in (".md", ".txt"):
                with open(p, "r") as f:
                    content = f.read()
                state["loaded_artifacts"][p.stem] = content
                if len(content) > 2000:
                    content = content[:2000] + "\n... [truncated]"
                return f"**Content**:\n```\n{content}\n```"

            else:
                return f"Error: Unsupported file type '{ext}' for {path}'"

        except Exception as e:
            return f"Error reading {path}: {str(e)}"

    return read_artifact


def create_analyze_image_tool(
    state: dict[str, Any], llm_provider: str, llm_model: str
):
    @tool
    def analyze_image(path: str, question: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: Image not found: {path}"

        try:
            with open(p, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            media_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            media_type = media_types.get(p.suffix.lower(), "image/png")

            brief_question = f"{question}\n\nProvide a concise answer in 100 words or less."
            
            if llm_provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                vision_llm = ChatAnthropic(
                    model=llm_model, temperature=0.0,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                )
                message = vision_llm.invoke([{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        }},
                        {"type": "text", "text": brief_question},
                    ],
                }])
            elif llm_provider == "openai":
                from langchain_openai import ChatOpenAI
                vision_llm = ChatOpenAI(
                    model=llm_model, temperature=0.0,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
                message = vision_llm.invoke([{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:{media_type};base64,{image_data}",
                        }},
                        {"type": "text", "text": brief_question},
                    ],
                }])
            else:
                return f"Error: Unsupported provider for vision: {llm_provider}"

            desc = message.content if isinstance(message.content, str) else str(message.content)

            if "image_analyses" not in state:
                state["image_analyses"] = {}
            state["image_analyses"][p.name] = desc

            return desc

        except Exception as e:
            return f"Error analyzing image {path}: {str(e)}"

    return analyze_image


def create_compute_stats_tool(state: dict[str, Any]):
    @tool
    def compute_stats(artifact_key: str, query: str) -> str:
        if "loaded_artifacts" not in state:
            return "Error: No artifacts loaded yet. Use read_artifact first."

        if artifact_key not in state["loaded_artifacts"]:
            available = list(state["loaded_artifacts"].keys())
            return f"Error: Artifact '{artifact_key}' not found. Available: {available}"

        artifact = state["loaded_artifacts"][artifact_key]
        if not isinstance(artifact, pd.DataFrame):
            return (
                f"Error: '{artifact_key}' is not a DataFrame "
                f"(type: {type(artifact).__name__}). compute_stats works only "
                f"on CSV/Parquet artifacts."
            )

        df = artifact
        q = query.lower()

        try:
            if "correlation" in q or "corr" in q:
                num = df.select_dtypes(include="number")
                if num.shape[1] < 2:
                    return "Not enough numeric columns for correlation."
                if num.shape[1] > 8:
                    num = num.iloc[:, :8]
                corr = num.corr()
                return f"**Content**:\n```\n{content}\n```"

            elif "monthly" in q or "month" in q:
                date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
                if not date_cols:
                    return "No date/time column found for monthly aggregation."
                tmp = df.copy()
                tmp[date_cols[0]] = pd.to_datetime(tmp[date_cols[0]], errors="coerce")
                tmp["_month"] = tmp[date_cols[0]].dt.to_period("M")
                nums = tmp.select_dtypes(include="number").columns.tolist()[:5]
                monthly = tmp.groupby("_month")[nums].agg(["mean", "sum", "count"])
                output = monthly.to_string()
                if len(output) > 1500:
                    output = output[:1500] + "\n... [truncated]"
                return f"**Content**:\n```\n{content}\n```"

            elif "hour" in q:
                hour_cols = [c for c in df.columns if "hour" in c.lower()]
                if hour_cols:
                    nums = df.select_dtypes(include="number").columns.tolist()
                    hourly = df.groupby(hour_cols[0])[nums].mean()
                    return f"**Content**:\n```\n{content}\n```"
                date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
                if date_cols:
                    tmp = df.copy()
                    tmp[date_cols[0]] = pd.to_datetime(tmp[date_cols[0]], errors="coerce")
                    tmp["_hour"] = tmp[date_cols[0]].dt.hour
                    nums = tmp.select_dtypes(include="number").columns.tolist()
                    hourly = tmp.groupby("_hour")[nums].mean()
                    return f"**Content**:\n```\n{content}\n```"
                return "No hour or datetime column found."

            elif "winning" in q or "losing" in q or "win" in q or "loss" in q:
                pnl_cols = [c for c in df.columns if "pnl" in c.lower() and "cum" not in c.lower()]
                if not pnl_cols:
                    return "No daily pnl column found."
                pnl_col = pnl_cols[0]
                if "losing" in q or "loss" in q:
                    subset = df[df[pnl_col] < 0]
                    label = "Losing Days"
                else:
                    subset = df[df[pnl_col] > 0]
                    label = "Winning Days"
                if subset.empty:
                    return f"No {label.lower()} found."
                if len(subset) > 20:
                    subset = subset.head(20)
                    note = f" (showing first 20 of {len(df[df[pnl_col] < 0] if 'losing' in q else df[df[pnl_col] > 0])})"
                else:
                    note = ""
                return subset.to_string(index=False) + note

            elif "value_counts" in q or "distribution" in q:
                for col in df.columns:
                    if col.lower() in q:
                        vc = df[col].value_counts().head(20)
                        return vc.to_string()
                results = []
                for col in df.select_dtypes(exclude="number").columns:
                    vc = df[col].value_counts().head(10)
                    results.append(f"{col}:\n{vc.to_string()}")
                return "\n\n".join(results) if results else "No categorical columns."

            elif "comparison" in q or "compare" in q:
                model_cols = [c for c in df.columns if "model" in c.lower()]
                if model_cols:
                    nums = df.select_dtypes(include="number").columns.tolist()
                    comp = df.groupby(model_cols[0])[nums].mean()
                    return comp.to_string()
                return df.to_string()

            elif "describe" in q:
                for col in df.columns:
                    if col.lower() in q:
                        return df[col].describe().to_string()
                return df.describe().to_string()

            else:
                parts = [
                    f"Shape: {df.shape}",
                    f"Columns: {', '.join(df.columns.tolist())}",
                ]
                nulls = df.isnull().sum()
                if nulls.any():
                    parts.append("Missing values by column:")
                    parts.append(nulls[nulls > 0].to_string())
                return "\n".join(parts)

        except Exception as e:
            return f"Error computing stats for '{artifact_key}': {str(e)}"

    return compute_stats


def create_search_market_context_tool(state: dict[str, Any]):
    @tool
    def search_market_context(query: str) -> str:
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    results.append({
                        "title": r.get("title", ""),
                        "body": r.get("body", "")[:200],
                        "href": r.get("href", ""),
                    })

            if not results:
                return "No search results found."

            if "market_context" not in state:
                state["market_context"] = []
            state["market_context"].extend(results)

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"{i}. {r['title']}\n{r['body']}\nSource: {r['href']}"
                )

            return "\n\n".join(formatted)

        except Exception as e:
            return f"Error searching the web: {str(e)}"

    return search_market_context


def create_write_report_tool(state: dict[str, Any], config: Any):
    @tool
    def write_report(sections: str) -> str:
        try:
            sections_data = json.loads(sections)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON for sections: {e}"

        if not isinstance(sections_data, list):
            return "Error: sections must be a JSON array of {title, content} objects."

        report_lines = [
            "# Interpretation Report",
            "",
            "*Generated by the Interpretation Agent*",
            "",
            "---",
            "",
        ]

        for section in sections_data:
            title = section.get("title", "Untitled Section")
            content = section.get("content", "")
            report_lines.append(f"## {title}")
            report_lines.append("")
            report_lines.append(content)
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        report_content = "\n".join(report_lines)

        output_path = Path(config.report_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report_content)

        state["report_content"] = report_content
        state["report_path"] = str(output_path)

        return (
            f"Report written to: {output_path} "
            f"({len(report_content)} chars, {len(sections_data)} sections)"
        )

    return write_report
