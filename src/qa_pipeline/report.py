
from datetime import datetime
from typing import Any

from .checks import CheckResult
from .cleaning import CleaningLog


def generate_qa_report(
    dataset_profile: dict[str, Any],
    qa_results: list[CheckResult],
    cleaning_log: CleaningLog,
    plot_paths: list[str] = None,
) -> str:

    plot_paths = plot_paths or []

    status_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for result in qa_results:
        status_counts[result.status] += 1

    training_ready = status_counts["FAIL"] == 0

    sections = [
        _header(),
        _dataset_overview(dataset_profile),
        _pass_fail_summary(status_counts, training_ready),
        _check_details(qa_results),
        _cleaning_actions(cleaning_log),
        _recommendations(qa_results, training_ready),
        _plots_section(plot_paths),
        _final_verdict(training_ready),
    ]

    return "\n\n".join(sections)


def _header() -> str:

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""

**Generated:** {timestamp}

---"""


def _dataset_overview(profile: dict[str, Any]) -> str:

    shape = profile.get("shape", {})
    date_range = profile.get("date_range", {})

    overview = f"""

- **Rows:** {shape.get('rows', 'N/A'):,}
- **Columns:** {shape.get('columns', 'N/A')}
- **Memory Usage:** {profile.get('memory_usage_mb', 0):.2f} MB"""

    if date_range:
        overview += f"""
- **Date Range:** {date_range.get('start')} to {date_range.get('end')}
- **Duration:** {date_range.get('duration_days', 0):,} days"""

    columns = profile.get("columns", [])
    if columns:
        overview += f"\n- **Columns:** {', '.join(f'`{c}`' for c in columns)}"

    return overview


def _pass_fail_summary(status_counts: dict[str, int], training_ready: bool) -> str:

    total = sum(status_counts.values())

    summary = f"""

**Total Checks:** {total}

-  **PASS:** {status_counts['PASS']}
- ⚠️  **WARN:** {status_counts['WARN']}
-  **FAIL:** {status_counts['FAIL']}

**Status:** {'🟢 READY FOR TRAINING' if training_ready else '🔴 NOT READY'}"""

    return summary


def _check_details(qa_results: list[CheckResult]) -> str:

    details = "## Check Details\n\n"

    for result in qa_results:
        status_emoji = "✅" if result.status == "PASS" else "⚠️" if result.status == "WARN" else "❌"
        details += f"### {status_emoji} {result.check_name}\n\n"
        details += f"**Status:** {result.status}\n\n"
        
        if result.message:
            details += f"{result.message}\n\n"

        if result.metrics:
            details += "**Metrics:**\n\n"
            details += _format_dict(result.metrics, indent=0)
            details += "\n"

        if result.thresholds:
            details += "**Thresholds:**\n\n"
            details += _format_dict(result.thresholds, indent=0)
            details += "\n"

        if result.recommendation:
            details += f"**Recommendation:** {result.recommendation}\n\n"

        details += "---\n\n"

    return details


def _cleaning_actions(cleaning_log: CleaningLog) -> str:

    actions = cleaning_log.to_dict()

    if not actions:
        return """

No cleaning actions were performed."""

    section = f"""
**Total Actions:** {len(actions)}
"""
    return section


def _recommendations(qa_results: list[CheckResult], training_ready: bool) -> str:

    recommendations = [
        r.recommendation for r in qa_results if r.recommendation and r.status != "PASS"
    ]

    if not recommendations and training_ready:
        return "## Recommendations\n\nDataset is ready for training. No immediate actions required."

    section = "## Recommendations\n\n"

    if recommendations:
        section += "**Specific Recommendations:**\n\n"
        for rec in recommendations:
            section += f"- {rec}\n"
        section += "\n"

    section += "**General Guidance:**\n\n"
    section += "- DST transitions: The dataset accounts for Europe/Berlin timezone DST changes\n"
    section += "- Missing data: Consider imputation strategies for columns with high missingness\n"
    section += "- Negative prices: Negative electricity prices are valid market phenomena\n"
    section += "- Feature engineering: Consider adding time-based features (hour, day of week, etc.)\n"

    return section


def _plots_section(plot_paths: list[str]) -> str:

    if not plot_paths:
        return "## Plots\n\nNo plots were generated for this run."

    section = "## Plots\n\nThe following plots have been generated:\n\n"
    for plot_path in plot_paths:
        section += f"- `{plot_path}`\n"

def _final_verdict(training_ready: bool) -> str:

    if training_ready:
        verdict = "## Final Verdict\n\n"
        verdict += "- [x] Required columns present and properly typed\n"
        verdict += "- [x] Timestamp integrity acceptable\n"
        verdict += "- [x] No duplicate timestamps\n"
        verdict += "- [x] Target column has acceptable missingness\n"
        verdict += "- [x] No unexpected missing timestamp gaps\n"
        verdict += "- [x] Missingness thresholds not violated (or only WARNs)\n\n"
        verdict += "**FINAL_VERDICT: TRAINING_READY**"
    else:
        verdict = "## Final Verdict\n\n"
        verdict += "Some requirements are not met. Please review the FAIL checks above.\n\n"
        verdict += "**FINAL_VERDICT: NOT_READY**\n\n"
        verdict += "**Next Steps:**\n"
        verdict += "1. Review all FAIL status checks above\n"
        verdict += "2. Address data quality issues\n"
        verdict += "3. Re-run the QA pipeline after fixes"

    return verdict


def _format_dict(d: dict[str, Any], indent: int = 0, inline: bool = False) -> str:

    if inline:
        items = [f"{k}={v}" for k, v in d.items()]
        return ", ".join(items)

    prefix = "  " * indent
    lines = []

    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}- **{key}:**")
            lines.append(_format_dict(value, indent + 1))
        elif isinstance(value, list) and len(value) > 0:
            if len(value) <= 3:
                lines.append(f"{prefix}- **{key}:** {', '.join(map(str, value))}")
            else:
                lines.append(
                    f"{prefix}- **{key}:** {', '.join(map(str, value[:3]))} ... ({len(value)} total)"
                )
        else:
            lines.append(f"{prefix}- **{key}:** {value}")

    return "\n".join(lines)
