
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class IssueSeverity(Enum):

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(Enum):

    MISSING_DATA = "missing_data"
    TIME_SERIES_GAP = "time_series_gap"
    OUTLIER = "outlier"
    CONSISTENCY = "consistency"
    DUPLICATE = "duplicate"
    SCHEMA = "schema"
    DST_TRANSITION = "dst_transition"


@dataclass
class Issue:

    id: str
    name: str
    category: IssueCategory
    severity: IssueSeverity
    status: str
    description: str
    metadata: dict[str, Any]
    fixable: bool
    fix_strategy: Optional[str] = None


class IssueParser:

    def __init__(self):

        self.issue_mapping = {
            "missingness": IssueCategory.MISSING_DATA,
            "target_readiness": IssueCategory.MISSING_DATA,
            "time_series_integrity": IssueCategory.TIME_SERIES_GAP,
            "value_sanity": IssueCategory.OUTLIER,
            "cross_field_consistency": IssueCategory.CONSISTENCY,
            "schema_types": IssueCategory.SCHEMA,
        }

    def parse_qa_results(self, qa_results_path: Path | str) -> list[Issue]:

        qa_results_path = Path(qa_results_path)
        if not qa_results_path.exists():
            raise FileNotFoundError(f"QA results not found: {qa_results_path}")

        with open(qa_results_path, "r") as f:
            data = json.load(f)

        issues = []
        checks = data.get("checks", [])

        for check in checks:
            if check.get("id") == "missingness" and check.get("status") in ["WARN", "FAIL"]:
                column_issues = self._parse_missingness_check(check)
                issues.extend(column_issues)
            else:
                issue = self._parse_check(check)
                if issue and issue.status in ["WARN", "FAIL"]:
                    issues.append(issue)

        return issues

    def _parse_missingness_check(self, check: dict) -> list[Issue]:

        issues = []
        metrics = check.get("metrics", {})
        status = check.get("status", "")
        
        warn_columns = metrics.get("warn_threshold_exceeded", [])
        fail_columns = metrics.get("fail_threshold_exceeded", [])
        per_column = metrics.get("per_column", {})
        
        for column in warn_columns + fail_columns:
            if column in per_column:
                missing_pct = per_column[column].get("percent", 0)
                
                issue = Issue(
                    id=f"missingness_{column}",
                    name=f"Missing Data: {column}",
                    category=IssueCategory.MISSING_DATA,
                    severity=IssueSeverity.HIGH if column in fail_columns else IssueSeverity.MEDIUM,
                    status="FAIL" if column in fail_columns else "WARN",
                    description=f"{column} has {missing_pct:.2f}% missing data",
                    metadata={"column": column, "missing_pct": missing_pct},
                    fixable=missing_pct < 25.0,
                    fix_strategy=self._suggest_fix_strategy_for_column(column, missing_pct),
                )
                issues.append(issue)
        
        return issues

    def _parse_check(self, check: dict) -> Optional[Issue]:

        check_id = check.get("id", "")
        name = check.get("name", "")
        status = check.get("status", "")
        message = check.get("message", "")
        metadata = check.get("metrics", {})

        category = self._categorize_check(check_id)

        severity = self._determine_severity(check_id, status, metadata)

        fixable = self._is_fixable(category, metadata)

        fix_strategy = self._suggest_fix_strategy(category, check_id, metadata)

        return Issue(
            id=check_id,
            name=name,
            category=category,
            severity=severity,
            status=status,
            description=message,
            metadata=metadata,
            fixable=fixable,
            fix_strategy=fix_strategy,
        )

    def _categorize_check(self, check_id: str) -> IssueCategory:

        if check_id in self.issue_mapping:
            return self.issue_mapping[check_id]

        for key, category in self.issue_mapping.items():
            if key in check_id or check_id in key:
                return category

        return IssueCategory.MISSING_DATA

    def _determine_severity(
        self, check_id: str, status: str, metadata: dict
    ) -> IssueSeverity:

        if status == "FAIL":
            if check_id == "target_readiness":
                return IssueSeverity.CRITICAL
            if check_id == "schema_types":
                return IssueSeverity.CRITICAL
            if metadata.get("duplicate_count", 0) > 0:
                return IssueSeverity.CRITICAL

        if status == "FAIL":
            if check_id == "time_series_integrity":
                return IssueSeverity.HIGH
            if check_id == "missingness":
                return IssueSeverity.HIGH
            if check_id == "cross_field_consistency":
                return IssueSeverity.HIGH

        if status == "WARN":
            return IssueSeverity.MEDIUM

        return IssueSeverity.LOW

    def _is_fixable(self, category: IssueCategory, metadata: dict) -> bool:

        fixable_categories = {
            IssueCategory.MISSING_DATA,
            IssueCategory.TIME_SERIES_GAP,
            IssueCategory.DUPLICATE,
            IssueCategory.DST_TRANSITION,
        }

        if category in fixable_categories:
            if category == IssueCategory.MISSING_DATA:
                missing_pct = metadata.get("missing_pct", 0)
                if missing_pct > 25:
                    return False

            return True

        return False

    def _suggest_fix_strategy(
        self, category: IssueCategory, check_id: str, metadata: dict
    ) -> Optional[str]:

        if category == IssueCategory.TIME_SERIES_GAP:
            if metadata.get("duplicate_count", 0) > 0:
                return "remove_duplicates"
            if metadata.get("missing_timestamps_total", 0) > 0:
                if metadata.get("dst_spring_gaps", 0) > 0 and metadata.get("unexpected_gaps", 0) == 0:
                    return "handle_dst_transitions"
                return "fill_timestamp_gaps"
            return "fill_timestamp_gaps"

        elif category == IssueCategory.MISSING_DATA:
            column = metadata.get("column", "")
            missing_pct = metadata.get("missing_pct", 0)

            if "price_neighbors" in column:
                return "impute_by_correlation"
            elif "gen_total" in column:
                return "impute_by_calculation"
            elif "load" in column:
                return "impute_by_seasonal_pattern"
            elif missing_pct < 5:
                return "impute_by_interpolation"
            elif missing_pct < 15:
                return "impute_by_seasonal_naive"
            else:
                return "flag_for_review"

        elif category == IssueCategory.DUPLICATE:
            return "remove_duplicates"

        elif category == IssueCategory.DST_TRANSITION:
            return "handle_dst_transitions"

        return None

    def _suggest_fix_strategy_for_column(
        self, column: str, missing_pct: float
    ) -> Optional[str]:

        if "price_neighbors" in column:
            return "impute_by_correlation"
        elif "gen_total" in column:
            return "impute_by_calculation"
        elif "load" in column:
            return "impute_by_seasonal_pattern"
        elif "gen_other" in column:
            return "impute_by_seasonal_naive"
        elif missing_pct < 5:
            return "impute_by_interpolation"
        elif missing_pct < 15:
            return "impute_by_seasonal_naive"
        else:
            return "flag_for_review"

    def prioritize_issues(self, issues: list[Issue]) -> list[Issue]:

        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
        }

        return sorted(
            issues,
            key=lambda issue: (
                severity_order[issue.severity],
                not issue.fixable,
                issue.category.value,
            ),
        )

    def create_fix_plan(self, issues: list[Issue]) -> dict:

        prioritized = self.prioritize_issues(issues)

        fix_groups = {}
        for issue in prioritized:
            if issue.fixable and issue.fix_strategy:
                strategy = issue.fix_strategy
                if strategy not in fix_groups:
                    fix_groups[strategy] = []
                fix_groups[strategy].append(issue)

        unfixable = [issue for issue in prioritized if not issue.fixable]

        plan = {
            "total_issues": len(issues),
            "fixable_issues": len(issues) - len(unfixable),
            "unfixable_issues": len(unfixable),
            "fix_groups": fix_groups,
            "unfixable": unfixable,
            "execution_order": self._determine_execution_order(fix_groups),
        }

        return plan

    def _determine_execution_order(self, fix_groups: dict) -> list[str]:

        priority_order = [
            "handle_dst_transitions",
            "remove_duplicates",
            "fill_timestamp_gaps",
            "impute_by_calculation",
            "impute_by_correlation",
            "impute_by_interpolation",
            "impute_by_seasonal_pattern",
            "impute_by_seasonal_naive",
            "flag_for_review",
        ]

        execution_order = [
            strategy for strategy in priority_order if strategy in fix_groups
        ]

        return execution_order
