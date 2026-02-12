
from typing import Optional

import pandas as pd

from ..domain.electricity import ElectricityMarketDomain


class ValidationTools:

    def __init__(self):

        self.domain = ElectricityMarketDomain()

    def validate_fix(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        column: str,
    ) -> dict:

        if column not in df_before.columns or column not in df_after.columns:
            return {"error": f"Column '{column}' not found in both DataFrames"}

        before = df_before[column]
        after = df_after[column]

        before_stats = {
            "count": int(before.count()),
            "missing": int(before.isna().sum()),
            "mean": float(before.mean()) if before.count() > 0 else None,
            "std": float(before.std()) if before.count() > 1 else None,
            "min": float(before.min()) if before.count() > 0 else None,
            "max": float(before.max()) if before.count() > 0 else None,
        }

        after_stats = {
            "count": int(after.count()),
            "missing": int(after.isna().sum()),
            "mean": float(after.mean()) if after.count() > 0 else None,
            "std": float(after.std()) if after.count() > 1 else None,
            "min": float(after.min()) if after.count() > 0 else None,
            "max": float(after.max()) if after.count() > 0 else None,
        }

        issues = []

        new_nans = after_stats["missing"] - before_stats["missing"]
        if new_nans > 0:
            issues.append(f"Introduced {new_nans} new NaN values")

        if before_stats["mean"] is not None and after_stats["mean"] is not None:
            mean_change_pct = (
                abs(after_stats["mean"] - before_stats["mean"])
                / abs(before_stats["mean"])
                * 100
            )
            if mean_change_pct > 20:
                issues.append(
                    f"Mean changed significantly: {mean_change_pct:.1f}%"
                )

        if before_stats["max"] is not None and after_stats["max"] is not None:
            if after_stats["max"] > before_stats["max"] * 1.5:
                issues.append(
                    f"New maximum outlier: {after_stats['max']:.2f} vs {before_stats['max']:.2f}"
                )

        if before_stats["min"] is not None and after_stats["min"] is not None:
            if after_stats["min"] < before_stats["min"] * 1.5:
                issues.append(
                    f"New minimum outlier: {after_stats['min']:.2f} vs {before_stats['min']:.2f}"
                )

        validation_passed = len(issues) == 0

        result = {
            "column": column,
            "validation_passed": validation_passed,
            "before_stats": before_stats,
            "after_stats": after_stats,
            "missing_reduced_by": int(before_stats["missing"] - after_stats["missing"]),
        }

        if issues:
            result["issues"] = issues

        return result

    def cross_validate_generation(self, df: pd.DataFrame) -> dict:

        validation_results = []
        failed_rows = 0

        for idx, row in df.iterrows():
            is_valid, error_msg = self.domain.validate_generation_consistency(row)
            if not is_valid:
                failed_rows += 1
                validation_results.append(
                    {"index": idx, "timestamp": row.get("timestamp"), "error": error_msg}
                )

        consistency_score = 1.0 - (failed_rows / len(df)) if len(df) > 0 else 0.0

        result = {
            "method": "cross_validate_generation",
            "total_rows": len(df),
            "failed_rows": failed_rows,
            "consistency_score": float(consistency_score),
            "validation_passed": consistency_score > 0.95,
        }

        if failed_rows > 0:
            result["sample_failures"] = validation_results[:5]

        return result

    def validate_no_duplicates(self, df: pd.DataFrame, column: str = "timestamp") -> dict:

        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}

        duplicates = df[column].duplicated()
        duplicate_count = duplicates.sum()

        result = {
            "method": "validate_no_duplicates",
            "column": column,
            "total_rows": len(df),
            "duplicate_count": int(duplicate_count),
            "validation_passed": duplicate_count == 0,
        }

        if duplicate_count > 0:
            result["duplicate_values"] = (
                df[duplicates][column].head(10).tolist()
            )

        return result

    def validate_range(
        self, df: pd.DataFrame, column: str, min_value: float, max_value: float
    ) -> dict:

        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}

        series = df[column].dropna()

        below_min = (series < min_value).sum()
        above_max = (series > max_value).sum()
        out_of_range = below_min + above_max

        result = {
            "method": "validate_range",
            "column": column,
            "min_value": min_value,
            "max_value": max_value,
            "total_values": len(series),
            "below_min": int(below_min),
            "above_max": int(above_max),
            "out_of_range": int(out_of_range),
            "validation_passed": out_of_range == 0,
        }

        if out_of_range > 0:
            result["pct_out_of_range"] = float(out_of_range / len(series) * 100)

        return result

    def validate_no_negative_generation(self, df: pd.DataFrame) -> dict:

        gen_columns = [col for col in df.columns if col.startswith("gen_") and col.endswith("_mwh")]

        issues = []
        total_negative = 0

        for col in gen_columns:
            negative_count = (df[col] < -0.1).sum()
            if negative_count > 0:
                issues.append(
                    {
                        "column": col,
                        "negative_count": int(negative_count),
                        "min_value": float(df[col].min()),
                    }
                )
                total_negative += negative_count

        result = {
            "method": "validate_no_negative_generation",
            "columns_checked": gen_columns,
            "total_negative_values": total_negative,
            "validation_passed": total_negative == 0,
        }

        if issues:
            result["issues"] = issues

        return result

    def validate_correlation_preserved(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        col1: str,
        col2: str,
        tolerance: float = 0.1,
    ) -> dict:

        if col1 not in df_before.columns or col2 not in df_before.columns:
            return {"error": f"Columns not found in before DataFrame"}

        if col1 not in df_after.columns or col2 not in df_after.columns:
            return {"error": f"Columns not found in after DataFrame"}

        corr_before = df_before[[col1, col2]].corr().loc[col1, col2]
        corr_after = df_after[[col1, col2]].corr().loc[col1, col2]

        corr_change = abs(corr_after - corr_before)

        validation_passed = corr_change <= tolerance

        result = {
            "method": "validate_correlation_preserved",
            "col1": col1,
            "col2": col2,
            "correlation_before": float(corr_before),
            "correlation_after": float(corr_after),
            "correlation_change": float(corr_change),
            "tolerance": tolerance,
            "validation_passed": validation_passed,
        }

        return result

    def run_all_validations(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        fixed_column: Optional[str] = None,
    ) -> dict:

        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "validations": [],
        }

        if fixed_column:
            fix_validation = self.validate_fix(df_before, df_after, fixed_column)
            results["validations"].append(fix_validation)

        dup_validation = self.validate_no_duplicates(df_after, "timestamp")
        results["validations"].append(dup_validation)

        gen_validation = self.cross_validate_generation(df_after)
        results["validations"].append(gen_validation)

        neg_gen_validation = self.validate_no_negative_generation(df_after)
        results["validations"].append(neg_gen_validation)

        all_passed = all(
            v.get("validation_passed", False) for v in results["validations"]
        )
        results["all_validations_passed"] = all_passed
        results["total_validations"] = len(results["validations"])
        results["passed_validations"] = sum(
            1 for v in results["validations"] if v.get("validation_passed", False)
        )

        return results
