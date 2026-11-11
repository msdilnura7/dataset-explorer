from io import BytesIO
import json
import math
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


APP_NAME = "Dataset Explorer"
MAX_FILE_SIZE = 5 * 1024 * 1024
RUN_REQUIRED_FIELDS = (
    "format_version",
    "run_id",
    "dataset_id",
    "split_id",
    "features",
    "target",
    "positive_class",
    "model",
    "penalty",
    "C",
    "seed",
    "threshold",
    "evaluation_split",
    "metrics",
    "note",
)
RUN_METRIC_FIELDS = ("train_f1", "f1", "precision", "recall")
RUN_CONDITION_FIELDS = (
    "evaluation_split",
    "dataset_id",
    "split_id",
    "target",
    "positive_class",
    "features",
    "threshold",
    "seed",
)
RUN_SETTING_FIELDS = (
    "dataset_id",
    "split_id",
    "features",
    "target",
    "positive_class",
    "model",
    "penalty",
    "C",
    "seed",
    "threshold",
    "evaluation_split",
    "note",
)


def read_evaluation_json(uploaded_file, label: str):
    if uploaded_file is None:
        return None, None
    if not uploaded_file.name.lower().endswith(".json"):
        return None, f"{label} must be a .json file."

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        return None, f"{label} JSON file is empty."
    try:
        record = json.loads(file_bytes.decode("utf-8-sig"))
    except UnicodeDecodeError:
        return None, f"{label} JSON file must use UTF-8 encoding."
    except json.JSONDecodeError as error:
        return None, f"{label} JSON is malformed (line {error.lineno}, column {error.colno})."

    if not isinstance(record, dict):
        return None, f"{label} JSON must contain one record as an object."
    if type(record.get("format_version")) is not int or record["format_version"] != 1:
        return None, f"{label} record must include format_version: 1."
    return record, None


def json_number_is_finite(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_run_metric(value) -> bool:
    return json_number_is_finite(value) and 0 <= value <= 1


def run_metric_value(record: dict, metric: str):
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None, "metrics must be an object"
    if metric not in metrics:
        return None, f"metrics.{metric} is missing"
    value = metrics[metric]
    if not valid_run_metric(value):
        return None, f"metrics.{metric} must be a finite number from 0 to 1"
    return value, None


def run_field_is_known(record: dict, field: str) -> bool:
    if field not in record or record[field] is None:
        return False
    value = record[field]
    if field in {"dataset_id", "split_id", "target", "positive_class", "evaluation_split", "model", "penalty"}:
        return isinstance(value, str) and bool(value.strip())
    if field == "features":
        return isinstance(value, list) and all(
            isinstance(feature, str) and bool(feature.strip()) for feature in value
        )
    if field in {"threshold", "C"}:
        return json_number_is_finite(value)
    if field == "seed":
        return (isinstance(value, str) and bool(value.strip())) or json_number_is_finite(value)
    if field == "note":
        return isinstance(value, str)
    if field == "format_version":
        return type(value) is int and value == 1
    return True


def run_values_equal(field: str, value_a, value_b) -> bool:
    if field == "features":
        return set(value_a) == set(value_b)
    return value_a == value_b


def display_run_value(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def display_run_setting(record: dict, field: str) -> str:
    if field not in record or record[field] is None:
        return "Not recorded"
    if isinstance(record[field], str) and not record[field].strip() and field != "note":
        return "(blank)"
    if field == "features" and not run_field_is_known(record, field):
        return "Invalid feature list"
    if field == "note" and record[field] == "":
        return "(empty)"
    return display_run_value(record[field])


def assess_run_conditions(record_a: dict, record_b: dict):
    values = {"A": {}, "B": {}}
    unavailable = []
    for label, record in (("A", record_a), ("B", record_b)):
        for field in RUN_CONDITION_FIELDS:
            if not run_field_is_known(record, field):
                unavailable.append(f"{label} {field} is missing or not usable")
            else:
                values[label][field] = record[field]

    differences = []
    for field in RUN_CONDITION_FIELDS:
        if field not in values["A"] or field not in values["B"]:
            continue
        if not run_values_equal(field, values["A"][field], values["B"][field]):
            differences.append(
                f"{field} differs (A: {display_run_value(values['A'][field])}; "
                f"B: {display_run_value(values['B'][field])})"
            )

    for label in ("A", "B"):
        evaluation_split = values[label].get("evaluation_split")
        if evaluation_split is not None and evaluation_split != "validation":
            differences.append(
                f"{label} evaluation_split is {display_run_value(evaluation_split)}; both must be 'validation'"
            )

    if differences:
        details = differences[:]
        if unavailable:
            details.append("Other context is unavailable: " + "; ".join(unavailable))
        return "Conditions differ", details
    if unavailable:
        return "Insufficient context", unavailable
    return "Same recorded conditions", []


def show_run_comparison() -> None:
    st.subheader("Compare evaluation runs (JSON)")
    st.caption(
        "Upload two format_version 1 evaluation records. Comparisons use recorded values only; "
        "no settings are filled in or inferred."
    )
    upload_columns = st.columns(2)
    with upload_columns[0]:
        file_a = st.file_uploader("A", type=["json"], key="run_comparison_a")
    with upload_columns[1]:
        file_b = st.file_uploader("B", type=["json"], key="run_comparison_b")

    if file_a is not None:
        st.caption(f"A filename: {file_a.name}")
    if file_b is not None:
        st.caption(f"B filename: {file_b.name}")

    record_a, error_a = read_evaluation_json(file_a, "A")
    record_b, error_b = read_evaluation_json(file_b, "B")
    if error_a:
        st.error(error_a)
    if error_b:
        st.error(error_b)
    for label, record in (("A", record_a), ("B", record_b)):
        if record is not None:
            run_id = record.get("run_id")
            run_id_text = run_id if isinstance(run_id, str) and run_id.strip() else "Not recorded"
            st.caption(f"{label} run ID: {run_id_text}")
            missing_fields = [field for field in RUN_REQUIRED_FIELDS if field not in record]
            if missing_fields:
                st.warning(
                    f"{label} record is missing field(s): {', '.join(missing_fields)}. "
                    "Missing values are marked unavailable; no defaults are used."
                )

    if file_a is None or file_b is None:
        st.info("Upload both JSON files to compare the evaluation runs.")
        return
    if error_a or error_b:
        st.info("The run comparison is unavailable until both files contain valid format_version 1 records.")
        return

    condition_status, condition_details = assess_run_conditions(record_a, record_b)
    st.markdown(f"**Condition comparison: {condition_status}.**")
    if condition_details:
        for detail in condition_details:
            st.write(f"- {detail}")
    st.caption(
        "Model, penalty, and C are recorded settings; their differences alone do not change the condition status. "
        "IDs are labels, not audited data hashes. Metric direction is not a quality verdict."
    )

    metric_values = {"A": {}, "B": {}}
    metric_issues = []
    for label, record in (("A", record_a), ("B", record_b)):
        for metric in RUN_METRIC_FIELDS:
            value, issue = run_metric_value(record, metric)
            metric_values[label][metric] = value
            if issue:
                metric_issues.append(f"{label} {issue}")

    card_columns = st.columns(3)
    metric_labels = (("F1", "f1"), ("Precision", "precision"), ("Recall", "recall"))
    for card, (label, metric) in zip(card_columns, metric_labels):
        value_a = metric_values["A"][metric]
        value_b = metric_values["B"][metric]
        shown_a = "Unavailable" if value_a is None else str(value_a)
        shown_b = "Unavailable" if value_b is None else str(value_b)
        if value_a is not None and value_b is not None and condition_status == "Same recorded conditions":
            delta = Decimal(str(value_b)) - Decimal(str(value_a))
            card.metric(
                label,
                f"{shown_a} → {shown_b}",
                delta=f"{delta:+.6g}",
                delta_color="off" if delta == 0 else "normal",
            )
        elif value_a is not None and value_b is not None:
            card.metric(
                label,
                f"{shown_a} → {shown_b}",
                delta="Not comparable",
                delta_color="off",
            )
        else:
            card.metric(label, f"{shown_a} → {shown_b}", delta="Delta unavailable", delta_color="off")

    st.caption(
        "Metric deltas are shown only when both records have the same known validation conditions. "
        "Up (+) is green, down (−) is red, and zero is neutral; direction is not a quality verdict."
    )
    if metric_issues:
        st.warning(
            "Invalid or missing metrics: " + "; ".join(metric_issues) +
            ". Affected values are unavailable and are not used in calculations."
        )

    with st.expander("Recorded settings and differences", expanded=False):
        setting_labels = {
            "dataset_id": "Dataset ID",
            "split_id": "Split ID",
            "features": "Features (order ignored for comparison)",
            "target": "Target",
            "positive_class": "Positive class",
            "model": "Model",
            "penalty": "Penalty",
            "C": "C",
            "seed": "Seed",
            "threshold": "Threshold",
            "evaluation_split": "Evaluation split",
            "note": "Note",
        }
        rows = []
        for field in RUN_SETTING_FIELDS:
            known_a = run_field_is_known(record_a, field)
            known_b = run_field_is_known(record_b, field)
            if not known_a or not known_b:
                difference = "Unavailable"
            elif run_values_equal(field, record_a[field], record_b[field]):
                difference = "Same"
            else:
                difference = "Changed"
            rows.append(
                {
                    "Setting": setting_labels[field],
                    "A": display_run_setting(record_a, field),
                    "B": display_run_setting(record_b, field),
                    "Difference": difference,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("Training F1 minus evaluation F1", expanded=False):
        gap_rows = []
        for label, record in (("A", record_a), ("B", record_b)):
            evaluation_split = record.get("evaluation_split")
            train_f1 = metric_values[label]["train_f1"]
            evaluation_f1 = metric_values[label]["f1"]
            if evaluation_split != "validation":
                gap = f"Unavailable (evaluation_split: {display_run_value(evaluation_split)})"
            elif train_f1 is None or evaluation_f1 is None:
                gap = "Unavailable (train_f1 or f1 is invalid or missing)"
            else:
                gap = str(Decimal(str(train_f1)) - Decimal(str(evaluation_f1)))
            gap_rows.append(
                {
                    "Run": label,
                    "Evaluation split": display_run_setting(record, "evaluation_split"),
                    "train_f1": "Unavailable" if train_f1 is None else str(train_f1),
                    "f1": "Unavailable" if evaluation_f1 is None else str(evaluation_f1),
                    "train_f1 − f1": gap,
                }
            )
        st.dataframe(pd.DataFrame(gap_rows), width="stretch", hide_index=True)


def show_numeric_inspection(data: pd.DataFrame) -> None:
    st.subheader("Inspect numeric values")
    numeric_columns = [
        column
        for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column].dtype)
        and not pd.api.types.is_bool_dtype(data[column].dtype)
    ]
    if not numeric_columns:
        st.info("No numeric columns are available to inspect in this CSV.")
        return

    selected_column = st.selectbox(
        "Numeric column to inspect",
        numeric_columns,
        key="inspection_numeric_column",
    )
    non_missing_values = data[selected_column].dropna()
    finite_mask = np.isfinite(non_missing_values).astype(bool)
    finite_values = non_missing_values.loc[finite_mask].copy()
    usable_count = len(finite_values)
    excluded_count = len(data) - usable_count

    count_columns = st.columns(2)
    count_columns[0].metric("Usable finite values", usable_count)
    count_columns[1].metric("Missing or excluded values", excluded_count)
    st.caption("Missing or excluded counts include empty/NaN values and positive or negative infinity.")

    if usable_count == 0:
        st.info("This column has no usable finite values to plot or inspect.")
        return

    if usable_count < 4:
        st.info(
            "Fewer than four usable finite values are available. The boxplot is shown, but an IQR verdict "
            "and inspection flags are not produced."
        )
        q1 = finite_values.quantile(0.25, interpolation="linear")
        q3 = finite_values.quantile(0.75, interpolation="linear")
        median = finite_values.median()
        whisker_low = finite_values.min()
        whisker_high = finite_values.max()
        values_to_mark = finite_values
        marker_label = "Usable finite values"
        needs_inspection = None
    else:
        q1 = finite_values.quantile(0.25, interpolation="linear")
        q3 = finite_values.quantile(0.75, interpolation="linear")
        iqr = q3 - q1
        lower_boundary = q1 - 1.5 * iqr
        upper_boundary = q3 + 1.5 * iqr
        needs_inspection = (finite_values < lower_boundary) | (finite_values > upper_boundary)
        values_to_mark = finite_values.loc[needs_inspection]
        in_bounds_values = finite_values.loc[~needs_inspection]
        median = finite_values.median()
        whisker_low = in_bounds_values.min()
        whisker_high = in_bounds_values.max()
        marker_label = "Needs inspection"

        st.caption(
            "IQR rule: pandas linear Q1/Q3; IQR = Q3 − Q1; boundaries = Q1 − 1.5 × IQR and "
            "Q3 + 1.5 × IQR. Only values strictly outside the boundaries are flagged."
        )
        first_boundary_row = st.columns(3)
        first_boundary_row[0].metric("Q1", str(q1))
        first_boundary_row[1].metric("Q3", str(q3))
        first_boundary_row[2].metric("IQR", str(iqr))
        second_boundary_row = st.columns(3)
        second_boundary_row[0].metric("Lower boundary", str(lower_boundary))
        second_boundary_row[1].metric("Upper boundary", str(upper_boundary))
        second_boundary_row[2].metric("Needs inspection", int(needs_inspection.sum()))

    box_stats = [{
        "label": str(selected_column),
        "q1": q1,
        "med": median,
        "q3": q3,
        "whislo": whisker_low,
        "whishi": whisker_high,
        "fliers": [],
    }]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bxp(
        box_stats,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#22ff05", "edgecolor": "#103c37"},
        medianprops={"color": "#103c37", "linewidth": 2},
        whiskerprops={"color": "#103c37"},
        capprops={"color": "#103c37"},
    )
    if len(values_to_mark):
        horizontal_jitter = np.linspace(-0.07, 0.07, len(values_to_mark))
        axis.scatter(
            1 + horizontal_jitter,
            values_to_mark.to_numpy(),
            color="#d94b32" if needs_inspection is not None else "#2563a6",
            edgecolors="white",
            linewidths=0.6,
            s=55,
            label=marker_label,
            zorder=3,
        )
    if needs_inspection is not None:
        axis.axhline(lower_boundary, color="#9b5a25", linestyle="--", linewidth=1, label="IQR boundaries")
        if upper_boundary != lower_boundary:
            axis.axhline(upper_boundary, color="#9b5a25", linestyle="--", linewidth=1, label="_nolegend_")
    axis.set_title(f"Boxplot of {selected_column}")
    axis.set_ylabel("Value")
    axis.set_xticks([1], [str(selected_column)])
    axis.set_facecolor("#f4fff8")
    axis.tick_params(colors="#103c37")
    axis.yaxis.label.set_color("#103c37")
    axis.title.set_color("#103c37")
    axis.grid(axis="y", color="#103c37", alpha=0.18)
    if len(values_to_mark) or needs_inspection is not None:
        axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    if needs_inspection is not None:
        if needs_inspection.any():
            st.markdown("#### Rows marked Needs inspection")
            flagged_positions = data.index.get_indexer(values_to_mark.index)
            matching_rows = data.loc[values_to_mark.index].copy()
            row_id_column = "CSV data row (1-based)"
            while row_id_column in matching_rows.columns:
                row_id_column = f"_{row_id_column}"
            matching_rows.insert(0, row_id_column, flagged_positions + 1)
            matching_rows.insert(1, "Inspection status", "Needs inspection")
            st.caption("CSV data row numbers exclude the header and skipped blank lines.")
            st.dataframe(matching_rows, width="stretch", hide_index=True)
        else:
            st.info("No values need inspection under this IQR rule.")


def comparison_cell_is_missing(value: str) -> bool:
    return value.strip() == ""


def read_comparison_csv(uploaded_file, label: str, separator: str):
    file_bytes = uploaded_file.getvalue()
    if not uploaded_file.name.lower().endswith(".csv"):
        return None, f"{label} must be a .csv file."
    if not file_bytes:
        return None, f"{label} file is empty."
    if len(file_bytes) > MAX_FILE_SIZE:
        return None, f"{label} file is larger than 5 MB."

    try:
        frame = pd.read_csv(
            BytesIO(file_bytes),
            sep=separator,
            header=0,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            skip_blank_lines=False,
        )
    except pd.errors.EmptyDataError:
        return None, f"{label} file has no readable header."
    except Exception as error:
        return None, f"{label} file could not be read as UTF-8 CSV: {error}"

    if len(frame.columns) == 0:
        return None, f"{label} file has no column names."
    return frame, None


def show_version_comparison(separator: str) -> None:
    st.subheader("Compare CSV versions")
    st.caption("Comparison uploads use the selected separator above and preserve every field as text.")
    upload_columns = st.columns(2)
    with upload_columns[0]:
        before_file = st.file_uploader(
            "BEFORE (old)",
            type=["csv"],
            help="UTF-8 CSV, up to 5 MB.",
            key="comparison_before_upload",
        )
    with upload_columns[1]:
        after_file = st.file_uploader(
            "AFTER (new)",
            type=["csv"],
            help="UTF-8 CSV, up to 5 MB.",
            key="comparison_after_upload",
        )

    if before_file is not None:
        st.caption(f"BEFORE filename: {before_file.name}")
    if after_file is not None:
        st.caption(f"AFTER filename: {after_file.name}")
    if before_file is None or after_file is None:
        st.info("Upload both BEFORE and AFTER CSV files to see the comparison summary.")
        return

    before_data, before_error = read_comparison_csv(before_file, "BEFORE", separator)
    after_data, after_error = read_comparison_csv(after_file, "AFTER", separator)
    if before_error:
        st.error(before_error)
    if after_error:
        st.error(after_error)
    if before_error or after_error:
        st.info("The comparison summary is unavailable until both files are valid.")
        return

    before_missing = {
        column: int(before_data[column].map(comparison_cell_is_missing).sum())
        for column in before_data.columns
    }
    after_missing = {
        column: int(after_data[column].map(comparison_cell_is_missing).sum())
        for column in after_data.columns
    }
    before_summary = (len(before_data), len(before_data.columns), sum(before_missing.values()))
    after_summary = (len(after_data), len(after_data.columns), sum(after_missing.values()))

    card_columns = st.columns(3)
    for card, label, before_count, after_count in zip(
        card_columns,
        ("Rows", "Columns", "Missing cells"),
        before_summary,
        after_summary,
    ):
        card.metric(
            label,
            f"{before_count} → {after_count}",
            delta=f"Δ {after_count - before_count:+d}",
            delta_color="off",
        )
    st.caption("Deltas are AFTER − BEFORE. Difference labels describe version changes, not data quality.")

    before_column_set = set(before_data.columns)
    after_column_set = set(after_data.columns)
    common_columns = [column for column in before_data.columns if column in after_column_set]
    added_columns = [column for column in after_data.columns if column not in before_column_set]
    removed_columns = [column for column in before_data.columns if column not in after_column_set]

    with st.expander("Added and removed column names", expanded=False):
        if added_columns:
            st.markdown(":green[**Added**]")
            st.write(added_columns)
        else:
            st.markdown(":gray[**Unchanged**] — no columns were added.")
        if removed_columns:
            st.markdown(":red[**Removed**]")
            st.write(removed_columns)
        else:
            st.markdown(":gray[**Unchanged**] — no columns were removed.")

    with st.expander("Missing counts for common columns", expanded=False):
        if not common_columns:
            st.info("There are no column names shared by both files to compare.")
        else:
            missing_rows = []
            changed_columns = []
            unchanged_columns = []
            for column in common_columns:
                before_count = before_missing[column]
                after_count = after_missing[column]
                status = "Changed" if before_count != after_count else "Unchanged"
                (changed_columns if status == "Changed" else unchanged_columns).append(column)
                missing_rows.append(
                    {
                        "Column name": column,
                        "BEFORE missing": before_count,
                        "AFTER missing": after_count,
                        "Delta (AFTER − BEFORE)": after_count - before_count,
                        "Difference": status,
                    }
                )
            st.dataframe(pd.DataFrame(missing_rows), width="stretch", hide_index=True)
            if changed_columns:
                st.markdown(":orange[**Changed**] — missing-cell counts differ for:")
                st.write(changed_columns)
            if unchanged_columns:
                st.markdown(":gray[**Unchanged**] — missing-cell counts match for:")
                st.write(unchanged_columns)

    with st.expander("Added and removed records (requires a valid common ID)", expanded=False):
        if not common_columns:
            st.info("Record matching is unavailable because there is no column shared by both files to use as an ID.")
        else:
            selected_id = st.selectbox(
                "Common ID column (exact text match)",
                common_columns,
                index=None,
                placeholder="Choose an ID column",
            )
            if selected_id is None:
                st.info("Record matching is unavailable until you select a common ID column. No record counts are inferred.")
            else:
                before_ids = before_data[selected_id]
                after_ids = after_data[selected_id]
                before_blank_ids = before_ids.map(comparison_cell_is_missing)
                after_blank_ids = after_ids.map(comparison_cell_is_missing)
                before_duplicate_ids = before_ids.loc[~before_blank_ids].duplicated(keep=False).any()
                after_duplicate_ids = after_ids.loc[~after_blank_ids].duplicated(keep=False).any()
                id_issues = []
                if before_blank_ids.any():
                    id_issues.append("BEFORE contains blank IDs")
                if after_blank_ids.any():
                    id_issues.append("AFTER contains blank IDs")
                if before_duplicate_ids:
                    id_issues.append("BEFORE contains duplicate nonblank IDs")
                if after_duplicate_ids:
                    id_issues.append("AFTER contains duplicate nonblank IDs")

                if id_issues:
                    st.info("Record matching is unavailable: " + "; ".join(id_issues) + ".")
                else:
                    before_id_set = set(before_ids.tolist())
                    after_id_set = set(after_ids.tolist())
                    added_ids = [value for value in after_ids if value not in before_id_set]
                    removed_ids = [value for value in before_ids if value not in after_id_set]
                    st.caption("IDs are matched exactly as text; leading zeros and row order are preserved.")

                    if added_ids:
                        st.markdown(f":green[**Added**] — {len(added_ids)} record(s).")
                        added_records = after_data.loc[after_ids.isin(set(added_ids))].copy()
                        st.dataframe(added_records, width="stretch", hide_index=True)
                    else:
                        st.markdown(":gray[**Unchanged**] — Added records: 0.")
                    if removed_ids:
                        st.markdown(f":red[**Removed**] — {len(removed_ids)} record(s).")
                        removed_records = before_data.loc[before_ids.isin(set(removed_ids))].copy()
                        st.dataframe(removed_records, width="stretch", hide_index=True)
                    else:
                        st.markdown(":gray[**Unchanged**] — Removed records: 0.")


st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --explorer-main: #22ff05;
        --explorer-background: #137166;
        --explorer-ink: #103c37;
        --explorer-paper: #f4fff8;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--explorer-background);
        color: var(--explorer-paper);
    }
    [data-testid="stMain"], [data-testid="stHeader"] { background: transparent; }
    [data-testid="stHeader"] { color: var(--explorer-paper); }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    h2, h3, p, label, [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] {
        color: var(--explorer-paper);
    }
    [data-testid="stMetricValue"] { color: var(--explorer-main); }
    [data-testid="stMetricLabel"] { color: var(--explorer-paper); }
    [data-testid="stSelectbox"] div[role="combobox"] {
        background: var(--explorer-paper);
        color: var(--explorer-ink);
        border-color: var(--explorer-main);
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--explorer-paper);
        color: var(--explorer-ink);
        border: 2px dashed var(--explorer-main);
        border-radius: 14px;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--explorer-main);
        color: #12362e;
        border: 0;
    }
    [data-testid="stDataFrame"] {
        background: var(--explorer-paper);
        border: 1px solid rgba(34, 255, 5, .65);
        border-radius: 12px;
        overflow: hidden;
    }
    .explorer-hero {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(250px, .9fr);
        align-items: center;
        gap: 1rem;
        margin: 0 0 1.6rem;
        padding: clamp(1.25rem, 3vw, 2rem);
        border: 1px solid var(--explorer-main);
        border-radius: 22px;
        background: linear-gradient(120deg, #0d5c54, #105f55);
        box-shadow: 0 14px 34px rgba(5, 37, 33, .22);
    }
    .explorer-kicker {
        margin: 0 0 .45rem;
        color: var(--explorer-main);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .explorer-hero h1 {
        margin: 0;
        color: var(--explorer-main);
        font-size: clamp(2rem, 4vw, 3.3rem);
        line-height: 1.08;
        letter-spacing: -.035em;
    }
    .explorer-hero-copy {
        max-width: 38rem;
        margin: .8rem 0 0;
        color: var(--explorer-paper);
        font-size: 1.05rem;
        line-height: 1.55;
    }
    .explorer-art {
        display: flex;
        align-items: end;
        justify-content: center;
        gap: clamp(.5rem, 2.3vw, 1.5rem);
        min-height: 140px;
    }
    .explorer-cat {
        width: clamp(88px, 11vw, 128px);
        height: auto;
        overflow: visible;
        transform-origin: center bottom;
        filter: drop-shadow(0 8px 7px rgba(0, 20, 17, .2));
        animation: explorer-laugh 2.2s ease-in-out infinite;
    }
    .explorer-cat-two { animation-delay: .3s; }
    @keyframes explorer-laugh {
        0%, 100% { transform: rotate(0) translateY(0); }
        20% { transform: rotate(-3deg) translateY(-3px); }
        45% { transform: rotate(2deg) translateY(0); }
        70% { transform: rotate(-2deg) translateY(-2px); }
    }
    @media (max-width: 680px) {
        .explorer-hero { grid-template-columns: 1fr; }
        .explorer-art { min-height: 115px; justify-content: flex-start; }
    }
    @media (prefers-reduced-motion: reduce) {
        .explorer-cat { animation: none; }
    }
    </style>
    <section class="explorer-hero" aria-label="Dataset Explorer welcome banner">
      <div>
        <p class="explorer-kicker">A friendly look at your data</p>
        <h1>Dataset Explorer</h1>
        <p class="explorer-hero-copy">Upload a CSV to explore its rows, columns, missing values, and category totals.</p>
      </div>
      <div class="explorer-art" role="img" aria-label="Two laughing cat caricatures">
        <svg class="explorer-cat explorer-cat-one" viewBox="0 0 130 140" aria-hidden="true">
          <path d="M19 135c2-20 15-31 46-31s44 11 46 31z" fill="#ffb54a" stroke="#fff1d6" stroke-width="3"/>
          <path d="M25 66 16 17l39 28M105 66l9-49-39 28" fill="#e78c34" stroke="#713f2d" stroke-width="4" stroke-linejoin="round"/>
          <path d="M27 64c0-24 16-40 38-40s38 16 38 40v17c0 23-17 40-38 40S27 104 27 81z" fill="#ffb54a" stroke="#713f2d" stroke-width="4"/>
          <path d="m27 59 18-12m-13 26 13-8m58-6L85 47m13 26-13-8" fill="none" stroke="#c36c2b" stroke-width="5" stroke-linecap="round"/>
          <path d="M42 69q7-9 14 0m18 0q7-9 14 0" fill="none" stroke="#422d2a" stroke-width="4" stroke-linecap="round"/>
          <path d="m58 81 7 5 7-5q-2 9-7 9t-7-9" fill="#a64f57" stroke="#713f2d" stroke-width="2" stroke-linejoin="round"/>
          <path d="M65 90q-13 0-15 9 0 12 15 12t15-12q-2-9-15-9z" fill="#923b50" stroke="#713f2d" stroke-width="3"/>
          <path d="M58 105q7 5 14 0" fill="none" stroke="#ff9eae" stroke-width="4" stroke-linecap="round"/>
          <path d="m44 85-25-5m25 12-28 2m70-9 25-5m-25 12 28 2" fill="none" stroke="#fff1d6" stroke-width="2.5" stroke-linecap="round"/>
          <circle cx="39" cy="84" r="5" fill="#ff7f83" opacity=".75"/><circle cx="91" cy="84" r="5" fill="#ff7f83" opacity=".75"/>
        </svg>
        <svg class="explorer-cat explorer-cat-two" viewBox="0 0 130 140" aria-hidden="true">
          <path d="M19 135c2-20 15-31 46-31s44 11 46 31z" fill="#b9b2ff" stroke="#f2efff" stroke-width="3"/>
          <path d="M25 66 16 17l39 28M105 66l9-49-39 28" fill="#aaa1e8" stroke="#49405e" stroke-width="4" stroke-linejoin="round"/>
          <path d="M27 64c0-24 16-40 38-40s38 16 38 40v17c0 23-17 40-38 40S27 104 27 81z" fill="#c5bdf2" stroke="#49405e" stroke-width="4"/>
          <path d="M34 39q31-23 62 0" fill="none" stroke="#8278bd" stroke-width="6" stroke-linecap="round"/>
          <path d="M42 69q7-9 14 0m18 0q7-9 14 0" fill="none" stroke="#302b43" stroke-width="4" stroke-linecap="round"/>
          <path d="m58 81 7 5 7-5q-2 9-7 9t-7-9" fill="#9a526f" stroke="#49405e" stroke-width="2" stroke-linejoin="round"/>
          <path d="M65 90q-13 0-15 9 0 12 15 12t15-12q-2-9-15-9z" fill="#77425f" stroke="#49405e" stroke-width="3"/>
          <path d="M58 105q7 5 14 0" fill="none" stroke="#f3a5d4" stroke-width="4" stroke-linecap="round"/>
          <path d="m44 85-25-5m25 12-28 2m70-9 25-5m-25 12 28 2" fill="none" stroke="#fff8ff" stroke-width="2.5" stroke-linecap="round"/>
          <circle cx="39" cy="84" r="5" fill="#f38cc5" opacity=".8"/><circle cx="91" cy="84" r="5" fill="#f38cc5" opacity=".8"/>
        </svg>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

show_run_comparison()

separator_choice = st.selectbox("CSV separator", ["Comma (,)", "Semicolon (;)"])
separator = "," if separator_choice == "Comma (,)" else ";"
show_version_comparison(separator)
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    help="The file must use UTF-8 encoding, include a header row, and be no larger than 5 MB.",
)

if uploaded_file is None:
    st.info("Choose a CSV file to get started.")
    st.subheader("Inspect numeric values")
    st.info("Upload a CSV file to choose a numeric column and inspect its values.")
else:
    file_bytes = uploaded_file.getvalue()

    if not uploaded_file.name.lower().endswith(".csv"):
        st.error("Wrong file type. Please choose a file with the .csv extension.")
    elif not file_bytes:
        st.error("This file is empty. Please choose a CSV file with a header and data.")
    elif len(file_bytes) > MAX_FILE_SIZE:
        st.error("This file is larger than 5 MB. Please choose a smaller CSV file.")
    else:
        try:
            data = pd.read_csv(
                BytesIO(file_bytes),
                sep=separator,
                header=0,
                encoding="utf-8-sig",
                keep_default_na=True,
                na_filter=True,
                skip_blank_lines=True,
            )
            if len(data.columns) == 0:
                st.error("This file has no header row. Please choose a CSV file with column names.")
            else:
                row_count, column_count = data.shape
                metric_columns = st.columns(4)
                metric_columns[0].metric("Rows", row_count)
                metric_columns[1].metric("Columns", column_count)
                metric_columns[2].metric("Extra exact duplicate rows", int(data.duplicated(keep="first").sum()))
                space_only_counts = data.apply(
                    lambda column: column.map(
                        lambda value: isinstance(value, str) and bool(value) and value.strip(" ") == ""
                    ).sum()
                ).astype(int)
                metric_columns[3].metric("Space-only cells", int(space_only_counts.sum()))

                st.subheader("Column information")
                missing_counts = data.isna().sum()
                if row_count:
                    missing_percentages = missing_counts / row_count * 100
                else:
                    missing_percentages = pd.Series(0.0, index=data.columns)
                column_info = pd.DataFrame(
                    {
                        "Column name": data.columns.astype(str),
                        "Pandas type": [str(dtype) for dtype in data.dtypes],
                        "Missing count": missing_counts.to_numpy(),
                        "Missing percent": [f"{value:.2f}%" for value in missing_percentages],
                        "Space-only count": space_only_counts.to_numpy(),
                    }
                )
                st.dataframe(column_info, width="stretch", hide_index=True)

                st.caption(
                    "Missing values follow pandas read_csv defaults: empty fields and recognized default "
                    "NA markers (such as NaN or NA) count as missing. Strings made only of one or more "
                    "ordinary spaces are counted separately above, not as ordinary missing values. 0 and '0' "
                    "are not counted as missing or space-only. Blank lines are skipped. The uploaded values "
                    "are not cleaned or changed."
                )

                st.subheader("Preview (first 100 rows)")
                st.dataframe(data.head(100), width="stretch")

                show_numeric_inspection(data)

                st.subheader("Category totals")
                if column_count == 0:
                    st.info("The chart is unavailable because this file has no columns to select.")
                else:
                    non_numeric_columns = [column for column in data.columns if not pd.api.types.is_numeric_dtype(data[column].dtype)]
                    default_category = non_numeric_columns[0] if non_numeric_columns else data.columns[0]
                    category_column = st.selectbox(
                        "Category column",
                        list(data.columns),
                        index=list(data.columns).index(default_category),
                    )
                    missing_category_rows = int(data[category_column].isna().sum())
                    st.caption(f"Rows with a missing category in '{category_column}': {missing_category_rows}")

                    numeric_columns = [
                        column for column in data.columns
                        if pd.api.types.is_numeric_dtype(data[column].dtype) and column != category_column
                    ]
                    if not numeric_columns:
                        if not any(pd.api.types.is_numeric_dtype(data[column].dtype) for column in data.columns):
                            st.info(
                                "The chart is unavailable because this CSV has no columns pandas read as numeric. "
                                "The overview above is still available."
                            )
                        else:
                            st.info(
                                "The chart needs a separate numeric column. Choose a different category column "
                                "or add another numeric column. The overview above is still available."
                            )
                    else:
                        numeric_column = st.selectbox("Numeric column", numeric_columns)
                        grouped = data.groupby(category_column, dropna=True, sort=False)[numeric_column].agg(
                            total="sum", available_values="count"
                        )
                        groups_with_data = grouped[grouped["available_values"] > 0].sort_values(
                            "total", ascending=False, kind="stable"
                        )
                        groups_without_data = grouped[grouped["available_values"] == 0]

                        total_column = f"Sum of {numeric_column}"
                        totals_rows = [
                            {category_column: category, total_column: row["total"]}
                            for category, row in groups_with_data.iterrows()
                        ]
                        totals_rows.extend(
                            {category_column: category, total_column: "No data"}
                            for category in groups_without_data.index
                        )
                        totals_table = pd.DataFrame(totals_rows, columns=[category_column, total_column])
                        if not totals_table.empty:
                            totals_table[total_column] = totals_table[total_column].map(str)

                        st.caption(
                            f"Aggregation: sum of non-missing '{numeric_column}' values for each '{category_column}'. "
                            "Missing numbers are ignored, not replaced with zero. Rows with missing categories "
                            "are excluded from groups and reported above. Groups with no numeric values show No data "
                            "in the table and are omitted from the chart."
                        )

                        if len(groups_with_data) > 20:
                            st.caption(
                                f"The chart shows only the top 20 of {len(groups_with_data)} categories with data, "
                                f"sorted by total (highest first). The table below includes all {len(totals_table)} "
                                "category totals."
                            )
                        elif len(groups_with_data):
                            st.caption(
                                f"The chart shows all {len(groups_with_data)} categories with data, sorted by total "
                                "(highest first). The table includes all category totals."
                            )
                        else:
                            st.info("There are no categories with numeric values to chart. The totals table is below.")

                        if len(groups_with_data):
                            chart_rows = [
                                {category_column: category, total_column: row["total"]}
                                for category, row in groups_with_data.head(20).iterrows()
                            ]
                            st.bar_chart(
                                pd.DataFrame(chart_rows),
                                x=category_column,
                                y=total_column,
                                x_label=category_column,
                                y_label=f"Sum of {numeric_column}",
                            )

                        st.caption("Exact totals for all categories:")
                        st.dataframe(totals_table, width="stretch", hide_index=True)
        except pd.errors.EmptyDataError:
            st.error("This file is empty or has no readable header. Please choose a non-empty CSV file.")
        except Exception as error:
            st.error(f"This file could not be read as UTF-8 CSV. Check the file and separator, then try again. Details: {error}")
