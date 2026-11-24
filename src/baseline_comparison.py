import os
import time
from typing import Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import prepare_dataset_pair, restore_imputed_data
from src.evaluation import calculate_imputation_metrics, run_imputation_pipeline
from src.gain import train_gain


METHODS = ["GAIN", "Mean", "Median"]

#methods for imputation: mean
def mean_imputation(data, mask, categorical_columns=()):
    """Fill missing values with observed column means or categorical modes."""
    arr = data.copy()
    categorical_columns = set(categorical_columns)
    for col_idx in range(arr.shape[1]):
        observed = arr[:, col_idx][mask[:, col_idx].astype(bool)]
        if observed.size == 0:
            fill_value = 0.0
        elif col_idx in categorical_columns:
            # Categorical baselines use the most frequent observed code (the mode).
            values, counts = np.unique(observed, return_counts=True)
            fill_value = float(values[np.argmax(counts)])
        else:
            fill_value = float(np.mean(observed))
        arr[mask[:, col_idx] == 0, col_idx] = fill_value
    return arr


#methods for imputation: Median
def median_imputation(data, mask, categorical_columns=()):
    """Fill missing values with observed column medians or categorical modes."""
    arr = data.copy()
    categorical_columns = set(categorical_columns)
    for col_idx in range(arr.shape[1]):
        observed = arr[:, col_idx][mask[:, col_idx].astype(bool)]
        if observed.size == 0:
            fill_value = 0.0
        elif col_idx in categorical_columns:
            values, counts = np.unique(observed, return_counts=True)
            fill_value = float(values[np.argmax(counts)])
        else:
            fill_value = float(np.median(observed))
        arr[mask[:, col_idx] == 0, col_idx] = fill_value
    return arr


#methods for imputation: GAIN
def gain_imputation(data, mask, epochs=200, batch_size=32, alpha=100, lr=1e-3):
    """Train GAIN and return the imputed scaled data."""
    generator = train_gain(data, mask, epochs=epochs, batch_size=batch_size, alpha=alpha, lr=lr)
    imputed_scaled = run_imputation_pipeline(generator, data, mask)
    return imputed_scaled


#save the imputed data to an Excel file
def save_imputed_file(dataset_name, method, raw_df, mask, imputed_data, output_dir):
    """Insert imputed values into the original dataset and save an Excel file."""
    df_imputed = raw_df.copy()
    missing_positions = np.argwhere(mask == 0)
    for row_idx, col_idx in missing_positions:
        df_imputed.iat[row_idx, col_idx] = imputed_data[row_idx, col_idx]

    output_path = os.path.join(output_dir, f"{dataset_name}_{method}_imputed.xlsx")
    df_imputed.to_excel(output_path, index=False, header=False)
    return output_path

#Run and score one imputation method for one dataset
def evaluate_single_method(dataset_name, incomplete_path, original_path, method, output_dir, epochs=200, batch_size=32, alpha=100, lr=1e-3):
    prepared = prepare_dataset_pair(incomplete_path, original_path)
    incomplete_data = prepared.incomplete_data
    mask = prepared.mask
    categorical_columns = prepared.categorical_columns
    original_data_raw = prepared.original_df.to_numpy(dtype=object)
    start = time.perf_counter()

    if method == "GAIN":
        imputed_scaled = gain_imputation(incomplete_data, mask, epochs=epochs, batch_size=batch_size, alpha=alpha, lr=lr)
    elif method == "Mean":
        imputed_scaled = mean_imputation(incomplete_data, mask, categorical_columns)
    elif method == "Median":
        imputed_scaled = median_imputation(incomplete_data, mask, categorical_columns)
    else:
        raise ValueError(f"Unsupported method: {method}")

    elapsed = time.perf_counter() - start
    imputed_data = restore_imputed_data(imputed_scaled, prepared.scaler, prepared.category_values)
    metrics = calculate_imputation_metrics(original_data_raw, imputed_data, mask, categorical_columns)

    save_imputed_file(dataset_name, method, prepared.incomplete_df, mask, imputed_data, output_dir)

    return {
        "Dataset": dataset_name,
        "Method": method,
        **metrics,
        "Runtime_s": float(elapsed),
        "Missing_Rate": float((mask == 0).mean()),
        "Imputed_File": os.path.join(output_dir, f"{dataset_name}_{method}_imputed.xlsx"),
    }


def build_dataset_list(results_xlsx_path="Table-NRMS-AE.xlsx", dataset_names=None, start_index=None, end_index=None):
    #select dataset variants from the results workbook and optional filters
    if not os.path.exists(results_xlsx_path):
        raise FileNotFoundError(f"Results table {results_xlsx_path} not found.")

    df_results = pd.read_excel(results_xlsx_path)
    if dataset_names is None:
        dataset_names = df_results["Datasets"].dropna().tolist()

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    selected = []
    for idx, row in df_results.iterrows():
        dataset_name = row.get("Datasets")
        if pd.isna(dataset_name):
            continue
        dataset_name = str(dataset_name).strip()
        if dataset_names is not None and dataset_name not in dataset_names:
            base_name = dataset_name.split('_')[0]
            if base_name not in dataset_names and not any(dataset_name.startswith(f"{name}_") for name in dataset_names):
                continue
        if start_index is not None and idx < start_index:
            continue
        if end_index is not None and idx >= end_index:
            continue
        selected.append(dataset_name)
    return selected


def _save_excel_only(df, output_path):
    #Save a DataFrame as an XLSX file and return the resulting path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".xlsx"):
        df.to_excel(output_path, index=False)
        return output_path

    xlsx_path = output_path + ".xlsx"
    df.to_excel(xlsx_path, index=False)
    return xlsx_path


def _format_seconds(seconds):
    #Format a runtime in milliseconds, seconds, minutes, or hours.
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def _format_metric(value):
    #Format a metric for progress output.
    return "N/A" if value is None or pd.isna(value) else f"{value:.4f}"

#Write per-method score files and a combined method summary
def write_method_score_files(df, output_dir, methods=None):
    selected_methods = METHODS if methods is None else methods
    for method in selected_methods:
        subset = df[df["Method"] == method][["Dataset", "NRMS", "AE", "Runtime_s", "Missing_Rate", "Imputed_File"]].copy()
        method_path = os.path.join(output_dir, f"{method.lower()}_scores.xlsx")
        xlsx_path = _save_excel_only(subset, method_path)
        print(f"Saved {method} scores: {xlsx_path}")

    summary = df.groupby("Method")["NRMS"].mean().reset_index().rename(columns={"NRMS": "Mean_NRMS"})
    summary2 = df.groupby("Method")["AE"].mean().reset_index().rename(columns={"AE": "Mean_AE"})
    summary3 = df.groupby("Method")["Runtime_s"].mean().reset_index().rename(columns={"Runtime_s": "Mean_Runtime_s"})
    summary = summary.merge(summary2, on="Method").merge(summary3, on="Method")
    summary_path = os.path.join(output_dir, "method_summary.xlsx")
    xlsx_path = _save_excel_only(summary, summary_path)
    print(f"Saved method summary: {xlsx_path}")
    return summary


def generate_comparison_charts(df, output_dir):
    #Create NRMS and categorical-accuracy comparison charts.
    if df.empty:
        return []

    required = ["Dataset", "Method", "NRMS", "AE"]
    for col in required:
        if col not in df.columns:
            return []

    pivot = df.pivot(index="Dataset", columns="Method", values="NRMS")
    chart_paths = []
    if pivot.notna().any().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("NRMS by Method")
        ax.set_ylabel("NRMS")
        ax.set_xlabel("Dataset")
        ax.legend(title="Method")
        fig.tight_layout()
        nrmse_path = os.path.join(output_dir, "nrms_comparison.png")
        fig.savefig(nrmse_path, dpi=200)
        plt.close(fig)
        chart_paths.append(nrmse_path)

    pivot_ae = df.pivot(index="Dataset", columns="Method", values="AE")
    if pivot_ae.notna().any().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_ae.plot(kind="bar", ax=ax)
        ax.set_title("Categorical Accuracy (AE) by Method")
        ax.set_ylabel("AE (fraction correct)")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Dataset")
        ax.legend(title="Method")
        fig.tight_layout()
        ae_path = os.path.join(output_dir, "ae_comparison.png")
        fig.savefig(ae_path, dpi=200)
        plt.close(fig)
        chart_paths.append(ae_path)

    if chart_paths:
        print(f"Saved charts: {', '.join(chart_paths)}")
    return chart_paths

#Comaprison of the performance of GAIN, Mean, Median methods
def compare_datasets(results_xlsx_path="Table-NRMS-AE.xlsx", incomplete_dir="data/incomplete_datasets", original_dir="data/original_datasets", dataset_names=None, start_index=None, end_index=None, output_path="comparison_results.xlsx", methods=None):
    """Compare selected imputation methods across selected datasets."""
    selected_names = build_dataset_list(results_xlsx_path, dataset_names, start_index, end_index)
    selected_methods = METHODS if methods is None else [
        method for method in METHODS
        if method.lower() in {str(value).strip().lower() for value in methods}
    ]
    if not selected_methods:
        raise ValueError("No supported methods selected. Use GAIN, Mean, or Median.")

    output_dir = os.path.dirname(output_path) or "data/results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 90}\nComparison run for {len(selected_names)} dataset(s) using {', '.join(selected_methods)}\n{'=' * 90}")

    rows = []
    method_times = []
    total_datasets = len(selected_names)
    total_methods = total_datasets * len(selected_methods)
    for dataset_position, dataset_name in enumerate(selected_names, start=1):
        print(f"\n{'=' * 90}\nDATASET {dataset_position}/{total_datasets}: {dataset_name}\n{'=' * 90}")
        base_dataset = dataset_name.split('_')[0]
        incomplete_path = os.path.join(incomplete_dir, base_dataset, f"{dataset_name}.xlsx")
        original_path = os.path.join(original_dir, f"{base_dataset}.xlsx")
        if not os.path.exists(incomplete_path):
            incomplete_path = os.path.join(incomplete_dir, f"{dataset_name}.xlsx")

        if not os.path.exists(incomplete_path) or not os.path.exists(original_path):
            print(f"Skipping {dataset_name}: missing files.")
            continue

        for method_position, method in enumerate(selected_methods, start=1):
            completed_methods = len(method_times)
            if method_times:
                average_method_time = sum(method_times) / len(method_times)
                eta = _format_seconds(average_method_time * (total_methods - completed_methods))
            else:
                eta = "pending (first method)"
            print(f"\nProgress: [{completed_methods}/{total_methods}] | Method {method_position}/{len(selected_methods)}: {method} | ETA: {eta}", flush=True)
            result = evaluate_single_method(dataset_name, incomplete_path, original_path, method, output_dir=output_dir)
            rows.append(result)
            method_times.append(result["Runtime_s"])
            completed_methods = len(method_times)
            average_method_time = sum(method_times) / completed_methods
            eta = _format_seconds(average_method_time * (total_methods - completed_methods))
            print(f"Completed {method} for {dataset_name} | NRMS: {_format_metric(result['NRMS'])} | AE: {_format_metric(result['AE'])} | Runtime: {_format_seconds(result['Runtime_s'])} | Progress: [{completed_methods}/{total_methods}] | ETA: {eta}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No comparison results generated.")
        return df

    xlsx_path = _save_excel_only(df, output_path)
    write_method_score_files(df, output_dir, selected_methods)
    generate_comparison_charts(df, output_dir)

    print(f"\nComparison saved to {xlsx_path}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare GAIN against Mean and Median imputation")
    parser.add_argument("--dataset", type=str, help="Run a specific dataset or base name, e.g. 4-gauss_NN_1 or 4-gauss")
    parser.add_argument("--datasets", type=str, help="Comma-separated list of dataset names")
    parser.add_argument("--start", type=int, help="Start row index from Table-NRMS-AE.xlsx")
    parser.add_argument("--end", type=int, help="End row index from Table-NRMS-AE.xlsx")
    parser.add_argument("--output", type=str, default="comparison_results.xlsx", help="Output Excel path")
    parser.add_argument("--methods", type=str, help="Comma-separated methods: GAIN, Mean, Median")
    args = parser.parse_args()

    selected = None
    if args.dataset:
        selected = [args.dataset]
    if args.datasets:
        values = [v.strip() for v in args.datasets.split(',') if v.strip()]
        selected = values if selected is None else selected + values
    methods = None
    if args.methods:
        methods = [value.strip() for value in args.methods.split(',') if value.strip()]

    compare_datasets(
        results_xlsx_path="Table-NRMS-AE.xlsx",
        dataset_names=selected,
        start_index=args.start,
        end_index=args.end,
        output_path=args.output,
        methods=methods,
    )
