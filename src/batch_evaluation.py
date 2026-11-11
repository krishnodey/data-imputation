import os
import time
import numpy as np
import pandas as pd
import torch
from src.data_loader import prepare_dataset_pair, restore_imputed_data
from src.gain import train_gain
from src.evaluation import calculate_imputation_metrics, run_imputation_pipeline

def _format_seconds(seconds):
    if seconds is None:
        return "unknown"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _progress_bar(current, total, width=30):
    if total <= 0:
        filled = width
    else:
        filled = max(0, min(width, int(round((current / total) * width))))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def evaluate_all_datasets(results_xlsx_path="Table-NRMS-AE.xlsx", incomplete_dir="data/incomplete_datasets", original_dir="data/original_datasets", result_dir="./data/results", dataset_names=None, start_index=None, end_index=None):
    """
    Automates loading each incomplete dataset variation, running GAIN,
    calculating numeric NRMS and categorical AE, recording runtime separately,
    and saving the imputed dataset into the ./result/ folder.

    dataset_names can be a string, list, or tuple and supports either exact names
    or their base dataset names such as '4-gauss'. If start_index/end_index are
    provided, only a slice of the table is evaluated.
    """
    if not os.path.exists(results_xlsx_path):
        print(f"Results table {results_xlsx_path} not found.")
        return

    os.makedirs(result_dir, exist_ok=True)
    df_results = pd.read_excel(results_xlsx_path)
    if "Runtime_s" not in df_results.columns:
        df_results["NRMS"] = np.nan
        df_results["AE"] = np.nan
        df_results["Runtime_s"] = np.nan

    if start_index is not None:
        start_index = max(0, int(start_index))
    else:
        start_index = 0

    if end_index is not None:
        end_index = min(len(df_results), int(end_index))
    else:
        end_index = len(df_results)

    selected_indices = list(range(start_index, end_index))

    if dataset_names is not None:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        dataset_names = [str(d).strip() for d in dataset_names if str(d).strip()]
        if dataset_names:
            selected_indices = []
            for idx, row in df_results.iterrows():
                dataset_name = row.get("Datasets")
                if pd.isna(dataset_name):
                    continue
                dataset_name = str(dataset_name).strip()
                base_dataset = dataset_name.split('_')[0]
                if any(dataset_name == d or base_dataset == d or dataset_name.startswith(f"{d}_") for d in dataset_names):
                    if start_index <= idx < end_index:
                        selected_indices.append(idx)

    if not selected_indices:
        print("No datasets selected. Nothing to run.")
        return

    print(f"Selected {len(selected_indices)} dataset(s) for evaluation.")

    print("\n" + "=" * 90)
    print("BATCH RUN STARTED")
    print("=" * 90, flush=True)

    observed_times = []
    total_jobs = len(selected_indices)
    for job_index, idx in enumerate(selected_indices, start=1):
        row = df_results.loc[idx]
        dataset_name = row.get("Datasets")
        if pd.isna(dataset_name):
            continue
        dataset_name = str(dataset_name).strip()

        print("\n" + "=" * 90)
        print(f"DATASET {job_index}/{total_jobs}: {dataset_name}")
        print("=" * 90)
        if observed_times:
            avg_time = sum(observed_times) / len(observed_times)
            remaining_jobs = max(0, total_jobs - job_index)
            est_remaining = avg_time * remaining_jobs
            print(f"Progress: {_progress_bar(job_index, total_jobs)}", flush=True)
            print(f"Average runtime so far: {_format_seconds(avg_time)} | ETA: {_format_seconds(est_remaining)} remaining", flush=True)
        else:
            print(f"Progress: {_progress_bar(job_index, total_jobs)}", flush=True)
            print("ETA: pending (first run)", flush=True)

        base_dataset = dataset_name.split('_')[0]
        incomplete_path = os.path.join(incomplete_dir, base_dataset, f"{dataset_name}.xlsx")
        original_path = os.path.join(original_dir, f"{base_dataset}.xlsx")

        if not os.path.exists(incomplete_path):
            incomplete_path = os.path.join(incomplete_dir, f"{dataset_name}.xlsx")

        if os.path.exists(incomplete_path) and os.path.exists(original_path):
            try:
                prepared = prepare_dataset_pair(incomplete_path, original_path)
                incomplete_data = prepared.incomplete_data
                mask = prepared.mask

                start_time = time.perf_counter()
                generator = train_gain(incomplete_data, mask, epochs=200, batch_size=32)
                imputed_data_scaled = run_imputation_pipeline(generator, incomplete_data, mask)
                execution_time = time.perf_counter() - start_time
                observed_times.append(execution_time)

                imputed_data = restore_imputed_data(
                    imputed_data_scaled, prepared.scaler, prepared.category_values
                )
                metrics = calculate_imputation_metrics(
                    prepared.original_df.to_numpy(dtype=object),
                    imputed_data,
                    mask,
                    prepared.categorical_columns,
                )

                df_results.at[idx, "NRMS"] = metrics["NRMS"]
                df_results.at[idx, "AE"] = metrics["AE"]
                df_results.at[idx, "Runtime_s"] = execution_time
                print(
                    f"Completed {dataset_name} | NRMS: {metrics['NRMS'] if metrics['NRMS'] is not None else 'N/A'}"
                    f" | AE: {metrics['AE'] if metrics['AE'] is not None else 'N/A'}"
                    f" | Runtime: {_format_seconds(execution_time)}",
                    flush=True,
                )

                df_imputed = prepared.incomplete_df.copy()
                missing_positions = np.argwhere(mask == 0)
                for row_idx, col_idx in missing_positions:
                    df_imputed.iat[row_idx, col_idx] = imputed_data[row_idx, col_idx]

                result_file_path = os.path.join(result_dir, f"{dataset_name}_imputed.xlsx")
                df_imputed.to_excel(result_file_path, index=False, header=False)
                print(f"Saved imputed file to {result_file_path}", flush=True)

            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
        else:
            print(f"Files not found for {dataset_name}. Skipping...")

    df_results.to_excel(results_xlsx_path, index=False)

    if observed_times:
        avg_time = sum(observed_times) / len(observed_times)
        total_estimate = avg_time * total_jobs
        print("\n" + "=" * 90)
        print(f"Batch evaluation complete! Results saved to {results_xlsx_path} and imputed files in {result_dir}/")
        print(f"Average dataset runtime: {_format_seconds(avg_time)} | Estimated total batch time: {_format_seconds(total_estimate)}")
    else:
        print("\n" + "=" * 90)
        print(f"Batch evaluation complete! Results saved to {results_xlsx_path} and imputed files in {result_dir}/")