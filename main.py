import argparse
import os

from src.batch_evaluation import evaluate_all_datasets
from src.baseline_comparison import compare_datasets


def parse_dataset_list(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        values = [item.strip() for item in raw_value.split(',') if item.strip()]
        return values or None
    return [str(item).strip() for item in raw_value if str(item).strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIN dataset imputation runner")
    parser.add_argument("dataset_name", nargs="?", help="Single dataset name, e.g. 4-gauss_NN_1")
    parser.add_argument("--dataset", "-d", type=str, help="Run one dataset or base dataset name, e.g. 4-gauss or 4-gauss_NN_1")
    parser.add_argument("--datasets", type=str, help="Comma-separated dataset names to run together")
    parser.add_argument("--start", type=int, help="Row index in Table-NRMS-AE.xlsx to start from")
    parser.add_argument("--end", type=int, help="Row index in Table-NRMS-AE.xlsx to stop before")
    parser.add_argument("--compare", action="store_true", help="Compare GAIN against Mean and Median imputation")
    parser.add_argument("--compare-output", type=str, default="comparison_results.xlsx", help="Comparison Excel output path")
    args = parser.parse_args()

    selected_datasets = parse_dataset_list(args.datasets)
    positional_dataset = args.dataset_name

    if positional_dataset:
        selected_datasets = [positional_dataset] if selected_datasets is None else selected_datasets + [positional_dataset]

    if args.dataset:
        selected_datasets = [args.dataset] if selected_datasets is None else selected_datasets + [args.dataset]

    print("=== GAIN Data Mining Course Project Pipeline ===")

    results_path = "Table-NRMS-AE.xlsx"
    if args.compare:
        print("\n--- Running GAIN vs Mean vs Median comparison ---")
        compare_datasets(
            results_xlsx_path=results_path,
            dataset_names=selected_datasets,
            start_index=args.start,
            end_index=args.end,
            output_path=args.compare_output,
        )
    elif os.path.exists(results_path):
        print(f"\n--- Found {results_path}! Starting batch evaluation ---")
        evaluate_all_datasets(
            results_xlsx_path=results_path,
            dataset_names=selected_datasets,
            start_index=args.start,
            end_index=args.end,
        )
    else:
        print(f"Results table {results_path} not found. Nothing to run.")

    print("\nPipeline execution finished successfully!")
