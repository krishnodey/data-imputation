# GAIN Missing Data Imputation Project

This project implements a GAIN-based imputation pipeline for missing-value recovery in tabular datasets and compares it against simple baseline methods such as mean and median imputation.

## Project Overview

The project is designed to:
- load incomplete datasets from Excel or CSV files
- create and validate missing-data masks
- train a GAIN model for imputation
- evaluate imputation performance using NRMS
- compare GAIN against mean and median baselines
- save imputed datasets and comparison outputs for reporting

## Folder Structure

```text
gain_project/
├── main.py
├── README.md
├── Table-NRMS-AE.xlsx
├── data/
│   ├── incomplete_datasets/
│   ├── original_datasets/
│   └── results/
├── src/
│   ├── analysis.py
│   ├── batch_evaluation.py
│   ├── baseline_comparison.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── gain.py
│   └── __init__.py
└── venv/
```

## Requirements

Python 3.10+

Install dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the main dependencies manually:

```bash
pip install numpy pandas scikit-learn torch openpyxl matplotlib
```

## Setup

Activate the virtual environment:

```bash
cd /Users/classroomservices/Documents/gain_project
source venv/bin/activate
```

## Running the Full Pipeline

To run the full batch evaluation using the results table:

```bash
python3 main.py
```

## Running a Single Dataset

Run one dataset by name:

```bash
python3 main.py 4-gauss_NN_1
```

Or with the flag form:

```bash
python3 main.py --dataset 4-gauss_NN_1
```

You can also pass multiple datasets:

```bash
python3 main.py --datasets 4-gauss_NN_1,4-gauss_NN_10
```

Or a row range from the Excel results table:

```bash
python3 main.py --start 0 --end 50
```

## Running Baseline Comparisons

Compare GAIN against mean and median imputation:

```bash
python3 main.py --compare --dataset 4-gauss_NN_1
```

Compare multiple datasets:

```bash
python3 main.py --compare --datasets 4-gauss_NN_1,4-gauss_NN_10
```

## Output Files

The project generates output in `data/results/`.

### Batch evaluation outputs
- imputed dataset files with names like:
  - `4-gauss_NN_1_imputed.xlsx`
- updated results table:
  - `Table-NRMS-AE.xlsx`

### Comparison outputs
- `comparison_results.xlsx` in the project root
- `gain_scores.xlsx`
- `mean_scores.xlsx`
- `median_scores.xlsx`
- `method_summary.xlsx`
- `nrms_comparison.png`
- `ae_comparison.png`
- individual imputed files such as:
  - `4-gauss_NN_1_GAIN_imputed.xlsx`
  - `4-gauss_NN_1_Mean_imputed.xlsx`
  - `4-gauss_NN_1_Median_imputed.xlsx`

## Evaluation Metric

Imputation metrics follow `Instructions.pdf` and are evaluated only at missing cells:

```text
NRMS = ||X_original - X_estimated||_F / ||X_original||_F
AE = number of correctly imputed categorical values / number of missing categorical values
```

Numeric attributes are evaluated with NRMS; categorical attributes are evaluated with AE. Mixed datasets report both, with a blank value when a metric does not apply. Runtime is recorded separately in `Runtime_s` and is not the AE metric. Mean and median baselines use the mode for categorical attributes.

Feature types are inferred from string-valued columns and integer-valued columns with at most 20 distinct values. `Letter` is treated as numeric because its integer features are pixel measurements. Numeric-coded columns that do not fit this rule can be declared in `FEATURE_TYPE_OVERRIDES` in `src/data_loader.py`.

## Main Components

### `src/data_loader.py`
Loads dataset files, creates the missing-value mask, fills missing values for the model input, and performs MinMax scaling.

### `src/gain.py`
Contains the GAIN generator/discriminator model and training loop.

### `src/evaluation.py`
Contains evaluation functions, including NRMS calculation and the imputation pipeline for generating filled values.

### `src/batch_evaluation.py`
Runs multiple datasets from the Excel results table and writes metric values back to the workbook.

### `src/baseline_comparison.py`
Compares GAIN with mean and median imputation, saves the imputed versions, and generates summary tables and charts.

## Notes

- The dataset files are assumed to be matrix-style tables with no header row unless explicitly modified.
- The project reads raw datasets with `header=None` so missing values are inserted back into the exact original coordinates.
- For comparison studies, all methods are evaluated on the same missing mask and original dataset to ensure a fair comparison.

## Typical Workflow

1. Prepare incomplete datasets in `data/incomplete_datasets/`
2. Prepare complete versions in `data/original_datasets/`
3. Update `Table-NRMS-AE.xlsx` with dataset names if needed
4. Run a dataset or batch
5. Compare GAIN with mean and median methods using `--compare`
6. Export result tables and figures for your report

## Example Commands

```bash
# single dataset
python3 main.py 4-gauss_NN_1

# single dataset with comparison
python3 main.py --compare --dataset 4-gauss_NN_1

# multiple datasets
python3 main.py --compare --datasets 4-gauss_NN_1,4-gauss_NN_10

# full batch run
python3 main.py
```

## License

This project is intended for academic and research use.
