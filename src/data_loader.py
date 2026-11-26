import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


MAX_INTEGER_CATEGORIES = 20
# Table names whose data files are spelled differently (table name -> file name).
DATASET_FILE_ALIASES = {
    "TTTEG": "TTTTEG",
}
FEATURE_TYPE_OVERRIDES = {
    "letter": {column_idx: "numeric" for column_idx in range(16)},
}


@dataclass
class PreparedDataset:
    """Store prepared data, masks, encodings, and scaling information."""
    incomplete_data: np.ndarray
    mask: np.ndarray
    scaler: MinMaxScaler
    original_df: pd.DataFrame
    incomplete_df: pd.DataFrame
    categorical_columns: tuple
    category_values: dict


def resolve_dataset_paths(dataset_name, incomplete_dir, original_dir):
    #Return the incomplete and original file paths for a dataset variant.
    base_dataset = dataset_name.split('_')[0]
    file_base = DATASET_FILE_ALIASES.get(base_dataset, base_dataset)
    file_name = file_base + dataset_name[len(base_dataset):]
    incomplete_path = os.path.join(incomplete_dir, file_base, f"{file_name}.xlsx")
    if not os.path.exists(incomplete_path):
        incomplete_path = os.path.join(incomplete_dir, f"{file_name}.xlsx")
    original_path = os.path.join(original_dir, f"{file_base}.xlsx")
    return incomplete_path, original_path


def _read_dataframe(file_path):
    #Read a headerless CSV or Excel dataset.
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, header=None)
    if file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path, header=None)
    raise ValueError("Unsupported file format. Use CSV or Excel.")


def _category_key(value):
    #Return a stable key for matching original category values.
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
    return str(value).strip()


def _is_categorical_column(series, dataset_name, column_idx):
    #Determine whether a dataset column should be treated as categorical.
    override = FEATURE_TYPE_OVERRIDES.get(dataset_name.lower(), {}).get(column_idx)
    if override is not None:
        return override == "categorical"

    if (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return True

    values = pd.to_numeric(series.dropna(), errors="coerce").to_numpy(dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    integer_valued = np.allclose(values, np.round(values))
    return integer_valued and np.unique(values).size <= MAX_INTEGER_CATEGORIES


def prepare_dataset_pair(incomplete_path, original_path):
    #Prepare paired data for GAIN using a complete reference schema.
    incomplete_df = _read_dataframe(incomplete_path)
    original_df = _read_dataframe(original_path)
    if incomplete_df.shape != original_df.shape:
        raise ValueError(
            f"Incomplete and original data shapes differ: {incomplete_df.shape} vs {original_df.shape}"
        )

    dataset_name = Path(original_path).stem.lower()
    #The mask is preserved in the original matrix orientation for missing-cell evaluation.
    mask = (~incomplete_df.isna()).to_numpy(dtype=float)
    incomplete_matrix = np.empty(incomplete_df.shape, dtype=float)
    categorical_columns = []
    category_values = {}

    if original_df.isna().any().any():
        raise ValueError("Original dataset must not contain missing values for evaluation.")

    for column_idx in range(original_df.shape[1]):
        original_column = original_df.iloc[:, column_idx]
        incomplete_column = incomplete_df.iloc[:, column_idx]
        if _is_categorical_column(original_column, dataset_name, column_idx):
            categorical_columns.append(column_idx)
            #Stable integer codes let GAIN and the baselines operate on categorical columns numerically.
            label_by_key = {}
            for value in original_column:
                label_by_key.setdefault(_category_key(value), value)
            ordered_keys = sorted(label_by_key)
            code_by_key = {key: code for code, key in enumerate(ordered_keys)}
            category_values[column_idx] = tuple(label_by_key[key] for key in ordered_keys)
            incomplete_matrix[:, column_idx] = [
                np.nan if pd.isna(value) else code_by_key[_category_key(value)]
                for value in incomplete_column
            ]
        else:
            incomplete_matrix[:, column_idx] = pd.to_numeric(incomplete_column, errors="raise").to_numpy(dtype=float)

    #Fit on observed values only so hidden values never influence scaling.
    fit_matrix = incomplete_matrix.copy()
    for column_idx in np.flatnonzero(np.isnan(fit_matrix).all(axis=0)):
        #A fully missing column has no observed range: use its category codes, or [0, 1] if numeric.
        upper = len(category_values[column_idx]) - 1 if column_idx in category_values else 1.0
        fit_matrix[0, column_idx] = 0.0
        fit_matrix[-1, column_idx] = upper
    scaler = MinMaxScaler()
    scaler.fit(fit_matrix)
    incomplete_scaled = scaler.transform(incomplete_matrix)
    incomplete_scaled = np.nan_to_num(incomplete_scaled, nan=0.0)

    return PreparedDataset(
        incomplete_data=incomplete_scaled,
        mask=mask,
        scaler=scaler,
        original_df=original_df,
        incomplete_df=incomplete_df,
        categorical_columns=tuple(categorical_columns),
        category_values=category_values,
    )


def restore_imputed_data(imputed_scaled, scaler, category_values):
    #Inverse-scale imputed values and restore categorical labels.
    restored_numeric = scaler.inverse_transform(np.asarray(imputed_scaled, dtype=float))
    restored = restored_numeric.astype(object)
    for column_idx, labels in category_values.items():
        codes = np.rint(restored_numeric[:, column_idx]).astype(int)
        codes = np.clip(codes, 0, len(labels) - 1)
        restored[:, column_idx] = np.asarray(labels, dtype=object)[codes]
    return restored


def load_and_preprocess_data(file_path):
    #Load and scale a headerless CSV or Excel dataset.
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, header=None)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, header=None)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    #Create mask before any imputation: 1 if observed, 0 if missing
    mask = (~df.isna()).astype(float).values

    #Encode categorical or object columns positionally by index
    for col_idx in range(df.shape[1]):
        col_data = df.iloc[:, col_idx]
        if col_data.dtype == 'object' or pd.api.types.is_categorical_dtype(col_data):
            not_na_mask = col_data.notna()
            if not_na_mask.sum() > 0:
                le = LabelEncoder()
                transformed = le.fit_transform(col_data[not_na_mask].astype(str))
                df.iloc[not_na_mask, col_idx] = transformed.astype(float)
            df.iloc[:, col_idx] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')

    #Fill missing values with 0 for initial matrix X_bar
    df_filled = df.fillna(0.0)

    #Scale all features to [0, 1] using MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_filled.values)

    print(f"Loaded dataset {file_path} with shape {data_scaled.shape}")
    return data_scaled, mask, scaler