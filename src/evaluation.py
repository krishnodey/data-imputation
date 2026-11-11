import numpy as np
import torch

def calculate_nrmse(original_data, imputed_data, mask):
    """
    Calculates the instruction-defined normalized Frobenius error for numeric data.
    """
    original = np.asarray(original_data, dtype=float)
    estimate = np.asarray(imputed_data, dtype=float)
    missing = np.asarray(mask) == 0
    if not np.any(missing):
        return None

    numerator = np.linalg.norm((original - estimate)[missing])
    denominator = np.linalg.norm(original)
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    return float(numerator / denominator)


def calculate_imputation_metrics(original_data, imputed_data, mask, categorical_columns):
    """Return NRMS for missing numeric cells and exact-match accuracy for categorical cells."""
    original = np.asarray(original_data, dtype=object)
    estimate = np.asarray(imputed_data, dtype=object)
    missing = np.asarray(mask) == 0
    categorical_columns = set(categorical_columns)
    numeric_columns = [idx for idx in range(original.shape[1]) if idx not in categorical_columns]
    categorical_columns = sorted(categorical_columns)

    nrms = None
    if numeric_columns:
        numeric_missing = missing[:, numeric_columns]
        if np.any(numeric_missing):
            nrms = calculate_nrmse(
                np.asarray(original[:, numeric_columns], dtype=float),
                np.asarray(estimate[:, numeric_columns], dtype=float),
                np.where(numeric_missing, 0.0, 1.0),
            )

    ae = None
    if categorical_columns:
        categorical_missing = missing[:, categorical_columns]
        if np.any(categorical_missing):
            matches = np.equal(original[:, categorical_columns], estimate[:, categorical_columns])
            ae = float(np.mean(matches[categorical_missing]))

    return {"NRMS": nrms, "AE": ae}


def run_imputation_pipeline(generator, incomplete_data, mask):
    with torch.no_grad():
        z_test = torch.rand(incomplete_data.shape[0], incomplete_data.shape[1])
        x_bar_tensor = torch.tensor(incomplete_data, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        imputed_tensor = generator(x_bar_tensor, mask_tensor, z_test)
        return imputed_tensor.numpy()
