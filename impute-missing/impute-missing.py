import numpy as np

def impute_missing(X, strategy='mean'):
    # Ensure input is a float-based numpy array
    X = np.array(X, dtype=float, copy=True)
    is_1d = X.ndim == 1
    
    # Standardize to 2D (N, 1) for consistent column-wise logic
    if is_1d:
        X = X.reshape(-1, 1)
        
    # Calculate column-wise stats (ignoring existing NaNs)
    if strategy == 'mean':
        fill_values = np.nanmean(X, axis=0)
    elif strategy == 'median':
        fill_values = np.nanmedian(X, axis=0)
    else:
        raise ValueError("Strategy must be 'mean' or 'median'")

    # Fallback: Replace NaN statistics (from all-NaN columns) with 0.0
    fill_values = np.nan_to_num(fill_values, nan=0.0)

    # Identify row/column indices of NaN values
    rows, cols = np.where(np.isnan(X))
    
    # Broadcast the fill_values into the holes
    X[rows, cols] = fill_values[cols]
    
    # Flatten if the original input was 1D
    return X.flatten() if is_1d else X