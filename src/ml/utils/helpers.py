import numpy as np


def vcorrcoef(X, Y):
    """
    NumPy vectorized correlation coefficient
    adapted from:
    https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
    """
    Xm = X - np.mean(X)
    Ym = Y - np.mean(Y)
    r_num = np.sum(Xm * Ym)
    r_den = np.sqrt(np.sum(Xm**2) * np.sum(Ym**2))
    return r_num / r_den

def impute_with_column_means(X):
    """
    Impute missing values in an array with the mean value of each column.

    Args:
    - X: numpy array with missing values

    Returns:
    - X_imputed: numpy array with missing values imputed using column means
    """
    # Calculate the mean of each column, ignoring NaN values
    column_means = np.nanmean(X, axis=0)

    # Find indices of NaN values
    nan_indices = np.isnan(X)

    # Replace NaN values with column means
    X_imputed = np.where(nan_indices, np.expand_dims(column_means, axis=0), X)

    return X_imputed