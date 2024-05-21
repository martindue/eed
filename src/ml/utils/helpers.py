import numpy as np


def vcorrcoef(X, Y):
    """
    NumPy correlation coefficient
    adapted from:
    https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
    """
    Xm = X - np.mean(X)
    Ym = Y - np.mean(Y)
    r_num = np.sum(Xm * Ym)
    r_den = np.sqrt(np.sum(Xm**2) * np.sum(Ym**2))
    return r_num / r_den

def vcorrcoef_rolling(X,Y, window_size):
    
    Xm = X - X.rolling(window_size, center=True).mean().bfill().ffill()
    Ym = Y - Y.rolling(window_size,center=True).mean().bfill().ffill()
    
    r_num = Xm*Ym
    r_den = np.sqrt(np.sum(Xm**2) * np.sum(Ym**2))
    return r_num / r_den

def vcorrcoef_vec(X,Y):
    '''
    NumPy vectorized correlation coefficient
    adapted from:
    https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
    '''
    Xm = X - np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    Ym = Y - np.reshape(np.mean(Y,axis=1),(Y.shape[0],1))
    r_num = np.sum(Xm*Ym,axis=1)
    r_den = np.sqrt(np.sum(Xm**2,axis=1) * np.sum(Ym**2, axis=1))
    return r_num / r_den

def linear_interpol_with_pandas(X):
    """
    Impute missing values in an array with the mean value of each column using pandas.

    Args:
    - X: numpy array with missing values

    Returns:
    - X_imputed: numpy array with missing values imputed using column means
    """
    import pandas as pd

    # Convert to pandas DataFrame
    df = pd.DataFrame(X)

    # Linearly interpolate missing values
    df_imputed = df.interpolate()

    # Convert back to numpy array
    X_imputed = df_imputed.to_numpy()

    return X_imputed
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


def round_up(f, min_val=3):
    """Rounds input value up.
    Parameters:
        f       --  input value
        min_val --  minimum value to retun
    Returns:
        Rounded value
    """
    w = np.int32(np.ceil(f))
    w = min_val if w < min_val else w
    return w
