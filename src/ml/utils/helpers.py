import numpy as np
import scipy.io as scio

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
    r_den = np.sqrt(np.sum(Xm**2, axis=0) * np.sum(Ym**2, axis=0))
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


def flip_heading(log, col):
    # Flip WCS for selected recordings: applies for Simulator SEP logs
    mask = log[col].values < 0
    log.loc[mask, col] = np.pi + log.loc[mask, col]
    mask = np.logical_or(mask, log[col].values == 0)
    log.loc[~mask, col] = -(np.pi - log.loc[~mask, col])
    return log
##
# mat utils
def mat_check_keys(mat):
    """
    Checks if entries in dictionary are mat-objects
    and converts them to nested dictionaries
    """
    for key in mat:
        if isinstance(mat[key], scio.matlab.mio5_params.mat_struct):
            mat[key] = mat2dict(mat[key])
    return mat
def mat2dict(matobj):
    """
    A recursive function that constructs nested dictionaries from matobjects
    """
    dictionary = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scio.matlab.mio5_params.mat_struct):
            dictionary[strg] = mat2dict(elem)
        else:
            dictionary[strg] = elem
    return dictionary
def loadmat(filename):
    """
    Replaces spio.loadmat
    """
    data = scio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return mat_check_keys(data)


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
