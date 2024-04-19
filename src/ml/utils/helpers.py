
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