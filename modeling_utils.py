"""
Document
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

import scipy.stats as stats


def q_trunc(series: pd.Series, low=1e-4, high=1 - 1e-4):
    """ """
    return series.clip(series.quantile(low), series.quantile(high))


def t_corr(series, shift=-1):
    """ """
    return series.corr(series.shift(shift))


def fit_linear(series):
    """ """
    x = np.arange(len(series))
    y = series.values

    a, b = np.polyfit(x, y, deg=1)

    y_fit = a * x + b

    return a, b, y_fit


def show_simple_autocorr(series, max_lag=10, step=1):
    """ """
    print("Simple autocorrelation (Pearson) by lag:")
    for k in range(1, max_lag + 1):
        print(f"lag {k:2d}: {series.autocorr(lag=k * step): .4f}")


def plot_auto(series, lags=10):
    """ """
    max_lag = 10

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title("ACF")

    plot_pacf(series, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title("PACF")

    plt.tight_layout()
    plt.show()


def plot_auto_shift(series, lags=40, shift=1):
    """ """
    acf_vals = acf(series, nlags=lags)[shift:]  # remove lag 0
    pacf_vals = pacf(series, nlags=lags)[shift:]  # remove lag 0

    lags = range(shift, lags + 1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    ax[0].stem(lags, acf_vals)
    ax[0].set_title("ACF (lag 1 to nlags)")

    ax[1].stem(lags, pacf_vals)
    ax[1].set_title("PACF (lag 1 to nlags)")

    plt.tight_layout()
    plt.show()


def acf_rolling(series: pd.Series, lag=1, window=20000, step=None):
    """ """
    if step is None:
        step = window

    autos = np.zeros(int((len(series) - window) / step) + 1)

    for i in range(0, len(series) - window, step):
        autos[int(i / step)] = series[i : i + window].autocorr(lag)

    return pd.Series(autos)


def moment_rolling(series: pd.Series, order=1, window=20000, step=None):
    """ """
    if step is None:
        step = window

    m = np.zeros(int((len(series) - window) / step) + 1)
    for i in range(0, len(series) - window, step):
        m[i // step] = stats.moment(series[i : i + window], order=order)

    return pd.Series(m)

def predict_r2(model, target, features, lag=10, cv=10, score='r2'):
  '''
  model = LinearRegression()
  '''

  y = target.shift(-lag).dropna()
  X = features.iloc[:-lag, :]

  scores = cross_val_score(model, X, y, cv=cv, scoring=score)
  if score == 'r2':
    scores = 1 - (1 - scores) * (len(y) - 1) / (len(y) - X.shape[1])

  return scores