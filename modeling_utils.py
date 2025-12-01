"""
Document
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

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


def predict_r2(model, target, features, lag=10, cv=10, score="r2"):
    """
    model = LinearRegression()
    """

    y = target.shift(-lag).dropna()
    X = features.iloc[:-lag, :]

    scores = cross_val_score(model, X, y, cv=cv, scoring=score)
    if score == "r2":
        scores = 1 - (1 - scores) * (len(y) - 1) / (len(y) - X.shape[1])

    return scores


def feature_shift(series, shifts):
    return (
        pd.concat({f"t-{shift}": series.shift(shift) for shift in shifts}, axis=1)
        .fillna(0)
    )


def feature_concat(features_list):
    return pd.concat(features_list, axis=1)


# ---------------------------------------------------------------------------
# Alpha modeling helpers
# ---------------------------------------------------------------------------


def prepare_supervised(series: pd.Series, lags: list[int], horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Build lagged features and aligned forward target for a time series."""

    X = feature_shift(series, lags)
    y = series.shift(-horizon)
    valid = y.notna()
    return X.loc[valid], y.loc[valid]


def train_linear_alpha(series: pd.Series, lags: list[int], horizon: int, model=None):
    """Fit a linear regression alpha on lagged features."""

    model = model or LinearRegression()
    X, y = prepare_supervised(series, lags, horizon)
    model.fit(X, y)
    preds = pd.Series(model.predict(X), index=X.index, name="pred")
    mse = mean_squared_error(y, preds)
    return model, preds, {"mse": mse}


def train_logit_alpha(series: pd.Series, lags: list[int], horizon: int, model=None):
    """Fit a logistic classifier on the sign of forward returns."""

    model = model or LogisticRegression(max_iter=500)
    X, y = prepare_supervised(series, lags, horizon)
    y_sign = (y > 0).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y_sign, test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, prob)
    full_prob = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="prob_up")
    return model, full_prob, {"val_auc": auc}


def timestamp_ms_to_ns(ts_ms: pd.Series | np.ndarray) -> pd.Series:
    """Convert millisecond timestamps to nanoseconds (hftbacktest default)."""

    return pd.to_numeric(ts_ms, errors="coerce").astype("int64") * 1_000_000


def load_okx_features(
    root: Path | str,
    max_files: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load OKX parquet feature dataset quickly, mirroring the hft_eda cleaning.

    The function mirrors the notebook workflow: concatenate parquet parts,
    replace `vwap == -1` with NaN, forward-fill gaps, and drop remaining NaNs.

    Args:
        root: Directory that holds the parquet dataset (e.g., `features-*.parquet`).
        max_files: Optionally limit the number of files to read (fallback path).
        columns: Optional column subset to speed up IO.

    Returns:
        Cleaned DataFrame indexed by timestamp.
    """

    root = Path(root)
    # Fast path: let pandas/pyarrow treat the directory as a parquet dataset.
    books: pd.DataFrame
    if max_files is None:
        try:
            books = pd.read_parquet(root, columns=columns)
        except Exception:
            files = sorted(root.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found under {root}")
            books = pd.read_parquet(files, columns=columns)
    else:
        files = sorted(root.rglob("*.parquet"))[:max_files]
        if not files:
            raise FileNotFoundError(f"No parquet files found under {root}")
        books = pd.read_parquet(files, columns=columns)

    books.loc[books.vwap == -1, "vwap"] = np.nan
    books = books.ffill().dropna(how="any")
    books = books.set_index("timestamp").sort_index()
    return books


def compute_log_returns(price: pd.Series, horizons=(1, 10)) -> pd.DataFrame:
    """Compute log returns for multiple horizons and fill leading NaNs with zeros."""

    rets = {f"rt{h}": np.log(price).diff(h).fillna(0) for h in horizons}
    return pd.DataFrame(rets, index=price.index)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score (mean/std) with a small epsilon safeguard."""

    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std.replace(0, np.nan) + 1e-9)


def lagged_correlations(feature: pd.Series, target: pd.Series, lags: list[int]):
    """Compute Pearson correlations of a feature with forward targets over many lags."""

    corrs = {}
    for lag in lags:
        corrs[f"t+{lag}"] = feature.corr(target.shift(-lag))
    return pd.Series(corrs)


def corr_heatmap(df: pd.DataFrame, title: str | None = None, figsize=(8, 6)):
    """Plot a correlation heatmap with a consistent style."""

    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
