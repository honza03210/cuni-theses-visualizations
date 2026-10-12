"""Baselines for yearly rates: a linear trend or a mean, with prediction intervals.

Rates are matrices of series x years. A baseline is fitted on the baseline years and predicts
later years; its standard error combines the scatter of the baseline years around the fit with
the uncertainty of the fit itself, so that "unexpected" can be judged against normal year-to-year
noise. The scatter never counts as smaller than the sampling noise of a count of that size, which
keeps series that happened to be flat in the baseline from looking infinitely surprising.
"""

from dataclasses import dataclass

import numpy as np


# Exact 97.5 % quantiles for the few degrees of freedom where the expansion below is poor
# (a training range of 3-7 years leaves 1-5).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def t_quantile(df, p: float = 0.975) -> np.ndarray:
    """Student's t quantile: exact for small df at 97.5 %, else the Cornish-Fisher expansion
    to the fourth term (within 0.005 for df >= 6)."""
    z = 1.959963984540054 if p == 0.975 else _normal_quantile(p)
    df = np.maximum(np.asarray(df, dtype="float64"), 1)
    t = (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
         + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
         + (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / (92160 * df**4))
    if p == 0.975:
        t = np.where(df <= 5, np.vectorize(lambda d: _T975.get(int(d), 0.0))(df), t)
    return t


def _normal_quantile(p: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


@dataclass
class Baseline:
    """A fitted baseline for every series."""
    intercept: np.ndarray
    slope: np.ndarray       # zero for a mean baseline
    resid_sd: np.ndarray    # scatter of the baseline years around the fit
    n: int                  # number of baseline years
    xbar: float
    sxx: float              # sum of squared deviations of the baseline years; inf for a mean baseline

    def predict(self, years) -> np.ndarray:
        """Expected rates, series x years, never below zero."""
        years = np.atleast_1d(np.asarray(years, dtype="float64"))
        return np.clip(self.intercept[:, None] + self.slope[:, None] * years[None, :], 0, None)

    def fit_var(self, years) -> np.ndarray:
        """Variance of the fitted line itself at `years`, relative to the residual variance."""
        years = np.atleast_1d(np.asarray(years, dtype="float64"))
        return 1 / self.n + (years - self.xbar) ** 2 / self.sxx

    @property
    def df(self) -> int:
        return self.n - (2 if np.isfinite(self.sxx) else 1)


def fit(rates: np.ndarray, years, train: tuple[int, int], kind: str = "trend") -> Baseline:
    """Fit a baseline to columns of `rates` (series x years) that fall in the training years."""
    years = np.asarray(years)
    mask = (years >= train[0]) & (years <= train[1])
    x = years[mask].astype("float64")
    y = np.asarray(rates, dtype="float64")[:, mask]
    xbar, n = x.mean(), len(x)
    if kind == "trend":
        sxx = ((x - xbar) ** 2).sum()
        slope = (y - y.mean(axis=1, keepdims=True)) @ (x - xbar) / sxx
        dof = n - 2
    else:
        sxx, slope, dof = np.inf, np.zeros(len(y)), n - 1
    intercept = y.mean(axis=1) - slope * xbar
    resid = y - (intercept[:, None] + slope[:, None] * x[None, :])
    resid_sd = np.sqrt((resid**2).sum(axis=1) / max(dof, 1))
    return Baseline(intercept, slope, resid_sd, n, xbar, sxx)


def sampling_sd(rate: np.ndarray, size: np.ndarray, scale: float, kind: str) -> np.ndarray:
    """Standard deviation of a rate from sampling alone.

    kind "share": `rate` is a share of `size` units (theses) in units of 1/scale (100 = percent),
    binomial. kind "count": `rate` is a count per `size` units (words) times `scale`, Poisson.
    The rate is taken as at least one unit (one thesis, one occurrence): a series expected to be
    near zero could still easily show up once by chance.
    """
    size = np.maximum(np.asarray(size, dtype="float64"), 1)
    rate = np.maximum(np.asarray(rate, dtype="float64"), scale / size)
    if kind == "share":
        p = np.clip(rate / scale, 0, 1)
        return scale * np.sqrt(p * (1 - p) / size)
    return np.sqrt(rate * scale / size)


def prediction_se(baseline: Baseline, years, size, scale: float, kind: str) -> np.ndarray:
    """Standard error of a new year's rate around its prediction, series x years."""
    expected = baseline.predict(years)
    sd = np.maximum(baseline.resid_sd[:, None], sampling_sd(expected, np.asarray(size)[None, :], scale, kind))
    return sd * np.sqrt(1 + baseline.fit_var(years))[None, :]


def interval(baseline: Baseline, years, size, scale: float, kind: str, level: float = 0.95):
    """Expected rates with the lower and upper prediction bounds, series x years."""
    expected = baseline.predict(years)
    half = t_quantile(baseline.df, 0.5 + level / 2) * prediction_se(baseline, years, size, scale, kind)
    return expected, np.clip(expected - half, 0, None), expected + half


def pooled(groups, train: tuple[int, int], period: tuple[int, int], scale: float, kind: str, baseline: str):
    """Actual and expected rates over the compared years, pooled over groups.

    `groups` is a list of (counts, sizes, years): counts is series x years (mentions or theses),
    sizes is the matching denominator per year (words or theses). Every group gets its own
    baseline, so pooling over faculties compares each faculty with its own past and lets the
    faculties' changing sizes weigh in, instead of mixing a changing population.
    Returns actual, expected and the standard error of expected - actual, per series.
    """
    actual_count = expected_count = var = 0
    total = 0
    for counts, sizes, years in groups:
        years = np.asarray(years)
        sizes = np.asarray(sizes, dtype="float64")
        rates = counts / np.maximum(sizes, 1) * scale
        b = fit(rates, years, train, baseline)
        cmp = (years >= period[0]) & (years <= period[1])
        w = sizes[cmp]
        exp_rates = b.predict(years[cmp])
        expected_count = expected_count + exp_rates @ w / scale
        actual_count = actual_count + counts[:, cmp].sum(axis=1)
        # Variance of the pooled expected rate: scatter of the compared years themselves around the
        # baseline, plus the uncertainty of the baseline at the compared years' (weighted) centre.
        mean_rate = exp_rates @ w / max(w.sum(), 1)
        sd = np.maximum(b.resid_sd, sampling_sd(mean_rate, w.sum() / max(cmp.sum(), 1), scale, kind))
        centre = (years[cmp] * w).sum() / max(w.sum(), 1)
        rel = (w**2).sum() / max(w.sum(), 1) ** 2 + b.fit_var(centre)[0]
        var = var + sd**2 * rel * w.sum() ** 2
        total += w.sum()
    total = max(total, 1)
    return actual_count / total * scale, expected_count / total * scale, np.sqrt(var) / total
