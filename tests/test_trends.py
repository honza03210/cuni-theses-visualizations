"""The baselines and their prediction intervals."""

import numpy as np
import pytest

from trends import fit, interval, pooled, prediction_se, t_quantile

YEARS = np.arange(2006, 2027)
TRAIN = (2013, 2022)


@pytest.mark.parametrize("df, exact", [(1, 12.706), (3, 3.182), (6, 2.447), (8, 2.306), (10, 2.228),
                                       (30, 2.042), (1000, 1.962)])
def test_t_quantile(df, exact):
    assert t_quantile(df) == pytest.approx(exact, abs=0.005)


def test_trend_recovers_a_line():
    rates = np.array([2.0 + 0.5 * (YEARS - 2013), 10 - 0.2 * (YEARS - 2013)])
    b = fit(rates, YEARS, TRAIN)
    assert b.slope == pytest.approx([0.5, -0.2])
    assert b.predict([2025])[:, 0] == pytest.approx([2 + 0.5 * 12, 10 - 0.2 * 12])
    assert b.resid_sd == pytest.approx([0, 0], abs=1e-9)


def test_mean_baseline_is_flat():
    rates = np.array([1.0 + (YEARS % 2)])  # alternating 1, 2
    b = fit(rates, YEARS, TRAIN, kind="mean")
    assert b.slope == pytest.approx([0])
    assert b.predict([2025, 2030])[0] == pytest.approx([1.5, 1.5])


def test_prediction_interval_covers_95_percent():
    rng = np.random.default_rng(1)
    n = 4000
    truth = 20 + 0.3 * (YEARS - 2013)
    rates = truth + rng.normal(0, 2, size=(n, len(YEARS)))
    b = fit(rates, YEARS, TRAIN)
    # Huge sizes make sampling noise negligible, so the scatter decides the width.
    expected, lower, upper = interval(b, [2025], [1e12], 1_000_000, "count")
    inside = (rates[:, YEARS == 2025] >= lower) & (rates[:, YEARS == 2025] <= upper)
    assert inside.mean() == pytest.approx(0.95, abs=0.015)


def test_sampling_floor_for_series_without_scatter():
    # A word never used before has no scatter; its interval must still reflect counting noise.
    b = fit(np.zeros((1, len(YEARS))), YEARS, TRAIN)
    se = prediction_se(b, [2025], [3000], 100, "share")
    assert se[0, 0] > 0


def test_pooled_single_group_matches_direct_rates():
    rng = np.random.default_rng(2)
    sizes = np.full(len(YEARS), 1e6)
    counts = rng.poisson(50, size=(3, len(YEARS))).astype(float)
    actual, expected, se = pooled([(counts, sizes, YEARS)], TRAIN, (2022, 2026), 1_000_000, "count", "mean")
    cmp = (YEARS >= 2022) & (YEARS <= 2026)
    assert actual == pytest.approx(counts[:, cmp].mean(axis=1))
    train = (YEARS >= TRAIN[0]) & (YEARS <= TRAIN[1])
    assert expected == pytest.approx(counts[:, train].mean(axis=1))
    assert (se > 0).all()


def test_pooled_by_group_removes_a_mix_shift():
    # Two groups with constant but different rates; the big-rate group grows. Pooled over both,
    # the overall rate rises, but each group is exactly at its baseline.
    years = YEARS
    small = np.where(years >= 2022, 1e6, 5e6)
    big = np.where(years >= 2022, 5e6, 1e6)
    g1 = (np.array([small * 10 / 1e6]), small, years)   # rate 10
    g2 = (np.array([big * 100 / 1e6]), big, years)      # rate 100
    actual, expected, _ = pooled([g1, g2], TRAIN, (2022, 2026), 1_000_000, "count", "mean")
    assert actual == pytest.approx(expected)
    mixed = (g1[0] + g2[0], small + big, years)
    actual_m, expected_m, _ = pooled([mixed], TRAIN, (2022, 2026), 1_000_000, "count", "mean")
    assert actual_m[0] > 2 * expected_m[0]
