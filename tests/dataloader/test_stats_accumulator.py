"""Numerical and sampling contracts shared by benchmark statistics tools."""

import numpy as np
import pytest

from openwam.dataloader.utils.stats_computation.accumulator import Accumulator


@pytest.mark.parametrize("method", ["update", "update_batch"])
def test_accumulator_matches_full_population_statistics(method):
    rows = np.random.RandomState(17).normal(size=(71, 4)).astype(np.float32)
    rows[:, -1] = 7
    accumulator = Accumulator(dim=4, reservoir_cap=100)
    for batch in np.array_split(rows, 5):
        getattr(accumulator, method)(batch)
    stats = accumulator.finalize()
    assert accumulator.count == len(rows)
    for key, expected in {
        "mean": rows.astype(np.float64).mean(axis=0),
        "std": np.where(rows.astype(np.float64).std(axis=0) < 1e-8, 1, rows.astype(np.float64).std(axis=0)),
        "min": rows.min(axis=0),
        "max": rows.max(axis=0),
        "q01": np.quantile(rows, 0.01, axis=0),
        "q99": np.quantile(rows, 0.99, axis=0),
    }.items():
        np.testing.assert_allclose(stats[key], expected, rtol=1e-6, atol=1e-7)


def test_reservoir_matches_sequential_algorithm_r_across_batches():
    rows = np.arange(300, dtype=np.float32).reshape(100, 3)
    cap = 7
    expected = rows[:cap].copy()
    rng = np.random.RandomState(23)
    for i in range(cap, len(rows)):
        position = rng.randint(0, i + 1)
        if position < cap:
            expected[position] = rows[i]
    accumulator = Accumulator(dim=3, reservoir_cap=cap, seed=23)
    for batch in np.array_split(rows, 6):
        accumulator.update_batch(batch)
    np.testing.assert_array_equal(accumulator._res, expected)
    assert accumulator._res_seen == len(rows)
    assert accumulator._res_n == cap


@pytest.mark.parametrize("method", ["update", "update_batch"])
@pytest.mark.parametrize(
    "rows", [np.ones((3, 1)), np.ones(3), np.array([[np.nan, 0]]), np.array([[np.inf, 0]]), np.array([[1e100, 0]])]
)
def test_invalid_batches_do_not_mutate_statistics(method, rows):
    accumulator = Accumulator(dim=2, reservoir_cap=5)
    with pytest.raises(ValueError):
        getattr(accumulator, method)(rows)
    assert accumulator.count == 0
    assert accumulator._res_seen == 0
    np.testing.assert_array_equal(accumulator.mean, [0, 0])


@pytest.mark.parametrize(
    "kwargs", [{"dim": 0}, {"dim": -1}, {"reservoir_cap": 0}, {"reservoir_cap": -1}, {"reservoir_cap": 1.5}]
)
def test_accumulator_rejects_invalid_dimensions_or_capacity(kwargs):
    with pytest.raises(ValueError, match="positive integers"):
        Accumulator(**kwargs)


def test_empty_statistics_cannot_produce_infinite_bounds():
    accumulator = Accumulator(dim=2, reservoir_cap=5)
    accumulator.update_batch(np.empty((0, 2)))
    with pytest.raises(ValueError, match="without any rows"):
        accumulator.finalize()
