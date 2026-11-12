"""Tests for the extracted normalization helper in ``utils.normalization``.

Mirrors ``test_robocoin_normalize.py`` but exercises the pure-function
form directly, so the helper can be safely reused by the upcoming OXE
readers without going through a RoboCOIN stub.
"""

from __future__ import annotations

import numpy as np
import pytest

from openwam.dataloader.utils.normalization import apply_normalization


def _stats():
    return {
        "mean": np.array([5.0, 0.0], dtype=np.float32),
        "std": np.array([2.0, 0.5], dtype=np.float32),
        "min": np.array([0.0, -1.0], dtype=np.float32),
        "max": np.array([10.0, 1.0], dtype=np.float32),
    }


class TestApplyNormalization:
    def test_no_stats_passthrough(self):
        arr = np.arange(20, dtype=np.float32).reshape(1, 20)
        out = apply_normalization(arr, None, "min-max")
        assert (out == arr).all()

    def test_none_mode_passthrough(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        for mode in (None, "none", "null"):
            out = apply_normalization(arr, _stats(), mode)
            np.testing.assert_allclose(out, arr, atol=1e-6)

    def test_min_max_min_maps_to_neg_one(self):
        arr = np.array([[0.0, -1.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_min_max_max_maps_to_pos_one(self):
        arr = np.array([[10.0, 1.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_min_max_mid_maps_to_zero(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_z_score_mean_maps_to_zero(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "z-score")
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_z_score_one_std_maps_to_one(self):
        arr = np.array([[7.0, 0.5]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "z-score")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_unknown_mode_passes_through(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "weird-mode")
        np.testing.assert_allclose(out, arr, atol=1e-6)

    def test_broadcast_across_leading_dims(self):
        arr = np.array([[0.0, -1.0], [10.0, 1.0], [5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(
            out,
            [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
            atol=1e-6,
        )


def _stats_with_quantile():
    return {
        "mean": np.array([0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 1.0], dtype=np.float32),
        "min": np.array([-100.0, -50.0], dtype=np.float32),  # outliers
        "max": np.array([100.0, 50.0], dtype=np.float32),
        "q01": np.array([-1.0, -2.0], dtype=np.float32),
        "q99": np.array([1.0, 2.0], dtype=np.float32),
    }


class TestQuantileNormalization:
    def test_q01_maps_to_neg_one(self):
        arr = np.array([[-1.0, -2.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_q99_maps_to_pos_one(self):
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_midpoint_maps_to_zero(self):
        arr = np.array([[0.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_outlier_clipped_to_pos_one(self):
        # Value 100 is way beyond q99=1 → clipped to 1.0
        arr = np.array([[100.0, 50.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_outlier_clipped_to_neg_one(self):
        arr = np.array([[-100.0, -50.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_quantile_missing_q01_raises(self):
        import pytest

        stats_no_q = {
            "mean": np.zeros(2, dtype=np.float32),
            "std": np.ones(2, dtype=np.float32),
            "min": -np.ones(2, dtype=np.float32),
            "max": np.ones(2, dtype=np.float32),
        }
        with pytest.raises(KeyError, match="quantile normalization requires"):
            apply_normalization(np.zeros((1, 2), dtype=np.float32), stats_no_q, "quantile")

    def test_quantile_no_op_for_null_stats(self):
        arr = np.array([[100.0, 100.0]], dtype=np.float32)
        out = apply_normalization(arr, None, "quantile")
        np.testing.assert_allclose(out, arr, atol=1e-7)

    def test_quantile_within_bounds_no_clip(self):
        arr = np.array([[0.5, 1.0], [-0.5, -1.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        # No clipping for values within [q01, q99]
        expected = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)


class TestDatasetStatsBoundary:
    def test_root_mode_does_not_borrow_statistics_outside_its_family(self, tmp_path):
        from openwam.dataloader.utils.normalization import find_dataset_stats

        family = tmp_path / "family"
        bucket = family / "category" / "robot" / "task"
        (bucket / "meta").mkdir(parents=True)
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "stats_robot.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="dataset family"):
            find_dataset_stats(bucket, "stats_robot.json", family_root=family)

    def test_standalone_bucket_stops_at_nearest_family_metadata(self, tmp_path):
        from openwam.dataloader.utils.normalization import find_dataset_stats

        family = tmp_path / "family"
        bucket = family / "nested" / "task"
        (bucket / "meta").mkdir(parents=True)
        (family / "meta").mkdir()
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "stats_robot.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="dataset family"):
            find_dataset_stats(bucket, "stats_robot.json")

    def test_symlinked_bucket_uses_configured_family_statistics(self, tmp_path):
        from openwam.dataloader.utils.normalization import find_dataset_stats

        physical = tmp_path / "physical" / "bucket"
        (physical / "meta").mkdir(parents=True)
        (physical.parent / "meta").mkdir()
        (physical.parent / "meta" / "stats_robot.json").write_text("physical")
        selected = tmp_path / "selected"
        (selected / "meta").mkdir(parents=True)
        expected = selected / "meta" / "stats_robot.json"
        expected.write_text("selected")
        alias = selected / "bucket"
        try:
            alias.symlink_to(physical, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks are unavailable")
        assert find_dataset_stats(alias, "stats_robot.json", family_root=selected) == expected

    def test_local_statistics_take_precedence_within_the_family(self, tmp_path):
        from openwam.dataloader.utils.normalization import find_dataset_stats

        bucket = tmp_path / "bucket"
        (bucket / "meta").mkdir(parents=True)
        expected = bucket / "meta" / "stats_robot.json"
        expected.write_text("local")
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "stats_robot.json").write_text("pooled")
        assert find_dataset_stats(bucket, "stats_robot.json", family_root=tmp_path) == expected

    def test_bucket_must_belong_to_explicit_family(self, tmp_path):
        from openwam.dataloader.utils.normalization import find_dataset_stats

        with pytest.raises(ValueError, match="outside its statistics family"):
            find_dataset_stats(tmp_path / "elsewhere", "stats.json", family_root=tmp_path / "family")

    def test_filename_cannot_escape_metadata_directory(self, tmp_path):
        from openwam.dataloader.utils.normalization import find_dataset_stats

        with pytest.raises(ValueError, match="single file name"):
            find_dataset_stats(tmp_path, "../stats.json")


class TestMaterializedStatsValidation:
    @pytest.mark.parametrize(
        "field,index,value,match",
        [
            ("std", 0, -1.0, "standard deviation"),
            ("std", 3, -1.0, "standard deviation"),
            ("min", 0, 2.0, "min <= max"),
            ("q01", 0, 2.0, "q01 <= q99"),
            ("q01", 3, 2.0, "q01 <= q99"),
            ("mean", 0, float("nan"), "non-finite"),
        ],
    )
    def test_corrupt_stats_are_rejected_before_rotation_identity_pinning(self, field, index, value, match):
        from openwam.dataloader.utils.normalization import materialize_eef_stats

        raw = {
            "mean": [0.0] * 10,
            "std": [1.0] * 10,
            "min": [-1.0] * 10,
            "max": [1.0] * 10,
            "q01": [-1.0] * 10,
            "q99": [1.0] * 10,
        }
        raw[field][index] = value
        with pytest.raises(ValueError, match=match):
            materialize_eef_stats(raw, "quantile", dim=10, strict_minmax=False, force_rot6d_identity=True)

    def test_optional_defaults_still_support_benchmark_min_max_stats(self):
        from openwam.dataloader.utils.normalization import materialize_eef_stats

        stats = materialize_eef_stats({"min": [-2.0] * 10, "max": [2.0] * 10}, "min-max", dim=10, strict_minmax=False)
        np.testing.assert_array_equal(stats["mean"], np.zeros(10))
        np.testing.assert_array_equal(stats["std"], np.ones(10))
