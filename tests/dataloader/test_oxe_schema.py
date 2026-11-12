"""Canonical DROID and shared Euler pose conversion regressions."""

import numpy as np
import pytest

from openwam.dataloader.utils.oxe_schema import (
    droid_euler7_to_arm10,
    droid_pose6_closedness_to_arm10,
    euler7_action_to_arm10,
)


class TestEuler7ActionToArm10:
    def test_shape(self):
        action = np.zeros((4, 7), dtype=np.float32)
        out = euler7_action_to_arm10(action)
        assert out.shape == (4, 10)
        assert out.dtype == np.float32

    def test_pos_grip_passthrough(self):
        action = np.array([[0.1, 0.2, 0.3, 0, 0, 0, 0.9]], dtype=np.float32)
        out = euler7_action_to_arm10(action)
        np.testing.assert_allclose(out[0, :3], [0.1, 0.2, 0.3], atol=1e-6)
        assert out[0, 9] == np.float32(0.9)


class TestDroidEuler7ToArm10:
    def test_closedness_is_inverted_to_canonical_openness(self):
        value = np.zeros((3, 7), dtype=np.float32)
        value[:, 6] = [0.0, 0.25, 1.0]

        out = droid_euler7_to_arm10(value)

        np.testing.assert_allclose(out[:, 9], [1.0, 0.75, 0.0], atol=1e-7)

    def test_pose_conversion_matches_generic_converter(self):
        value = np.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6, 0.25]], dtype=np.float32)

        droid = droid_euler7_to_arm10(value)
        generic = euler7_action_to_arm10(value)

        np.testing.assert_allclose(droid[:, :9], generic[:, :9], atol=1e-7)


class TestDroidPose6ClosednessToArm10:
    def test_assembles_arm_side_pose_and_inverts_only_closedness(self):
        pose = np.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=np.float32)
        closedness = np.array([[0.25]], dtype=np.float32)

        out = droid_pose6_closedness_to_arm10(pose, closedness)
        reference = droid_euler7_to_arm10(np.concatenate([pose, closedness], axis=-1))

        np.testing.assert_allclose(out, reference, atol=1e-7)
        np.testing.assert_allclose(out[:, :3], pose[:, :3], atol=1e-7)
        np.testing.assert_allclose(out[:, 9], [0.75], atol=1e-7)


@pytest.mark.parametrize("leading", [(), (2,), (2, 3)])
def test_euler_pose_conversion_preserves_arbitrary_leading_dimensions(leading):
    value = np.arange(int(np.prod(leading) or 1) * 7, dtype=np.float32).reshape(leading + (7,)) / 10
    expected = euler7_action_to_arm10(value.reshape(-1, 7)).reshape(leading + (10,))
    np.testing.assert_array_equal(euler7_action_to_arm10(value), expected)


@pytest.mark.parametrize("value", [np.zeros(8), np.zeros((2, 6)), np.array(0.0)])
def test_euler_pose_conversion_rejects_wrong_width_instead_of_dropping_fields(value):
    with pytest.raises(ValueError, match="width 7"):
        euler7_action_to_arm10(value)


@pytest.mark.parametrize("index", [0, 3, 6])
def test_droid_conversion_rejects_nonfinite_position_rotation_or_gripper(index):
    value = np.zeros((1, 7), dtype=np.float32)
    value[0, index] = np.nan
    with pytest.raises(ValueError, match="finite"):
        droid_euler7_to_arm10(value)


def test_droid_scalar_pose_is_rejected_with_an_actionable_error():
    with pytest.raises(ValueError, match="width 6"):
        droid_pose6_closedness_to_arm10(np.array(0.0), np.zeros(1))
