# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Verify automatic motion variation without a simulator or GPU."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


auto = load("variation_auto_test", "record_toy2box_auto.py")
variation = load("motion_variation_test", "toy2box_motion_variation.py")


def state():
    return {
        "eef_pos": np.array([0.5, -0.2, 0.8]), "eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "toy_truck_positions": np.array([0.6, -0.2, 0.38]),
        "toy_truck_orientations": np.array([1.0, 0.0, 0.0, 0.0]),
        "box_positions": np.array([0.85, 0.05, 0.5]), "box_orientations": np.array([1.0, 0.0, 0.0, 0.0]),
        "box_top_z": 0.62, "box_bottom_z": 0.4, "truck_bottom_z": 0.35, "opening_margin": -0.1,
    }


def sample_motion(seed):
    motion = variation.MotionVariation(seed)
    cfg = auto.ControllerConfig()
    values = []
    for distance in np.linspace(0.4, 0.005, 150):
        if motion.hold_requested("approach", 0.05):
            values.append(np.zeros(6))
        else:
            values.append(motion.modify("approach", np.array([distance, 0, 0]), np.zeros(3), cfg, 0.05))
        motion.observe_action(values[-1], 0.05)
    return np.array(values), motion


def test_seed_reproducibility_and_metadata():
    first, motion = sample_motion(42)
    second, repeated = sample_motion(42)
    different, _ = sample_motion(43)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert motion.export() == repeated.export()
    assert json.loads(json.dumps(motion.export()))["seed"] == 42


def test_speed_changes_smoothly_and_stays_below_original_limits():
    cfg = auto.ControllerConfig()
    motion = variation.MotionVariation(9, variation.MotionVariationConfig(
        position_offset=0, rotation_offset=0, pause_probability=0,
    ))
    speeds = []
    for _ in range(100):
        action = motion.modify("approach", np.array([0.3, 0, 0]), np.zeros(3), cfg, 0.05)
        assert np.linalg.norm(action[:3]) <= cfg.max_translation + 1e-8
        assert np.linalg.norm(action[3:]) <= cfg.max_rotation + 1e-8
        speeds.append(motion.speed)
    assert min(speeds) >= motion.cfg.min_speed
    assert max(speeds) <= motion.cfg.max_speed
    assert max(speeds) - min(speeds) > 0.2
    assert max(np.abs(np.diff(speeds))) < 0.15


def test_path_offsets_are_bounded_and_disappear_near_grasp():
    cfg, motion = auto.ControllerConfig(), variation.MotionVariation(3)
    for distance in np.linspace(0.4, 0.01, 80):
        motion.modify("approach", np.array([distance, 0, 0]), np.zeros(3), cfg, 0.05)
        assert np.linalg.norm(motion.last_offset) <= motion.cfg.position_offset + 1e-8
        assert np.linalg.norm(motion.last_rotation_offset) <= motion.cfg.rotation_offset + 1e-8
        assert motion.last_offset[2] == 0.0
        if distance <= motion.cfg.precision_radius:
            np.testing.assert_array_equal(motion.last_offset, np.zeros(3))
            np.testing.assert_array_equal(motion.last_rotation_offset, np.zeros(3))


@pytest.mark.parametrize("phase", ["descend", "lift", "lower", "retreat"])
def test_vertical_phases_have_no_lateral_or_rotation_perturbation(phase):
    cfg, motion = auto.ControllerConfig(), variation.MotionVariation(4)
    motion.modify(phase, np.array([0, 0, 0.3]), np.zeros(3), cfg, 0.05)
    action = motion.modify(phase, np.array([0, 0, 0.15]), np.zeros(3), cfg, 0.05)
    np.testing.assert_array_equal(action[:2], np.zeros(2))
    np.testing.assert_array_equal(action[3:], np.zeros(3))


def test_pause_preserves_grip_and_freezes_only_progress_watchdog():
    cfg = auto.ControllerConfig()
    motion = variation.MotionVariation(0, variation.MotionVariationConfig(
        pause_probability=1.0, target_idle_fraction=0,
    ))
    motion.modify("lift", np.array([0, 0, 0.3]), np.zeros(3), cfg, 0.05)
    motion.modify("lift", np.array([0, 0, 0.06]), np.zeros(3), cfg, 0.05)
    assert motion.pause_remaining > 0
    ctrl = auto.Toy2BoxController([], cfg, motion=motion)
    ctrl.phase, ctrl.phase_time, ctrl.stall_time, ctrl.last_gripper_command = "lift", 1.0, 4.0, -1.0
    action = ctrl.action(state(), 0.05)
    np.testing.assert_array_equal(action, [0, 0, 0, 0, 0, 0, -1])
    assert ctrl.phase_time == pytest.approx(1.0)
    assert ctrl.stall_time == 4.0
    assert ctrl.elapsed == 0.05
    assert motion.total_pause_seconds == 0.05
    for _ in range(20):
        motion.hold_requested("lift", 0.05)
    assert not motion.hold_requested("lift", 0.05)


def test_zero_strength_matches_standard_commands_exactly():
    cfg = auto.ControllerConfig()
    motion = variation.MotionVariation(8, variation.MotionVariationConfig(strength=0))
    pos, rot = np.array([0.1, -0.05, 0.08]), np.array([0.02, 0.1, -0.2])
    expected = np.r_[auto.limited(cfg.gain * pos, cfg.max_translation), auto.limited(cfg.gain * rot, cfg.max_rotation)]
    np.testing.assert_array_equal(motion.modify("approach", pos, rot, cfg, 0.05), expected)
    assert not motion.hold_requested("approach", 0.05)


@pytest.mark.parametrize("seed", [0, 7, 42, 101])
@pytest.mark.parametrize("frames", [320, 400, 600])
def test_idle_budget_is_distributed_and_counts_actual_zero_frames(seed, frames):
    """A long motion with two naturally idle frames and a final precision segment."""
    motion, cfg = variation.MotionVariation(seed), auto.ControllerConfig()
    actions, pause_indices = [], []
    for index in range(frames):
        phase = "release" if index >= frames - 15 else "approach"
        distance = 0.02 if index >= frames - 30 else 0.15
        motion.modify(phase, np.array([distance, 0, 0]), np.zeros(3), cfg, 0.05)
        while motion.hold_requested(phase, 0.05):
            hold = np.array([0, 0, 0, 0, 0, 0, -1.0])
            pause_indices.append(index)
            actions.append(hold)
            motion.observe_action(hold, 0.05)
        command = np.array([0.01, 0, 0, 0, 0, 0, -1.0])
        if index in (0, frames - 15):
            command[:6] = 0
        actions.append(command)
        motion.observe_action(command, 0.05)
    actual = (np.asarray(actions)[:, :6] == 0).all(1).mean()
    assert 0.18 <= actual <= 0.23
    assert motion.export()["actual_idle_fraction"] == actual
    assert motion.export()["applied_steps"] == len(actions)
    assert len(set(pause_indices)) >= 3
    assert max(pause_indices) < frames - 30  # No extra holds in the final precision segment.


def test_budget_counts_natural_idle_and_ignores_gripper_value():
    motion, cfg = variation.MotionVariation(42), auto.ControllerConfig()
    motion.modify("approach", np.array([0.15, 0, 0]), np.zeros(3), cfg, 0.05)
    for _ in range(100):
        motion.observe_action(np.array([0, 0, 0, 0, 0, 0, -1]), 0.05)
    for _ in range(100):
        motion.observe_action(np.array([1e-12, 0, 0, 0, 0, 0, 1]), 0.05)
    assert motion.export()["actual_idle_fraction"] == 0.5
    assert not motion.hold_requested("approach", 0.05)


def test_budget_pause_preserves_grip_and_watchdogs_and_ends_on_phase_change():
    motion, cfg = variation.MotionVariation(42), auto.ControllerConfig()
    motion.modify("lift", np.array([0, 0, 0.3]), np.zeros(3), cfg, 0.05)
    for _ in range(100):
        motion.observe_action(np.array([0, 0, 0.01, 0, 0, 0, -1]), 0.05)
    ctrl = auto.Toy2BoxController([], cfg, motion=motion)
    ctrl.phase, ctrl.phase_time, ctrl.stall_time, ctrl.last_gripper_command = "lift", 1.0, 4.0, -1.0
    action = ctrl.action(state(), 0.05)
    np.testing.assert_array_equal(action, [0, 0, 0, 0, 0, 0, -1])
    assert ctrl.phase_time == pytest.approx(1.0)
    assert ctrl.stall_time == 4.0
    assert motion.events[-1]["reason"] == "idle_budget"
    assert not motion.hold_requested("release", 0.05)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 100])
def test_varied_feedback_still_completes_pick_clear_lift_place_release(seed):
    cfg = auto.ControllerConfig()
    motion = variation.MotionVariation(seed)
    example = auto.GraspExample("demo_0", 0, np.array([0, 0, -0.02]), Rotation.identity())
    ctrl, obs = auto.Toy2BoxController([example], cfg, motion=motion), state()
    attached, offset = False, None
    for _ in range(1800):
        old_ee = obs["eef_pos"].copy()
        action = ctrl.action(obs, 0.05)
        motion.observe_action(action, 0.05)
        assert ctrl.failure is None, ctrl.failure
        assert np.linalg.norm(action[:3]) <= cfg.max_translation + 1e-7
        assert np.linalg.norm(action[3:6]) <= cfg.max_rotation + 1e-7
        if action[6] < 0 and not attached:
            attached, offset = True, old_ee - obs["toy_truck_positions"]
        if action[6] > 0:
            attached = False
        obs["eef_pos"] += action[:3]
        quat = (Rotation.from_rotvec(action[3:6]) * auto.rotation(obs["eef_quat"])).as_quat()
        obs["eef_quat"] = quat[[3, 0, 1, 2]]
        if attached:
            obs["toy_truck_positions"] = obs["eef_pos"] - offset
        obs["truck_bottom_z"] = obs["toy_truck_positions"][2] - 0.03
        obs["opening_margin"] = 0.015 - np.max(np.abs(obs["toy_truck_positions"][:2] - obs["box_positions"][:2]))
        if ctrl.phase == "transfer":
            assert obs["truck_bottom_z"] >= obs["box_top_z"] + cfg.transport_clearance - cfg.position_tolerance
        if ctrl.phase == "wait":
            break
    else:
        pytest.fail("Varied controller did not complete within the bounded trial")
    assert ctrl.lifted and ctrl.released
    assert auto.success_ready(ctrl, True)
    assert not auto.success_ready(ctrl, False)


def test_varied_style_rejects_manual_mode_and_invalid_strength(tmp_path):
    args = auto.create_parser().parse_args([
        "--reference_dataset", str(tmp_path / "reference.hdf5"),
        "--dataset_file", str(tmp_path / "output.hdf5"), "--motion_style", "varied", "--manual_guidance",
    ])
    args.headless = False
    with pytest.raises(ValueError, match="automatic collection"):
        auto.validate_args(args)
    args.manual_guidance = False
    args.motion_variation_strength = 1.5
    with pytest.raises(ValueError, match="between 0 and 1"):
        auto.validate_args(args)


@pytest.mark.parametrize("fraction", [-0.1, 0.51, float("nan"), float("inf")])
def test_rejects_invalid_idle_target(tmp_path, fraction):
    args = auto.create_parser().parse_args([
        "--reference_dataset", str(tmp_path / "reference.hdf5"),
        "--dataset_file", str(tmp_path / "output.hdf5"),
    ])
    args.headless = True
    args.target_idle_fraction = fraction
    with pytest.raises(ValueError, match="target_idle_fraction"):
        auto.validate_args(args)
