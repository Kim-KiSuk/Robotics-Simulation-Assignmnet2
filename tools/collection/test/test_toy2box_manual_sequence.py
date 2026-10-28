# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Full manual sequence tests without a simulator or GPU."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


auto = load_module("sequence_auto_test", "record_toy2box_auto.py")
seq = load_module("sequence_plan_test", "toy2box_manual_sequence.py")


def state():
    return {
        "eef_pos": np.array([0.9, -0.2, 0.93]), "eef_quat": np.array([1.0, 0, 0, 0]),
        "toy_truck_positions": np.array([0.65, -0.25, 0.38]),
        "toy_truck_orientations": np.array([1.0, 0, 0, 0]),
        "box_positions": np.array([0.92, 0.05, 0.50]), "box_orientations": np.array([1.0, 0, 0, 0]),
        "truck_bottom_z": 0.35, "box_top_z": 0.64, "box_bottom_z": 0.40,
    }


def plan(order="rotation_xy_z", grasp_rot=None):
    initial = state()
    grasp = initial["toy_truck_positions"] + [0.02, 0.01, -0.02]
    cfg = auto.ControllerConfig()
    return seq.build_sequence(
        initial, grasp, grasp_rot or Rotation.from_euler("XYZ", [0.3, -0.4, 0.7]),
        cfg, 0.005, 0.025, order,
    )


@pytest.mark.parametrize("order", ["rotation_xy_z", "xy_rotation_z"])
def test_whole_sequence_has_separate_xy_z_rotation_and_gripper_rows(order):
    steps, transport_z = plan(order)
    assert len(steps) == 11
    names = [step.name for step in steps]
    assert names[0] == "Raise to safe height"
    assert names.index("Align gripper rotation") < names.index("Descend to grasp")
    if order == "rotation_xy_z":
        assert names.index("Align gripper rotation") < names.index("Move XY above truck")
    else:
        assert names.index("Move XY above truck") < names.index("Align gripper rotation")
    for step in steps:
        keys = {key for key, _ in step.keys}
        if step.name in ("Move XY above truck", "Move XY above box"):
            assert keys <= {"W", "S", "A", "D"}
        elif step.name in ("Raise to safe height", "Descend to grasp", "Lift above box rim",
                           "Lower to release height", "Retreat upward"):
            assert keys <= {"Q", "E"}
        elif "rotation" in step.name:
            assert keys <= {"Z", "X", "T", "G", "C", "V"}
        elif "gripper" in step.name:
            assert step.keys == [("K", 1)]
    assert transport_z - 0.03 >= state()["box_top_z"] + auto.ControllerConfig().transport_clearance
    json.dumps([seq.asdict(step) for step in steps])


def test_full_plan_displays_every_row_at_once_and_n_does_not_move_robot():
    steps, z = plan()
    session = seq.ManualSequenceSession(steps, state(), SimpleNamespace(episode="demo_0"), z)
    text = session.text()
    for step in steps:
        assert step.name in text
    before = state()["eef_pos"].copy()
    session.advance()
    assert session.index == 1
    action = session.action(state(), 0.05, False)
    np.testing.assert_array_equal(action[:6], np.zeros(6))
    np.testing.assert_array_equal(session.target_pos, before)
    assert not session.lifted and not session.released


def test_session_only_accepts_actual_lift_followed_by_manual_release():
    steps, z = plan()
    session = seq.ManualSequenceSession(steps, state(), None, z)
    obs = state()
    session.action(obs, 60.0, False)
    assert session.failure is None
    assert not auto.success_ready(session, True)
    session.action(obs, 0.05, True)
    obs["toy_truck_positions"][2] += 0.1
    session.action(obs, 0.05, True)
    assert session.lifted
    assert not auto.success_ready(session, True)
    session.action(obs, 0.05, False)
    assert auto.success_ready(session, True)
    assert not auto.success_ready(session, False)


def test_full_rotation_sequence_matches_actual_tap_rotation_composition():
    target = Rotation.from_euler("XYZ", [0.3, -0.4, 0.7])
    steps, _ = plan(grasp_rot=target)
    step = next(step for step in steps if step.name == "Align gripper rotation")
    rot = Rotation.identity()
    for key, count in step.keys:
        axis, sign = auto.KEY_MOTION[key]
        delta = np.zeros(3)
        delta[axis - 3] = sign * count * 0.025
        rot = Rotation.from_euler("XYZ", delta) * rot
    assert (rot * target.inv()).magnitude() < 1e-6


def test_low_starting_pose_raises_before_rotating_or_moving_xy():
    obs = state()
    obs["eef_pos"][2] = 0.42
    steps, _ = seq.build_sequence(
        obs, obs["toy_truck_positions"] + [0.02, 0.01, -0.02], Rotation.identity(),
        auto.ControllerConfig(), 0.005, 0.025,
    )
    assert steps[0].keys[0][0] == "Q"
    assert steps[0].target_pos[2] >= obs["box_top_z"] + auto.ControllerConfig().transport_clearance


def test_full_sequence_flag_selects_manual_multiview_without_camera_recording(tmp_path):
    args = auto.create_parser().parse_args([
        "--reference_dataset", str(tmp_path / "reference.hdf5"),
        "--dataset_file", str(tmp_path / "manual_sequence.hdf5"), "--full_sequence",
    ])
    args.headless = False
    auto.validate_args(args)
    assert args.manual_guidance and args.multi_view
