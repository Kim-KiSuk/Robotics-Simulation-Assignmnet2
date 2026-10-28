# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Offline tests; no simulator, assets, CUDA, or external dataset is required."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SPEC = importlib.util.spec_from_file_location(
    "record_toy2box_auto", Path(__file__).parents[1] / "record_toy2box_auto.py"
)
auto = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = auto
SPEC.loader.exec_module(auto)


def wxyz(rot):
    return rot.as_quat()[[3, 0, 1, 2]]


def observation():
    return {
        "eef_pos": np.array([0.5, -0.2, 0.8]),
        "eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "toy_truck_positions": np.array([0.6, -0.2, 0.38]),
        "toy_truck_orientations": np.array([1.0, 0.0, 0.0, 0.0]),
        "box_positions": np.array([0.85, 0.05, 0.5]),
        "box_orientations": np.array([1.0, 0.0, 0.0, 0.0]),
        "box_top_z": 0.62,
        "box_bottom_z": 0.4,
        "truck_bottom_z": 0.35,
        "opening_margin": -0.1,
    }


def example():
    return auto.GraspExample("demo_0", 3, np.array([0.0, 0.0, -0.02]), Rotation.identity())


def controller():
    return auto.Toy2BoxController([example()], auto.ControllerConfig())


def make_reference(path):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": auto.TASK})
        demo = data.create_group("demo_0")
        demo.attrs["success"] = True
        actions = np.zeros((12, 7))
        actions[:, 6] = 1
        actions[1:3, 6] = -1  # Failed close attempt.
        actions[5:10, 6] = -1  # Successful close and lift.
        demo["actions"] = actions
        obs = demo.create_group("obs")
        initial = observation()
        initial["toy_truck_orientations"] = wxyz(Rotation.from_euler("z", np.pi / 2))
        for key in auto.POSE_KEYS:
            value = initial[key]
            obs[key] = np.repeat(value[None], 12, axis=0)
        truck = obs["toy_truck_positions"][:]
        ee = truck + np.array([0.0, 0.02, -0.02])
        truck[7:10, 2] += 0.1
        ee[7:10, 2] += 0.1
        obs["toy_truck_positions"][:] = truck
        obs["eef_pos"][:] = ee
    return path


def test_reference_skips_failed_closures_and_transforms_to_truck_frame(tmp_path):
    grasps = auto.load_grasps(make_reference(tmp_path / "reference.hdf5"))
    assert len(grasps) == 1
    assert grasps[0].step == 7
    np.testing.assert_allclose(grasps[0].offset, [0.02, 0.0, -0.02], atol=1e-7)
    assert (grasps[0].orientation * Rotation.from_euler("z", np.pi / 2)).magnitude() < 1e-6


def test_reference_rejects_wrong_task(tmp_path):
    path = make_reference(tmp_path / "reference.hdf5")
    with h5py.File(path, "a") as f:
        f["data"].attrs["env_args"] = json.dumps({"env_name": "OtherTask"})
    with pytest.raises(ValueError, match="Reference must be"):
        auto.load_grasps(path)


def test_reference_rejects_no_actual_lift(tmp_path):
    path = make_reference(tmp_path / "reference.hdf5")
    with h5py.File(path, "a") as f:
        f["data/demo_0/obs/toy_truck_positions"][:, 2] = 0.38
    with pytest.raises(ValueError, match="No successful"):
        auto.load_grasps(path)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.3])
def test_grasp_tracks_object_orientation(yaw):
    ctrl = controller()
    obs = observation()
    obs["toy_truck_orientations"] = wxyz(Rotation.from_euler("z", yaw))
    ctrl.choose_grasp(obs)
    expected = auto.rotation(obs["toy_truck_orientations"]) * example().orientation
    assert (ctrl.grasp_rot * expected.inv()).magnitude() < 1e-7


def simulate(grip=True, box_top=0.62):
    """Ideal kinematics exercise feedback and transitions, not contact physics."""
    ctrl, obs = controller(), observation()
    obs["box_top_z"] = box_top
    attached = False
    offset = None
    phases = []
    for _ in range(1600):
        old_ee = obs["eef_pos"].copy()
        action = ctrl.action(obs, 0.05)
        phases.append(ctrl.phase)
        if ctrl.failure:
            break
        if action[6] < 0 and not attached and grip:
            attached = True
            offset = old_ee - obs["toy_truck_positions"]
        if action[6] > 0:
            attached = False
        obs["eef_pos"] += action[:3]
        obs["eef_quat"] = wxyz(Rotation.from_rotvec(action[3:6]) * auto.rotation(obs["eef_quat"]))
        if attached:
            obs["toy_truck_positions"] = obs["eef_pos"] - offset
        obs["truck_bottom_z"] = obs["toy_truck_positions"][2] - 0.03
        obs["opening_margin"] = 0.015 - np.max(np.abs(
            obs["toy_truck_positions"][:2] - obs["box_positions"][:2]
        ))
        if ctrl.phase == "wait":
            break
    return ctrl, obs, phases


def test_closed_loop_finishes_pick_place_and_releases():
    ctrl, obs, phases = simulate()
    assert ctrl.failure is None
    assert ctrl.lifted
    assert ctrl.released
    assert ctrl.phase == "wait"
    assert set(phases) == {
        "settle", "approach", "descend", "close", "lift", "transfer", "lower", "release", "retreat", "wait"
    }
    np.testing.assert_allclose(obs["toy_truck_positions"][:2], obs["box_positions"][:2], atol=0.02)
    assert abs(obs["truck_bottom_z"] - obs["box_top_z"] - ctrl.cfg.release_clearance) < 0.02


def test_failed_pick_is_discarded_before_transport():
    ctrl, _, phases = simulate(grip=False)
    assert ctrl.failure == "truck did not lift with the gripper"
    assert "transfer" not in phases
    assert not ctrl.lifted


def test_slip_is_detected():
    ctrl = controller()
    obs = observation()
    ctrl.action(obs, 0.05)
    ctrl.phase = "transfer"
    ctrl.held_offset = np.zeros(3)
    ctrl.action(obs, 0.05)
    assert ctrl.failure == "truck slipped from the gripper"


def test_rotation_and_translation_are_bounded_and_correct_sign():
    ctrl = controller()
    obs = observation()
    ctrl.action(obs, 0.05)
    ctrl.target_pos = obs["eef_pos"] + np.array([2.0, 3.0, 1.0])
    ctrl.target_rot = Rotation.from_euler("z", 1.0)
    action = ctrl.action(obs, 0.05)
    assert np.linalg.norm(action[:3]) <= ctrl.cfg.max_translation + 1e-7
    assert np.linalg.norm(action[3:6]) <= ctrl.cfg.max_rotation + 1e-7
    assert action[5] > 0
    assert action[6] == 1


def test_timeout_is_bounded():
    ctrl = controller()
    obs = observation()
    for _ in range(500):
        ctrl.action(obs, 0.05)  # Robot stays still.
        if ctrl.failure:
            break
    assert ctrl.failure == "stalled in approach"


def test_nonfinite_quaternion_is_discarded_without_rotation_error():
    ctrl = controller()
    obs = observation()
    obs["eef_quat"][0] = np.nan
    ctrl.action(obs, 0.05)
    assert ctrl.failure == "non-finite observation"


def test_output_protection(tmp_path):
    source = make_reference(tmp_path / "reference.hdf5")
    args = auto.create_parser().parse_args(["--reference_dataset", str(source), "--dataset_file", str(source)])
    with pytest.raises(ValueError, match="Output must differ"):
        auto.validate_args(args)
    output = tmp_path / "output.hdf5"
    output.write_bytes(b"existing dataset")
    args.dataset_file = output
    with pytest.raises(FileExistsError):
        auto.validate_args(args)
    assert output.read_bytes() == b"existing dataset"


def test_rejects_nan_controller_parameter(tmp_path):
    args = auto.create_parser().parse_args([
        "--reference_dataset", str(tmp_path / "reference.hdf5"),
        "--dataset_file", str(tmp_path / "output.hdf5"), "--gain", "nan",
    ])
    with pytest.raises(ValueError, match="gain must be finite"):
        auto.validate_args(args)


@pytest.mark.parametrize("box_top", [0.62, 0.85])
def test_lift_clears_box_with_truck_bottom_before_horizontal_motion(box_top):
    ctrl, obs = controller(), observation()
    obs["box_top_z"] = box_top
    ctrl.action(obs, 0.05)
    ctrl.choose_grasp(obs)
    obs["eef_pos"] = obs["toy_truck_positions"] + example().offset
    ctrl.target_pos = obs["eef_pos"].copy()
    ctrl.phase = "close"
    ctrl.phase_time = ctrl.cfg.grip_seconds
    action = ctrl.action(obs, 0.05)
    assert ctrl.phase == "lift"
    np.testing.assert_allclose(action[:2], 0.0)
    required_rise = box_top + ctrl.cfg.transport_clearance - obs["truck_bottom_z"]
    assert ctrl.target_pos[2] >= obs["eef_pos"][2] + required_rise - 1e-9

    offset = obs["eef_pos"] - obs["toy_truck_positions"]
    for _ in range(100):
        assert ctrl.phase == "lift"
        action = ctrl.action(obs, 0.05)
        if ctrl.phase == "transfer":
            assert obs["truck_bottom_z"] >= box_top + ctrl.cfg.transport_clearance
            # The transfer target preserves the reached height: no diagonal lift.
            assert ctrl.target_pos[2] == pytest.approx(obs["eef_pos"][2])
            break
        np.testing.assert_allclose(action[:2], 0.0)
        obs["eef_pos"] += action[:3]
        obs["toy_truck_positions"] = obs["eef_pos"] - offset
        obs["truck_bottom_z"] = obs["toy_truck_positions"][2] - 0.03
    else:
        pytest.fail("Did not clear the box and enter transfer")


def test_geometry_height_limits_include_rotation_and_bottom_offset():
    corners = np.array(list(auto.product([-0.1, 0.1], [-0.2, 0.2], [-0.03, 0.04])))
    low, high = auto.height_limits(corners, np.array([0.0, 0.0, 0.5]), wxyz(Rotation.identity()))
    assert low == pytest.approx(0.47)
    assert high == pytest.approx(0.54)
    low, high = auto.height_limits(
        corners, np.array([0.0, 0.0, 0.5]), wxyz(Rotation.from_euler("x", np.pi / 2))
    )
    assert low == pytest.approx(0.3)
    assert high == pytest.approx(0.7)


@pytest.mark.parametrize("succeeded", [True, False])
@pytest.mark.parametrize("release_phase", ["release", "retreat", "wait"])
@pytest.mark.parametrize("manual_mode", ["automatic", "automatic_varied", "manual_guidance", "manual_sequence"])
def test_collection_resets_immediately_even_after_last_attempt(
    monkeypatch, tmp_path, succeeded, release_phase, manual_mode
):
    """Run the real collection loop with fake simulator services, including export/reset order."""
    import torch

    manual_guidance = manual_mode in ("manual_guidance", "manual_sequence")
    varied = manual_mode == "automatic_varied"
    events = []
    applied_actions = []
    human_action = torch.zeros(7)
    human_action[6] = 1

    class FakeKeyboard:
        def __init__(self, cfg):
            pass

        def reset(self):
            pass

        def add_callback(self, *args):
            pass

        def advance(self):
            return human_action.clone()

        def pop_presses(self):
            return []

    def step(action):
        applied_actions.append(action.clone())
        return observations, None, torch.tensor([False]), torch.tensor([False]), {}
    state = observation()
    # Real successful reference demos settle ~9 cm BELOW the box root.
    state["toy_truck_positions"] = state["box_positions"] + np.array([0.0, 0.0, -0.09])
    observations = {"policy": {k: torch.tensor(state[k][None]) for k in auto.POSE_KEYS}}

    def reset():
        events.append("scene reset")
        return observations, {}

    recorder = SimpleNamespace(exported_successful_episode_count=0)
    recorder.reset = lambda: events.append("buffer reset")
    recorder.record_pre_reset = lambda *a, **kw: events.append("finalize")
    recorder.set_success_to_episodes = lambda *a: events.append("mark success")

    def export(*args):
        events.append("export")
        recorder.exported_successful_episode_count += 1

    recorder.export_episodes = export
    sim = SimpleNamespace(
        reset=lambda: events.append("sim reset"), is_playing=lambda: True, is_stopped=lambda: False
    )
    env = SimpleNamespace(
        action_manager=SimpleNamespace(total_action_dim=7), recorder_manager=recorder,
        sim=sim, reset=reset, device="cpu", step_dt=0.05,
        step=step,
        close=lambda: events.append("close"),
        scene={"toy_truck": SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=torch.zeros((1, 3))))},
    )
    env.unwrapped = env
    cfg = SimpleNamespace(
        actions=SimpleNamespace(arm_action=SimpleNamespace()),
        observations=SimpleNamespace(policy=SimpleNamespace()),
        terminations=SimpleNamespace(
            success=SimpleNamespace(func=lambda *a, **kw: torch.tensor([True]), params={}),
            toy_truck_dropping=SimpleNamespace(func=lambda *a, **kw: torch.tensor([False]), params={}),
            time_out=None,
        ),
    )
    fake_modules = {
        "gymnasium": {"make": lambda *a, **kw: env},
        "isaaclab_tasks": {},
        "isaaclab.devices": {"Se3Keyboard": FakeKeyboard, "Se3KeyboardCfg": SimpleNamespace},
        "isaaclab.envs.mdp.recorders.recorders_cfg": {"ActionStateRecorderManagerCfg": SimpleNamespace},
        "isaaclab.managers": {"DatasetExportMode": SimpleNamespace(EXPORT_SUCCEEDED_ONLY=1)},
        "isaaclab_tasks.utils.parse_cfg": {"parse_env_cfg": lambda *a, **kw: cfg},
    }
    for name, attributes in fake_modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    stub_controller = SimpleNamespace(
        phase=release_phase, failure=None if succeeded else "truck did not lift with the gripper",
        released=succeeded,
        lifted=succeeded, reference=None, elapsed=0.1, transport_z=0.8, target_pos=np.zeros(3),
        target_rot=Rotation.identity(),
        grasp_pos=np.array([0.6, -0.2, 0.36]), grasp_rot=Rotation.identity(),
        choose_grasp=lambda state: None, text=lambda: "complete manual sequence", export_plan=lambda: [],
        action=lambda *a: np.array([0.02, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    controller_options = []

    def make_controller(*args, **kwargs):
        controller_options.append(kwargs)
        return stub_controller

    monkeypatch.setattr(auto, "Toy2BoxController", make_controller)
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    monkeypatch.setattr(auto, "create_tap_keyboard", lambda base, cfg: base(cfg))
    monkeypatch.setattr(auto, "KeyboardGuideDisplay", lambda *a: SimpleNamespace(
        update=lambda *a: None, close=lambda: None
    ))
    sequence_module = ModuleType("toy2box_manual_sequence")
    sequence_module.FullSequenceDisplay = lambda: SimpleNamespace(update=lambda *a: None, close=lambda: None)
    sequence_module.ManualSequenceSession = lambda *a: stub_controller
    sequence_module.ManualViewports = lambda: SimpleNamespace(recenter=lambda env: None, close=lambda: None)
    sequence_module.build_sequence = lambda *a: ([], 0.8)
    monkeypatch.setitem(sys.modules, "toy2box_manual_sequence", sequence_module)
    monkeypatch.setattr(auto.time, "sleep", lambda seconds: None)
    corners = np.array(list(auto.product([-0.05, 0.05], [-0.05, 0.05], [-0.03, 0.05])))
    monkeypatch.setattr(auto, "read_local_bounds", lambda *a: corners)
    args = auto.create_parser().parse_args([
        "--reference_dataset", str(tmp_path / "reference.hdf5"),
        "--dataset_file", str(tmp_path / "output.hdf5"),
        "--num_demos", "2", "--max_attempts", "2", "--num_success_steps", "1",
    ])
    args.device, args.headless = "cpu", not manual_guidance
    args.manual_guidance = manual_guidance
    args.full_sequence = manual_mode == "manual_sequence"
    args.multi_view = args.full_sequence
    if varied:
        args.motion_style = "varied"
    code = auto.collect(args, [example()], SimpleNamespace(is_running=lambda: True))
    assert code == (0 if succeeded else 2)
    assert events.count("scene reset") == 3  # Initial reset and reset after both attempts.
    if succeeded:
        assert events.index("export") < len(events) - 1 - events[::-1].index("scene reset")
    else:
        assert "export" not in events
    report = json.loads(args.dataset_file.with_suffix(".run.json").read_text())
    assert report["attempts"][0]["reset_completed"] is True
    assert report["attempts"][0]["success"] is succeeded
    assert report["attempts"][1]["reset_completed"] is True
    assert report["successful_demos"] == (2 if succeeded else 0)
    assert events.count("export") == (2 if succeeded else 0)
    if succeeded and manual_guidance:
        # Advisor says move; the human presses nothing. No motion is applied.
        for action in applied_actions:
            torch.testing.assert_close(action[0], human_action)
    assert report["mode"] == ("automatic" if varied else manual_mode)
    if varied:
        assert report["motion_style"] == "varied"
        assert [row["motion_variation"]["seed"] for row in report["attempts"]] == [args.seed, args.seed + 1]
        assert all(option["motion"].cfg.strength == args.motion_variation_strength for option in controller_options)
        assert report["target_idle_fraction"] == 0.21
        assert [row["motion_variation"]["applied_steps"] for row in report["attempts"]] == (
            [1, 1] if succeeded else [0, 0]
        )
    if args.full_sequence:
        assert report["operator_views"] == ["top", "robot_left"]
        assert report["attempts"][0]["full_sequence"] == []


def test_transfer_uses_actual_truck_footprint_and_release_opens_at_rim():
    ctrl, obs = controller(), observation()
    ctrl.action(obs, 0.05)
    obs["toy_truck_positions"] = obs["box_positions"] + np.array([0.0, 0.0, 0.4])
    obs["eef_pos"] = obs["toy_truck_positions"] + np.array([0.02, 0.0, -0.02])
    ctrl.held_offset = obs["eef_pos"] - obs["toy_truck_positions"]
    obs["truck_bottom_z"] = 0.87
    obs["opening_margin"] = 0.025
    ctrl.phase = "transfer"
    ctrl.lifted = True
    ctrl.target_pos = obs["eef_pos"] + np.array([0.1, 0.0, 0.0])
    ctrl.target_rot = Rotation.from_euler("y", 0.5)
    assert not ctrl.reached(obs["eef_pos"], auto.rotation(obs["eef_quat"]))
    for _ in range(6):
        action = ctrl.action(obs, 0.05)
        if ctrl.phase == "lower":
            break
    assert ctrl.phase == "lower"  # Do not wait for an exact wrist pose.
    assert action[6] == -1
    assert ctrl.target_pos[2] == pytest.approx(
        obs["eef_pos"][2] + obs["box_top_z"] + ctrl.cfg.release_clearance - obs["truck_bottom_z"]
    )
    obs["truck_bottom_z"] = obs["box_top_z"] + ctrl.cfg.release_clearance
    action = ctrl.action(obs, 0.05)
    assert ctrl.phase == "release"
    assert action[6] == 1
    np.testing.assert_allclose(action[:6], 0.0, atol=1e-7)
    assert ctrl.failure is None


@pytest.mark.parametrize("inside", [True, False])
def test_blocked_lower_opens_only_when_truck_is_inside_opening(inside):
    ctrl, obs = controller(), observation()
    ctrl.action(obs, 0.05)
    ctrl.phase = "lower"
    ctrl.held_offset = obs["eef_pos"] - obs["toy_truck_positions"]
    ctrl.target_pos = obs["eef_pos"] - np.array([0.0, 0.0, 0.2])
    ctrl.best_error = 0.2
    ctrl.stall_time = ctrl.cfg.stall_seconds
    obs["opening_margin"] = 0.02 if inside else -0.02
    obs["truck_bottom_z"] = obs["box_top_z"] + 0.1
    action = ctrl.action(obs, 0.05)
    if inside:
        assert ctrl.phase == "release"
        assert ctrl.failure is None
        assert action[6] == 1
    else:
        assert ctrl.failure == "truck moved outside the box opening before release"
        assert not ctrl.released


def test_opening_margin_accounts_for_object_and_box_rotations():
    obs = observation()
    box = np.array(list(auto.product([-0.15, 0.15], [-0.1, 0.1], [-0.1, 0.0])))
    truck = np.array(list(auto.product([-0.04, 0.04], [-0.025, 0.025], [-0.03, 0.04])))
    obs["box_orientations"] = wxyz(Rotation.from_euler("z", np.pi / 2))
    obs["toy_truck_orientations"] = wxyz(Rotation.from_euler("z", 0.3))
    obs["toy_truck_positions"] = obs["box_positions"] + np.array([0.0, 0.0, 0.3])
    assert auto.opening_margin(box, truck, obs, 0.015) > 0
    obs["toy_truck_positions"][0] += 0.2
    assert auto.opening_margin(box, truck, obs, 0.015) < 0


def test_success_needs_actual_task_success_and_release():
    ctrl = controller()
    ctrl.lifted = True
    ctrl.phase = "lower"
    assert not auto.success_ready(ctrl, True)
    ctrl.phase = "release"
    assert not auto.success_ready(ctrl, False)
    assert auto.success_ready(ctrl, True)


def test_grasp_offset_changes_over_box_still_trigger_open_command():
    ctrl, obs = controller(), observation()
    ctrl.action(obs, 0.05)
    ctrl.phase = "lower"
    ctrl.held_offset = np.zeros(3)  # Contact/slip makes the old offset invalid.
    obs["opening_margin"] = 0.02
    obs["truck_bottom_z"] = obs["box_bottom_z"] + 0.05
    action = ctrl.action(obs, 0.05)
    assert ctrl.failure is None
    assert ctrl.phase == "release"
    assert action[6] == 1


@pytest.mark.parametrize("axis,positive,negative", [
    (0, "W", "S"), (1, "A", "D"), (2, "Q", "E"),
    (3, "Z", "X"), (4, "T", "G"), (5, "C", "V"),
])
def test_key_count_uses_existing_keyboard_bindings(axis, positive, negative):
    action = np.zeros(7)
    action[6] = 1
    action[axis] = 0.03
    assert auto.plan_key_count(action[:3], action[3:6], 0.01, 0.01, 5) == (positive, 3)
    action[axis] = -0.03
    assert auto.plan_key_count(action[:3], action[3:6], 0.01, 0.01, 5) == (negative, 3)


def test_counted_guide_toggles_gripper_only_when_needed():
    obs, ctrl = observation(), controller()
    ctrl.action(obs, 0.05)
    taps = auto.TapMotionController(0.005, 0.025, 0.25)
    taps.clear_motion(obs)
    guide = auto.CountedKeyGuide(0.005, 0.025, 5)
    advice = np.zeros(7)
    advice[6] = -1
    assert "Tap K x 1 to CLOSE" in guide.update(ctrl, obs, advice, taps, [])
    taps.action(obs, ["K"], 0.05)
    assert "Wait" in guide.update(ctrl, obs, advice, taps, ["K"])
    taps.action(obs, [], 0.3)
    assert "No key input" in guide.update(ctrl, obs, advice, taps, [])
    advice[6] = 1
    assert "Tap K x 1 to OPEN" in guide.update(ctrl, obs, advice, taps, [])


def test_manual_guidance_does_not_reset_while_person_reads_instructions():
    ctrl = auto.Toy2BoxController([example()], auto.ControllerConfig(), manual_guidance=True)
    obs = observation()
    for _ in range(10):
        ctrl.action(obs, 30.0, False)
    assert ctrl.phase == "approach"
    assert ctrl.failure is None


def test_manual_close_waits_for_person_to_press_k():
    ctrl = auto.Toy2BoxController([example()], auto.ControllerConfig(), manual_guidance=True)
    obs = observation()
    ctrl.action(obs, 0.05, False)
    ctrl.phase = "close"
    ctrl.action(obs, 10.0, False)
    assert ctrl.phase == "close"
    assert ctrl.phase_time == 0.0
    ctrl.action(obs, ctrl.cfg.grip_seconds + 0.01, True)
    assert ctrl.phase == "lift"


def test_manual_release_waits_for_actual_open_command_before_counting_success():
    ctrl = auto.Toy2BoxController([example()], auto.ControllerConfig(), manual_guidance=True)
    obs = observation()
    ctrl.action(obs, 0.05, True)
    ctrl.phase, ctrl.lifted = "release", True
    ctrl.action(obs, 10.0, True)
    assert ctrl.phase == "release"
    assert not auto.success_ready(ctrl, True)
    ctrl.action(obs, 0.1, False)
    assert auto.success_ready(ctrl, True)


def test_manual_guidance_requires_visible_gui(tmp_path):
    args = auto.create_parser().parse_args([
        "--reference_dataset", str(tmp_path / "reference.hdf5"),
        "--dataset_file", str(tmp_path / "guided.hdf5"), "--manual_guidance",
    ])
    args.headless = True
    with pytest.raises(ValueError, match="requires a GUI"):
        auto.validate_args(args)


def test_press_count_ignores_held_key_and_requires_release_for_next_count():
    queue = auto.TapQueue()
    assert queue.press("W")
    for _ in range(20):
        assert not queue.press("W")
    assert queue.pop() == ["W"]
    queue.release("W")
    assert queue.press("W")
    assert queue.pop() == ["W"]
    assert queue.press("K")
    assert not queue.press("K")
    assert queue.pop() == ["K"]


def test_keyboard_event_adapter_counts_edges_and_preserves_held_state_on_reset(monkeypatch):
    carb = ModuleType("carb")
    carb.input = SimpleNamespace(KeyboardEventType=SimpleNamespace(KEY_PRESS="down", KEY_RELEASE="up"))
    monkeypatch.setitem(sys.modules, "carb", carb)

    class BaseKeyboard:
        def __init__(self, cfg):
            self._additional_callbacks = {}

        def reset(self):
            pass

    keyboard = auto.create_tap_keyboard(BaseKeyboard, None)

    def event(key, kind):
        keyboard._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name=key), type=kind))

    event("W", "down")
    for _ in range(10):
        event("W", "down")
    assert keyboard.pop_presses() == ["W"]
    keyboard.reset()
    event("W", "down")
    assert keyboard.pop_presses() == []
    event("W", "up")
    event("W", "down")
    assert keyboard.pop_presses() == ["W"]
    callbacks = []
    keyboard._additional_callbacks["R"] = lambda: callbacks.append("reset")
    event("R", "down")
    event("R", "down")
    assert callbacks == ["reset"]


def test_one_tap_tracks_one_fixed_pose_increment_without_repeated_motion():
    obs = observation()
    taps = auto.TapMotionController(0.005, 0.025, 0.25)
    initial = obs["eef_pos"].copy()
    for step in range(30):
        action = taps.action(obs, ["W"] if step == 0 else [], 0.05)
        obs["eef_pos"] += action[:3]
    np.testing.assert_allclose(taps.target_pos, initial + [0.005, 0, 0], atol=1e-7)
    np.testing.assert_allclose(obs["eef_pos"], taps.target_pos, atol=1e-6)
    assert not taps.busy


def test_queued_taps_preserve_counts_rotation_and_gripper_state():
    obs = observation()
    taps = auto.TapMotionController(0.005, 0.025, 0.25)
    initial = obs["eef_pos"].copy()
    for step in range(80):
        action = taps.action(obs, ["W", "W", "W", "C", "K"] if step == 0 else [], 0.05)
        obs["eef_pos"] += action[:3]
        obs["eef_quat"] = wxyz(Rotation.from_rotvec(action[3:6]) * auto.rotation(obs["eef_quat"]))
    np.testing.assert_allclose(taps.target_pos, initial + [0.015, 0, 0], atol=1e-7)
    np.testing.assert_allclose(taps.target_rot.as_rotvec(), [0, 0, 0.025], atol=1e-7)
    assert taps.closed
    assert action[6] == -1


def test_count_batch_stays_fixed_until_entered_and_replans_after_settling():
    obs, ctrl = observation(), controller()
    ctrl.action(obs, 0.05)
    ctrl.phase = "approach"
    ctrl.target_pos = obs["eef_pos"] + np.array([0.05, 0, 0])
    taps = auto.TapMotionController(0.005, 0.025, 0.25)
    taps.clear_motion(obs)
    guide = auto.CountedKeyGuide(0.005, 0.025, 5)
    advice = np.array([0.025, 0, 0, 0, 0, 0, 1])
    assert "Tap W x 5" in guide.update(ctrl, obs, advice, taps, [])
    taps.action(obs, ["W", "W", "W"], 0.05)
    assert "Remaining 2 / 5" in guide.update(ctrl, obs, advice, taps, ["W", "W", "W"])
    taps.action(obs, ["W", "W"], 0.05)
    assert "Wait" in guide.update(ctrl, obs, advice, taps, ["W", "W"])
    for _ in range(80):
        action = taps.action(obs, [], 0.05)
        obs["eef_pos"] += action[:3]
    assert "Tap W x 5" in guide.update(ctrl, obs, advice, taps, [])


def test_phase_boundary_discards_queued_old_phase_movement():
    obs = observation()
    taps = auto.TapMotionController(0.005, 0.025, 0.25)
    taps.action(obs, ["W", "W", "Q", "K"], 0.05)
    taps.clear_motion(obs)
    action = taps.action(obs, [], 0.05)
    np.testing.assert_array_equal(action[:6], np.zeros(6))
    assert not taps.pending
