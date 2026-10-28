# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Full manual Toy2Box sequence planning and operator-only top/left viewports.

No camera data is added to the policy observations or HDF5 recorder.
The complete plan is an estimate from the initial poses, not a sequence of
automatically executed actions. The operator can correct it using both views.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])


@dataclass
class SequenceStep:
    name: str
    keys: list[tuple[str, int]]
    note: str
    target_pos: list[float]
    target_quat: list[float]


def build_sequence(state, grasp_pos, grasp_rot, cfg, pos_step, rot_step, order="rotation_xy_z"):
    """Plan quantized movements, carrying rounding forward into subsequent rows."""
    steps = []
    pos, rot = state["eef_pos"].copy(), rotation(state["eef_quat"])
    truck, box = state["toy_truck_positions"], state["box_positions"]

    def add(name, keys, note):
        steps.append(SequenceStep(name, keys, note, pos.tolist(), rot.as_quat()[[3, 0, 1, 2]].tolist()))

    def translate(name, target, axes, note, round_up=False):
        nonlocal pos
        keys = []
        for axis in axes:
            delta = float(target[axis] - pos[axis])
            count = math.ceil(abs(delta) / pos_step - 1e-8) if round_up else math.floor(abs(delta) / pos_step + 0.5)
            if count:
                keys.append((("W", "S", "A", "D", "Q", "E")[2 * axis + (delta < 0)], count))
                pos[axis] += (1 if delta > 0 else -1) * count * pos_step
        add(name, keys, note)

    def orient():
        nonlocal rot
        angles = (grasp_rot * rot.inv()).as_euler("XYZ")
        keys = []
        # Left-multiplying yaw, pitch, then roll realizes intrinsic XYZ Euler
        # increments. Listing roll first would generate the wrong final pose.
        for axis in (2, 1, 0):
            count = math.floor(abs(float(angles[axis])) / rot_step + 0.5)
            if count:
                key = (("Z", "X"), ("T", "G"), ("C", "V"))[axis][int(angles[axis] < 0)]
                keys.append((key, count))
                delta = np.zeros(3)
                delta[axis] = (1 if angles[axis] > 0 else -1) * count * rot_step
                rot = Rotation.from_euler("XYZ", delta) * rot
        add("Align gripper rotation", keys, "Rotate at the safe height; keep XYZ fixed.")

    safe_z = max(float(pos[2]), float(grasp_pos[2] + cfg.approach_height),
                 float(state["box_top_z"] + cfg.transport_clearance + 0.08))
    safe = pos.copy()
    safe[2] = safe_z
    translate("Raise to safe height", safe, (2,), "Check height in LEFT view; no XY movement.", round_up=True)
    hover = np.array([grasp_pos[0], grasp_pos[1], pos[2]])
    if order == "rotation_xy_z":
        orient()
        translate("Move XY above truck", hover, (0, 1), "Use TOP view; keep height and rotation fixed.")
    elif order == "xy_rotation_z":
        translate("Move XY above truck", hover, (0, 1), "Use TOP view; keep the safe height.")
        orient()
    else:
        raise ValueError(f"Unknown sequence order: {order}")
    translate("Descend to grasp", grasp_pos, (2,), "Use LEFT view; keep XY and rotation fixed.")
    add("Close gripper", [("K", 1)], f"Tap once; wait at least {cfg.grip_seconds:g} s and check the grip.")
    # Predict object-to-gripper translation using the quantized grasp pose.
    offset = pos - truck
    bottom_offset = float(state["truck_bottom_z"] - truck[2])
    transport_z = max(float(truck[2] + cfg.lift_height),
                      float(state["box_top_z"] + cfg.transport_clearance - bottom_offset))
    lifted = pos.copy()
    lifted[2] = transport_z + offset[2]
    translate("Lift above box rim", lifted, (2,), "Verify truck BOTTOM clears the rim in LEFT view.", round_up=True)
    above_box = np.array([box[0] + offset[0], box[1] + offset[1], pos[2]])
    translate("Move XY above box", above_box, (0, 1), "Use TOP view; keep height and rotation fixed.")
    release = pos.copy()
    release[2] = state["box_top_z"] + cfg.release_clearance - bottom_offset + offset[2]
    translate("Lower to release height", release, (2,), "Use LEFT view; stop just ABOVE the rim.")
    add("Open gripper", [("K", 1)], f"Tap once; wait at least {cfg.release_seconds:g} s for the truck to fall in.")
    retreat = pos.copy()
    retreat[2] += 0.10
    translate("Retreat upward", retreat, (2,), "Check that the truck stays in the box.", round_up=True)
    add("Wait for success / reset", [], "Success is saved and counted automatically. R discards a failed attempt.")
    return steps, transport_z


class ManualSequenceSession:
    """Observe manual completion; never choose or apply a robot action."""

    def __init__(self, steps, state, reference, transport_z):
        self.steps, self.reference, self.transport_z = steps, reference, transport_z
        self.initial_truck_z = float(state["toy_truck_positions"][2])
        self.index = 0
        self.elapsed = 0.0
        self.failure = None
        self.lifted = self.released = False
        self.was_closed = False
        self.target_pos = state["eef_pos"].copy()
        self.target_rot = rotation(state["eef_quat"])

    @property
    def phase(self):
        return f"manual_{self.index + 1}: {self.steps[self.index].name}"

    def advance(self):
        self.index = min(self.index + 1, len(self.steps) - 1)

    def action(self, state, dt, gripper_closed):
        self.elapsed += dt
        if gripper_closed and state["toy_truck_positions"][2] - self.initial_truck_z >= 0.06:
            self.lifted = True
        if self.lifted and self.was_closed and not gripper_closed:
            self.released = True
        self.was_closed = gripper_closed
        self.target_pos, self.target_rot = state["eef_pos"].copy(), rotation(state["eef_quat"])
        # This value is for logging only; TapMotionController supplies real actions.
        return np.array([0, 0, 0, 0, 0, 0, -1 if gripper_closed else 1], dtype=np.float32)

    def text(self):
        lines = ["FULL SEQUENCE (estimates from initial poses; visual corrections allowed)",
                 "N: highlight next row only | R: discard/reset | P: pause",
                 "Tap/release once per count. Wait for the arm before the next row.", ""]
        for index, step in enumerate(self.steps):
            keys = " -> ".join(f"{key} x {count}" for key, count in step.keys) or "No key input"
            marker = ">>" if index == self.index else "  "
            lines.extend([f"{marker} {index + 1:02d}. {step.name}: {keys}", f"       {step.note}"])
        return "\n".join(lines)

    def export_plan(self):
        return [asdict(step) for step in self.steps]


class FullSequenceDisplay:
    def __init__(self):
        import omni.ui as ui

        self.window = ui.Window("Toy2Box Full Manual Sequence", width=630, height=760)
        self.window.setPosition(1160, 60)
        with self.window.frame:
            with ui.ScrollingFrame():
                self.label = ui.Label("Preparing full sequence...", word_wrap=True)
        self.last_text = None

    def update(self, phase, hint):
        if hint != self.last_text:
            self.label.text = hint
            print(f"[SEQUENCE]\n{hint}")
            self.last_text = hint

    def close(self):
        self.window.destroy()


class ManualViewports:
    """Two separate real viewports, sharing the same simulation and data recorder."""

    def __init__(self):
        import omni.usd
        from omni.kit.viewport.utility import create_viewport_window
        from pxr import Sdf, UsdGeom

        self.windows = []
        stage = omni.usd.get_context().get_stage()
        for index, (name, path) in enumerate([
            ("Toy2Box TOP (XY)", "/World/Toy2BoxManualViews/Top"),
            ("Toy2Box LEFT (+Y, XZ)", "/World/Toy2BoxManualViews/Left"),
        ]):
            camera = UsdGeom.Camera.Define(stage, path)
            camera.CreateFocalLengthAttr(24.0)
            camera.CreateHorizontalApertureAttr(20.955)
            camera.CreateClippingRangeAttr((0.01, 50.0))
            window = create_viewport_window(
                name, width=560, height=460, position_x=10 + index * 575, position_y=60,
                camera_path=Sdf.Path(path),
            )
            if window is None:
                self.close()
                raise RuntimeError(f"Could not create manual viewport: {name}")
            window.viewport_api.updates_enabled = True
            self.windows.append(window)

    def recenter(self, env):
        from isaacsim.core.utils.viewports import set_camera_view

        def cpu(value):
            return value[0].detach().cpu().numpy()

        center = (cpu(env.scene["toy_truck"].data.root_pos_w) + cpu(env.scene["box"].data.root_pos_w)) / 2
        base_rot = rotation(cpu(env.scene["robot"].data.root_quat_w))
        for window, offset in zip(self.windows, ([0, 0, 1.6], [0, 1.6, 0.2])):
            eye = center + base_rot.apply(offset)
            set_camera_view(eye, center, str(window.viewport_api.camera_path), viewport_api=window.viewport_api)

    def close(self):
        for window in self.windows:
            window.destroy()
        self.windows = []
