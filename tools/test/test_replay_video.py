# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise real MP4 encoding independently of the simulator/GPU."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("replay_video", Path(__file__).parents[1] / "replay_video.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_one_video_per_episode_with_correct_frames_and_simulation_fps(tmp_path):
    video = module.ReplayVideoWriter(str(tmp_path), fps=20)
    red = np.zeros((64, 64, 3), dtype=np.uint8)
    red[:, :, 0] = 240
    video.start("demo_0")
    for _ in range(12):
        video.append(red)
    video.start("demo_1")  # Finalizes the previous file.
    for _ in range(7):
        video.append(red)
    video.close()
    video.close()  # Cleanup is idempotent.
    for name, count in [("demo_0", 12), ("demo_1", 7)]:
        with imageio.get_reader(tmp_path / f"{name}.mp4") as reader:
            assert reader.count_frames() == count
            assert reader.get_meta_data()["fps"] == pytest.approx(20)
            assert reader.get_data(0)[:, :, 0].mean() > 200


def test_does_not_overwrite_an_existing_video(tmp_path):
    existing = tmp_path / "demo_0.mp4"
    existing.write_bytes(b"existing video")
    video = module.ReplayVideoWriter(str(tmp_path), fps=20)
    with pytest.raises(FileExistsError):
        video.start("demo_0")
    assert existing.read_bytes() == b"existing video"


def test_requires_an_episode_before_writing_frames(tmp_path):
    video = module.ReplayVideoWriter(str(tmp_path), fps=20)
    with pytest.raises(RuntimeError, match="Start an episode"):
        video.append(np.zeros((64, 64, 3), dtype=np.uint8))


def test_capture_renders_before_reading_and_warms_up_without_physics():
    events = []
    renders = 0
    red = np.zeros((64, 64, 3), dtype=np.uint8)
    red[:, :, 0] = 240

    def render_scene():
        nonlocal renders
        renders += 1
        events.append("render scene")

    def read_rgb(recompute=False):
        assert recompute is True
        events.append("read RGB")
        return red.copy() if renders >= 3 else np.zeros_like(red)

    env = SimpleNamespace(sim=SimpleNamespace(render=render_scene), render=read_rgb)
    capture = module.ReplayFrameCapture(env, warmup_frames=3)
    np.testing.assert_array_equal(capture.capture(), red)
    assert events[0] == "read RGB"  # Attach the render product before rendering.
    assert events[-2:] == ["render scene", "read RGB"]
    renders_before_next_frame = renders
    np.testing.assert_array_equal(capture.capture(), red)
    assert renders == renders_before_next_frame + 1  # No repeated warmup.


def test_capture_waits_for_rgb_data_instead_of_recording_black_frames():
    renders = 0

    def render_scene():
        nonlocal renders
        renders += 1

    def read_rgb(recompute=False):
        return np.full((64, 64, 3), 100 if renders >= 5 else 0, dtype=np.uint8)

    env = SimpleNamespace(sim=SimpleNamespace(render=render_scene), render=read_rgb)
    capture = module.ReplayFrameCapture(env, warmup_frames=1, max_render_attempts=10)
    assert capture.capture().mean() == 100
    assert renders == 5


def test_capture_fails_with_clear_error_when_renderer_never_produces_rgb():
    env = SimpleNamespace(
        sim=SimpleNamespace(render=lambda: None),
        render=lambda **kwargs: np.zeros((64, 64, 3), dtype=np.uint8),
    )
    capture = module.ReplayFrameCapture(env, warmup_frames=1, max_render_attempts=3)
    with pytest.raises(RuntimeError, match="empty/black RGB"):
        capture.capture()
