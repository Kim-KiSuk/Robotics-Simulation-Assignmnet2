# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Render representative action replays from Toy2Box HDF5 files with Isaac Lab.

Run with IsaacLab_RS/isaaclab.sh -p and --headless. Cameras are enabled by this
script. Original initial states and actions are replayed without early resets.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset_files", nargs="+", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--samples_per_dataset", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.samples_per_dataset < 1:
    parser.error("samples_per_dataset must be positive")
app = AppLauncher(args)

import json
import sys

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

sys.path.insert(0, str(Path(__file__).parent))
from replay_video import ReplayFrameCapture


def main():
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    env = None
    active_task = None
    try:
        with torch.inference_mode():
            for dataset in args.dataset_files:
                handler = HDF5DatasetFileHandler()
                handler.open(str(dataset))
                try:
                    task = handler.get_env_name()
                    if task != active_task:
                        if env is not None:
                            raise ValueError("All datasets must use the same task in one process")
                        cfg = parse_env_cfg(task, device=args.device, num_envs=1)
                        success_term = cfg.terminations.success
                        cfg.recorders = {}
                        cfg.terminations = {}
                        cfg.observations.policy.concatenate_terms = False
                        cfg.observations.policy.enable_corruption = False
                        cfg.actions.arm_action.use_relative_mode = True
                        cfg.actions.arm_action.scale = 1.0
                        cfg.viewer.resolution = (960, 540)
                        env = gym.make(task, cfg=cfg, render_mode="rgb_array").unwrapped
                        active_task = task
                    names = sorted(handler.get_episode_names(), key=lambda name: int(name.split("_")[-1]))
                    indices = np.unique(np.linspace(0, len(names) - 1, min(len(names), args.samples_per_dataset), dtype=int))
                    folder = args.output_dir / dataset.stem.removeprefix("toy2box_")
                    folder.mkdir(parents=True, exist_ok=True)
                    for index in indices:
                        name = names[int(index)]
                        destination = folder / f"{name}.mp4"
                        if destination.exists():
                            raise FileExistsError(destination)
                        episode = handler.load_episode(name, env.device)
                        env.sim.reset()
                        env.reset()
                        env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
                        capture = ReplayFrameCapture(env)
                        capture.capture()  # Render warmup without advancing physics.
                        frames, successes = 0, 0
                        peak_successes = 0
                        with imageio.get_writer(str(destination), fps=1 / env.step_dt, codec="libx264",
                                                quality=7, macro_block_size=1) as writer:
                            while True:
                                action = episode.get_next_action()
                                if action is None:
                                    break
                                env.step(action.unsqueeze(0))
                                writer.append_data(capture.capture())
                                frames += 1
                                success = bool(success_term.func(env, **success_term.params)[0])
                                successes = successes + 1 if success else 0
                                peak_successes = max(peak_successes, successes)
                        record = {
                            "dataset": dataset.name, "episode": name,
                            "video": str(destination.relative_to(args.output_dir)),
                            "frames": frames, "fps": 1 / env.step_dt,
                            "replay_task_success_10_steps": peak_successes >= 10,
                            "peak_consecutive_success_steps": peak_successes,
                            "replay_final_task_success": success,
                            "method": "initial_state_restore_then_recorded_action_replay",
                        }
                        records.append(record)
                        (args.output_dir / "replay_results.json").write_text(json.dumps(records, indent=2) + "\n")
                        print("[REPLAY] " + json.dumps(record), flush=True)
                finally:
                    handler.close()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.app.close()
