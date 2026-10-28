# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Seeded variation for the scripted Toy2Box demonstrator.

Adds smooth speed changes, bounded intermediate corrections, and brief pauses.
These remain automatically generated demonstrations. No task success or grip
criteria are changed, and no noise is applied to recorded states or observations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class MotionVariationConfig:
    strength: float = 1.0
    min_speed: float = 0.45
    max_speed: float = 1.0
    position_offset: float = 0.008
    rotation_offset: float = 0.025
    precision_radius: float = 0.04
    pause_probability: float = 0.55
    pause_min: float = 0.12
    pause_max: float = 0.40
    target_idle_fraction: float = 0.21
    idle_tail_seconds: float = 0.05


def limited(vector, maximum):
    return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-12))


class MotionVariation:
    MOVING_PHASES = {"approach", "descend", "lift", "transfer", "lower", "retreat"}

    def __init__(self, seed: int, cfg: MotionVariationConfig | None = None):
        self.seed = int(seed)
        self.cfg = cfg or MotionVariationConfig()
        self.rng = np.random.default_rng(self.seed)
        # Leave some room for frame rounding and the natural release wait.
        jitter = min(0.01, self.cfg.target_idle_fraction * 0.1)
        self.target_idle_fraction = self.cfg.strength * float(self.rng.uniform(
            max(0.0, self.cfg.target_idle_fraction - jitter), self.cfg.target_idle_fraction + jitter,
        ))
        self.total_steps = 0
        self.idle_steps = 0
        self.moving_seconds = 0.0
        self.next_burst_seconds = float(self.rng.uniform(0.7, 1.2))
        self.phase = None
        self.clock = 0.0
        self.pause_remaining = 0.0
        self.speed = 1.0
        self.total_pause_seconds = 0.0
        self.last_offset = np.zeros(3)
        self.last_rotation_offset = np.zeros(3)
        self.profiles = []
        self.events = []

    def observe_action(self, action, dt: float):
        """Count applied frames using the same exact-zero definition as summarize_demos.py."""
        idle = bool(np.all(np.asarray(action)[:6] == 0))
        self.total_steps += 1
        self.idle_steps += int(idle)
        if not idle:
            self.moving_seconds += dt

    def begin(self, phase: str, distance: float):
        cfg = self.cfg
        self.phase, self.clock, self.pause_remaining = phase, 0.0, 0.0
        self.initial_distance = max(distance, cfg.precision_radius * 2)
        self.period = float(self.rng.uniform(0.8, 1.8))
        self.angle = float(self.rng.uniform(0.0, 2 * math.pi))
        self.offset = np.zeros(3)
        self.rotation_offset = np.zeros(3)
        if phase in {"approach", "transfer"}:
            angle = float(self.rng.uniform(0.0, 2 * math.pi))
            radius = float(self.rng.uniform(0.4, 1.0)) * cfg.position_offset * cfg.strength
            self.offset[:2] = radius * np.array([math.cos(angle), math.sin(angle)])
        if phase == "approach":
            self.rotation_offset = limited(self.rng.normal(size=3), 1.0) * cfg.rotation_offset * cfg.strength
        self.pause_at = (
            float(self.rng.uniform(0.2, 0.6))
            if self.rng.random() < cfg.pause_probability * cfg.strength else None
        )
        self.pause_used = False
        self.profiles.append({
            "phase": phase, "speed_period": self.period, "speed_angle": self.angle,
            "position_offset": self.offset.tolist(), "rotation_offset": self.rotation_offset.tolist(),
            "pause_at_progress": self.pause_at,
        })

    def hold_requested(self, phase: str, dt: float) -> bool:
        if self.cfg.strength <= 0 or self.phase != phase or phase not in self.MOVING_PHASES:
            return False
        if self.pause_remaining <= 1e-9 and self.target_idle_fraction > 0:
            # Opening holds the attained pose for one exactly-zero frame.
            # Later feedback corrections are usually nonzero, even while the
            # arm appears still. Reserve only that frame; count all real idle.
            tail = math.ceil(self.cfg.idle_tail_seconds / dt)
            debt = (
                self.target_idle_fraction * (self.total_steps + tail) - self.idle_steps - tail
            ) / (1.0 - self.target_idle_fraction)
            if (
                debt >= math.ceil(0.20 / dt)
                and self.moving_seconds >= self.next_burst_seconds
                and self.distance > self.cfg.precision_radius
            ):
                steps = min(math.floor(debt), max(1, round(float(self.rng.uniform(0.25, 0.65)) / dt)))
                self.pause_remaining = steps * dt
                self.moving_seconds = 0.0
                self.next_burst_seconds = float(self.rng.uniform(0.7, 1.2))
                self.events.append({
                    "phase": phase, "reason": "idle_budget", "after_step": self.total_steps,
                    "pause_seconds": self.pause_remaining,
                })
        if self.pause_remaining <= 1e-9:
            return False
        self.pause_remaining = max(0.0, self.pause_remaining - dt)
        self.total_pause_seconds += dt
        return True

    def modify(self, phase, position_error, rotation_error, controller_cfg, dt):
        """Modify the command only; keep nominal goals and transition predicates intact."""
        cfg = self.cfg
        if phase not in self.MOVING_PHASES or cfg.strength <= 0:
            return np.r_[
                limited(controller_cfg.gain * position_error, controller_cfg.max_translation),
                limited(controller_cfg.gain * rotation_error, controller_cfg.max_rotation),
            ]
        distance = float(np.linalg.norm(position_error))
        self.distance = distance
        if phase != self.phase:
            self.begin(phase, distance)
        self.clock += dt
        progress = float(np.clip(1.0 - distance / self.initial_distance, 0.0, 1.0))
        # Corrections vanish completely over the final 4 cm. They never change
        # lift/descent XY or the orientation of a grasped object.
        taper = float(np.clip((distance - cfg.precision_radius) / cfg.precision_radius, 0.0, 1.0))
        envelope = math.sin(math.pi * progress) * taper
        self.last_offset = envelope * self.offset
        self.last_rotation_offset = envelope * self.rotation_offset
        wave = 0.5 + 0.5 * math.sin(2 * math.pi * self.clock / self.period + self.angle)
        desired_speed = 1.0 - cfg.strength * (1.0 - (cfg.min_speed + (cfg.max_speed - cfg.min_speed) * wave))
        self.speed += (desired_speed - self.speed) * (1.0 - math.exp(-dt / 0.15))
        if (
            self.target_idle_fraction <= 0
            and self.pause_at is not None and not self.pause_used and progress >= self.pause_at
        ):
            self.pause_used = True
            if distance > cfg.precision_radius:
                self.pause_remaining = float(self.rng.uniform(cfg.pause_min, cfg.pause_max))
                self.events.append({"phase": phase, "progress": progress, "pause_seconds": self.pause_remaining})
        return self.speed * np.r_[
            limited(controller_cfg.gain * (position_error + self.last_offset), controller_cfg.max_translation),
            limited(controller_cfg.gain * (rotation_error + self.last_rotation_offset), controller_cfg.max_rotation),
        ]

    def export(self):
        return {
            "seed": self.seed, "config": asdict(self.cfg), "profiles": self.profiles,
            "events": self.events, "total_pause_seconds": round(self.total_pause_seconds, 4),
            "target_idle_fraction": self.target_idle_fraction,
            "applied_steps": self.total_steps, "idle_steps": self.idle_steps,
            "actual_idle_fraction": self.idle_steps / self.total_steps if self.total_steps else 0.0,
        }
