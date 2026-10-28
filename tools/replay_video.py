# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Write one MP4 per replayed demonstration, without depending on Isaac Sim."""

from pathlib import Path


class ReplayFrameCapture:
    """Render headless scenes explicitly before reading the RGB annotator.

    ManagerBasedRLEnv.render(recompute=True) reads RGB without rendering.
    Headless environments without camera sensors also skip rendering in step().
    Initialize the annotator first, then warm up and render without advancing physics.
    """

    def __init__(self, env, warmup_frames: int = 8, max_render_attempts: int = 30):
        self.env = env
        self.warmup_frames = warmup_frames
        self.max_render_attempts = max_render_attempts
        self._initialized = False

    def capture(self):
        if not self._initialized:
            self.env.render(recompute=True)  # Attach the RGB annotator/render product.
            for _ in range(self.warmup_frames):
                self.env.sim.render()
            self._initialized = True
        for _ in range(self.max_render_attempts):
            self.env.sim.render()
            frame = self.env.render(recompute=True)
            if frame is not None and frame.size and frame.any():
                return frame
        raise RuntimeError(
            "Renderer returned only empty/black RGB frames. Video export stopped instead of saving a black movie. "
            "Check RTX rendering, camera pose, scene lighting, and NVIDIA driver availability."
        )


class ReplayVideoWriter:
    def __init__(self, directory: str, fps: float):
        import imageio.v2 as imageio

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._imageio = imageio
        self._writer = None

    def start(self, episode_name: str):
        self.close()
        path = self.directory / f"{episode_name}.mp4"
        if path.exists():
            raise FileExistsError(f"Video already exists: {path}. Choose another --video_folder.")
        self._writer = self._imageio.get_writer(
            str(path), fps=self.fps, codec="libx264", quality=8, macro_block_size=1
        )
        print(f"[VIDEO] Recording {path}")

    def append(self, frame):
        if self._writer is None:
            raise RuntimeError("Start an episode before recording frames")
        self._writer.append_data(frame)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
