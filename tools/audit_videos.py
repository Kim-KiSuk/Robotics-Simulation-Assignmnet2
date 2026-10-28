"""Check decoded video frames and create a contact sheet from representative replays."""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parents[1]
    videos = root / "videos"
    preview = videos / "previews"
    preview.mkdir(exist_ok=True)
    records = json.loads((videos / "replay_results.json").read_text())
    audit = []
    sheet = Image.new("RGB", (960, 6 * 210), "#15202b")
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(records):
        path = videos / record["video"]
        reader = imageio.get_reader(path)
        try:
            count = reader.count_frames()
            fps = float(reader.get_meta_data()["fps"])
            assert count == record["frames"], path
            assert fps == record["fps"], path
            frames = [reader.get_data(index) for index in [0, count // 2, count - 1]]
        finally:
            reader.close()
        means = [float(frame.mean()) for frame in frames]
        assert min(means) > 1.0 and all(frame.std() > 1 for frame in frames), path
        change = float(np.abs(frames[0].astype(float) - frames[-1]).mean())
        assert change > 0.1, path
        audit.append({"video": record["video"], "decoded_frames": count, "fps": fps,
                      "sample_frame_mean_intensities": means, "first_last_mean_absolute_change": change,
                      "nonblack_and_motion_verified": True})
        title = f"{Path(record['dataset']).stem} / {record['episode']}"
        draw.text((8, row * 210 + 5), title, fill="white")
        for column, frame in enumerate(frames):
            thumb = Image.fromarray(frame).resize((320, 180))
            sheet.paste(thumb, (column * 320, row * 210 + 25))
        Image.fromarray(frames[1]).resize((480, 270)).save(
            preview / f"{Path(record['dataset']).stem}_{record['episode']}.jpg", quality=88)
    sheet.save(preview / "contact_sheet.jpg", quality=90)
    (root / "reports/video_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Verified {len(audit)} videos; frame counts match original actions.")


if __name__ == "__main__":
    main()
