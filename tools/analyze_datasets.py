"""Audit original HDF5 files and generate reproducible CSV/JSON reports."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

FILES = {
    "toy2box_agibot_auto_100.hdf5": "automatic_standard_100",
    "toy2box_agibot_idle21_n20.hdf5": "automatic_varied_idle21_20_latest",
    "toy2box_agibot_auto_n20.hdf5": "automatic_varied_20_before_idle_budget",
    "toy2box_agibot_annotated_100.hdf5": "mimic_annotation_from_standard_100",
    "toy2box_agibot_n20_annotated.hdf5": "mimic_annotation_from_prior_20",
    "toy2box_agibot_mimic_n100_500.hdf5": "mimic_generated_from_standard_100",
    "toy2box_agibot_mimic_n20_500.hdf5": "mimic_generated_from_prior_20",
}
RELEASE = "https://github.com/Kim-KiSuk/Robotics-Simulation-Assignmnet2/releases/download/collection-2026-10-28"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    summaries, fingerprints, manifest = [], {}, []
    for filename, role in FILES.items():
        path = source / filename
        rows, signatures = [], {}
        with h5py.File(path, "r") as dataset:
            data = dataset["data"]
            metadata = json.loads(data.attrs["env_args"])
            sim = metadata["sim_args"]
            dt = sim["dt"] * sim["decimation"]
            for name in sorted(data, key=lambda key: int(key.split("_")[-1])):
                episode = data[name]
                actions = episode["actions"][:]
                assert actions.ndim == 2 and actions.shape[1] == 7
                assert np.isfinite(actions).all() and len(actions) > 0
                assert int(episode.attrs["num_samples"]) == len(actions)
                def check_observation(key, value):
                    if isinstance(value, h5py.Dataset):
                        assert len(value) == len(actions), f"{filename}/{name}/obs/{key} has inconsistent length"

                episode["obs"].visititems(check_observation)
                idle = (actions[:, :6] == 0).all(axis=1)
                rows.append({
                    "episode": name, "frames": len(actions), "seconds": len(actions) * dt,
                    "idle_frames": int(idle.sum()), "idle_fraction": float(idle.mean()),
                    "gripper_switches": int((np.diff(actions[:, 6]) != 0).sum()),
                    "success": int(episode.attrs.get("success", 0)),
                })
                digest = hashlib.sha256(actions.astype(np.float32).tobytes())
                digest.update(episode["obs/eef_pos"][:].astype(np.float32).tobytes())
                signatures[name] = digest.hexdigest()
            assert int(data.attrs["total"]) == sum(row["frames"] for row in rows)
        fingerprints[filename] = signatures
        with (output / f"{path.stem}.episodes.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "file": filename, "role": role, "task": metadata["env_name"], "fps": 1 / dt,
            "episodes": len(rows), "saved_successes": sum(row["success"] for row in rows),
            "total_frames": sum(row["frames"] for row in rows),
            "total_seconds": sum(row["seconds"] for row in rows),
            "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
            "min_idle_fraction": min(row["idle_fraction"] for row in rows),
            "mean_idle_fraction": float(np.mean([row["idle_fraction"] for row in rows])),
            "max_idle_fraction": max(row["idle_fraction"] for row in rows),
            "weighted_idle_fraction": sum(row["idle_frames"] for row in rows) / sum(row["frames"] for row in rows),
            "gripper_switches_min": min(row["gripper_switches"] for row in rows),
            "gripper_switches_max": max(row["gripper_switches"] for row in rows),
        }
        log = path.with_suffix(".run.json")
        if not log.exists():
            log = output / log.name
        if log.exists():
            report = json.loads(log.read_text())
            assert Path(report["dataset_file"]).name == filename
            assert report["successful_demos"] == summary["saved_successes"]
            attempts = report["attempts"]
            summary.update({
                "collection_mode": report.get("mode", "not_logged"),
                "motion_style": report.get("motion_style", "standard"), "seed": report["seed"],
                "target_idle_fraction": report.get("target_idle_fraction"),
                "attempts": len(attempts),
                "successful_attempts": sum(bool(attempt["success"]) for attempt in attempts),
                "attempt_success_fraction": sum(bool(attempt["success"]) for attempt in attempts) / len(attempts),
                "attempt_reasons": dict(Counter(attempt["reason"] for attempt in attempts)),
                "resets_completed": sum(bool(attempt.get("reset_completed")) for attempt in attempts),
            })
        summaries.append(summary)
        manifest.append({
            "file": filename, "role": role, "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "download_url": f"{RELEASE}/{filename}",
        })
        print(json.dumps(summary, ensure_ascii=False))

    lineage = []
    for raw, annotated in [
        ("toy2box_agibot_auto_100.hdf5", "toy2box_agibot_annotated_100.hdf5"),
        ("toy2box_agibot_auto_n20.hdf5", "toy2box_agibot_n20_annotated.hdf5"),
        ("toy2box_agibot_idle21_n20.hdf5", "toy2box_agibot_n20_annotated.hdf5"),
    ]:
        source_hashes = fingerprints[raw]
        annotation_hashes = set(fingerprints[annotated].values())
        lineage.append({
            "raw": raw, "annotated": annotated,
            "matched_episodes": sum(value in annotation_hashes for value in source_hashes.values()),
            "unmatched_raw_episodes": [key for key, value in source_hashes.items() if value not in annotation_hashes],
            "fingerprint": "SHA256(float32 actions + float32 eef_pos)",
        })
    (output / "dataset_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    (output / "dataset_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
    (output.parent / "data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    args = parser.parse_args()
    analyze(args.dataset_dir, args.output_dir)
