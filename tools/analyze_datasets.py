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
    "toy2box_agibot_idle21_n20.hdf5": "automatic_pace20",
    "toy2box_agibot_auto_n20.hdf5": "automatic_var20",
    "toy2box_agibot_annotated_100.hdf5": "mimic_annotation_from_standard_100",
    "toy2box_agibot_n20_annotated.hdf5": "mimic_annotation_from_var20",
    "toy2box_agibot_mimic_n100_500.hdf5": "mimic_generated_from_standard_100",
    "toy2box_agibot_mimic_n20_500.hdf5": "mimic_generated_from_var20",
}
RELEASE = "https://github.com/Kim-KiSuk/Robotics-Simulation-Assignmnet2/releases/download/collection-2026-10-28"
PRIMARY_FILES = {"toy2box_agibot_auto_100.hdf5", "toy2box_agibot_idle21_n20.hdf5"}
PUBLIC_NAMES = {
    "toy2box_agibot_auto_100.hdf5": ("AUTO-100", "toy2box_auto100.hdf5"),
    "toy2box_agibot_idle21_n20.hdf5": ("PACE-20", "toy2box_pace20.hdf5"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(source, output, include_related=False):
    output.mkdir(parents=True, exist_ok=True)
    summaries, fingerprints, manifest = [], {}, []
    for filename, role in FILES.items():
        if filename not in PRIMARY_FILES and not include_related:
            continue
        dataset_id, public_name = PUBLIC_NAMES.get(filename, ("VAR-20" if filename == "toy2box_agibot_auto_n20.hdf5" else role, filename))
        path = source / public_name
        if not path.exists():
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
        with (output / f"{Path(public_name).stem}.episodes.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "file": public_name, "original_file": filename, "dataset_id": dataset_id,
            "role": role, "task": metadata["env_name"], "fps": 1 / dt,
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
        logs = [source / Path(filename).with_suffix(".run.json"), path.with_suffix(".run.json"),
                output / "collection" / (dataset_id.replace("-", "").lower() + ".run.json"),
                output / Path(filename).with_suffix(".run.json")]
        log = next((candidate for candidate in logs if candidate.exists()), None)
        if log is not None:
            report = json.loads(log.read_text())
            assert Path(report["dataset_file"]).name in {filename, public_name}
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
        entry = {
            "file": public_name, "original_file": filename, "dataset_id": dataset_id,
            "role": role, "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if filename in PRIMARY_FILES:
            entry["download_url"] = f"{RELEASE}/{public_name}"
        else:
            entry["availability"] = "local_only_not_published"
        manifest.append(entry)
        print(json.dumps(summary, ensure_ascii=False))

    lineage = []
    for raw, annotated in [
        ("toy2box_agibot_auto_100.hdf5", "toy2box_agibot_annotated_100.hdf5"),
        ("toy2box_agibot_auto_n20.hdf5", "toy2box_agibot_n20_annotated.hdf5"),
        ("toy2box_agibot_idle21_n20.hdf5", "toy2box_agibot_n20_annotated.hdf5"),
    ]:
        if raw not in fingerprints or annotated not in fingerprints:
            continue
        source_hashes = fingerprints[raw]
        annotation_hashes = set(fingerprints[annotated].values())
        lineage.append({
            "raw": PUBLIC_NAMES.get(raw, (None, raw))[1], "annotated": annotated,
            "matched_episodes": sum(value in annotation_hashes for value in source_hashes.values()),
            "unmatched_raw_episodes": [key for key, value in source_hashes.items() if value not in annotation_hashes],
            "fingerprint": "SHA256(float32 actions + float32 eef_pos)",
        })
    (output / "dataset_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    if lineage:
        (output / "dataset_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
    (output.parent / "data").mkdir(parents=True, exist_ok=True)
    (output.parent / "data/manifest.json").write_text(json.dumps(
        [entry for entry in manifest if entry["original_file"] in PRIMARY_FILES], indent=2) + "\n")
    if include_related:
        (output / "related_local_datasets.json").write_text(json.dumps(
            [entry for entry in manifest if entry["original_file"] not in PRIMARY_FILES], indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--include_related", action="store_true", help="Also audit the five related local files.")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    args = parser.parse_args()
    analyze(args.dataset_dir, args.output_dir, args.include_related)
