"""Verify and apply the included Toy2Box project files to an installed IsaacLab_RS."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    snapshot = root / "isaaclab_project"
    manifest = json.loads((snapshot / "source_manifest.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab_root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Copy after verification; otherwise only show the plan.")
    args = parser.parse_args()
    target = args.isaaclab_root.resolve()
    if not (target / "isaaclab.sh").is_file():
        parser.error("isaaclab_root must contain isaaclab.sh")
    backup = target / "toy2box_overlay_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        source, destination = snapshot / relative, target / relative
        if sha256(source) != item["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {relative}")
        if destination.exists() and sha256(destination) == item["sha256"]:
            print("Unchanged:", relative)
            continue
        print("Copy:" if args.apply else "Would copy:", relative)
        if args.apply:
            if destination.exists():
                original = backup / relative
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, original)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    print("Applied verified files." if args.apply else "Verified only. Use --apply to copy; changed files are backed up.")


if __name__ == "__main__":
    main()
