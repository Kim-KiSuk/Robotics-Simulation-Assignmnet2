"""Download release datasets and verify their exact SHA256 checksums."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Include annotations, earlier 20 demos, and Mimic datasets.")
    parser.add_argument("--output_dir", type=Path, default=root / "data")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in json.loads((root / "data/manifest.json").read_text()):
        if not args.all and item["role"] not in ("automatic_standard_100", "automatic_varied_idle21_20_latest"):
            continue
        path = args.output_dir / item["file"]
        if path.exists():
            if path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]:
                print("Verified:", path.name)
                continue
            raise FileExistsError(f"Existing file does not match manifest: {path}")
        partial = path.with_suffix(path.suffix + ".part")
        try:
            print("Downloading:", item["file"], flush=True)
            request = urllib.request.Request(item["download_url"], headers={"User-Agent": "Toy2Box-Archive/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
            if partial.stat().st_size != item["bytes"] or digest(partial) != item["sha256"]:
                raise ValueError(f"Checksum or size mismatch: {item['file']}")
            partial.replace(path)
            print("Verified:", path.name)
        finally:
            partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
