"""Move old normal capture images into one shared images directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def iter_old_images(source_dir: Path, target_dir: Path):
    """Yield images from old timestamped session directories only."""
    target_dir = target_dir.resolve()
    for images_dir in sorted(source_dir.glob("*/images")):
        if not images_dir.is_dir():
            continue
        if images_dir.resolve() == target_dir:
            continue

        session_name = images_dir.parent.name
        for path in sorted(images_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield session_name, path


def unique_destination(target_dir: Path, filename: str) -> Path:
    """Return a destination path that does not overwrite an existing file."""
    destination = target_dir / filename
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    index = 1
    while True:
        candidate = target_dir / f"{stem}_dup{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="./dataset/fx11/no_hand/top/normal",
        help="Old normal root containing timestamped session directories.",
    )
    parser.add_argument(
        "--target",
        default="./dataset/fx11/no_hand/top/normal/images",
        help="Shared destination images directory.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files. Without this flag the script only prints a dry run.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)

    if not source_dir.exists():
        raise RuntimeError(f"Source directory does not exist: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    moves = []

    for session_name, source_path in iter_old_images(source_dir, target_dir):
        destination_name = f"{session_name}_{source_path.name}"
        destination_path = unique_destination(target_dir, destination_name)
        moves.append((source_path, destination_path))

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"{mode}: {len(moves)} image(s) will be moved")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")

    for source_path, destination_path in moves:
        print(f"{source_path} -> {destination_path}")
        if args.execute:
            shutil.move(str(source_path), str(destination_path))

    if not args.execute:
        print("\nDry run only. Re-run with --execute to move files.")


if __name__ == "__main__":
    main()
