# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Read stamp text from a back image and save independent evidence."""

import argparse
from pathlib import Path


def main() -> int:
    """Read one image with an explicit hand/ROI config and no business rules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Full-resolution back image")
    parser.add_argument("--config", type=Path, required=True, help="Explicit left/right stamp JSON config")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory (must not exist)")
    parser.add_argument("--capture-id", help="Caller-assigned capture identity")
    parser.add_argument("--inspection-id", help="Caller-assigned inspection identity")
    parser.add_argument("--source-kind", choices=("unknown", "fused_only"), default="unknown")
    args = parser.parse_args()

    import cv2

    from bmw_inspection.checks import StampReader, StampReaderConfig

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        config = StampReaderConfig.from_json(args.config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"Invalid config: {exc}")
    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        parser.error(f"Cannot decode image: {args.image}")
    try:
        result = StampReader(config).read(
            image, capture_id=args.capture_id, inspection_id=args.inspection_id, source_kind=args.source_kind,
        )
    except (ValueError, ModuleNotFoundError) as exc:
        parser.error(str(exc))
    result.save(args.output, source=str(args.image.resolve()))
    print(
        f"{result.state}: normalized={result.normalized_code!r}; "
        f"raw={result.raw_code!r}; "
        f"score={result.code_score}; reasons={result.reasons}; evidence: {args.output}"
    )
    return 3 if result.state == "review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
