"""Check one explicitly configured hole from an unmasked offline source image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """Save four-state diagnostic evidence; never load the production model suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "image", "observation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("part-id", "view-id", "source-channel", "source-kind"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--synthesized-test-data", action="store_true")
    args = parser.parse_args()
    import cv2
    from bmw_inspection.checks.hole import HoleResult, inspect_hole, load_config

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        config = load_config(args.config)
        observation = json.loads(args.observation.read_text(encoding="utf-8"))
        if not isinstance(observation, dict):
            raise ValueError("observation must be a JSON object")
        image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
        result = inspect_hole(image, config, part_id=args.part_id, view_id=args.view_id,
                              source_channel=args.source_channel, source_kind=args.source_kind,
                              observation=observation, synthesized_test_data=args.synthesized_test_data)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, cv2.error) as exc:
        result = HoleResult({"status": "ERROR", "reason": str(exc), "metrics": {},
                             "config_path": str(args.config.resolve()),
                             "observation_path": str(args.observation.resolve()),
                             "part_id": args.part_id, "view_id": args.view_id,
                             "source_channel": args.source_channel, "source_kind": args.source_kind,
                             "synthesized_test_data": args.synthesized_test_data}, {})
    try:
        result.save(args.output, source=str(args.image.resolve()))
    except OSError as exc:
        print(f"ERROR: evidence save failed: {exc}")
        return 2
    print(f"{result.payload['status']}: {result.payload['reason']}; evidence: {args.output}")
    return {"PASS": 0, "NG": 1, "ERROR": 2, "REVIEW": 3}[result.payload["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
