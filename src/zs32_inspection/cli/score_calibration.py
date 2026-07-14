"""Score immutable calibration/test crops with one frozen ZS32 model set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from zs32_inspection.calibration import GroundTruth, ScoreBranch, ScoreRow, SplitRole
from zs32_inspection.data.dataset_release import verify_dataset_release
from zs32_inspection.data.calibration_targets import (
    CalibrationTargetRecord,
    CalibrationTargetValue,
    load_calibration_targets,
)
from zs32_inspection.data.manifests import CANONICAL_MANIFEST_COLUMNS, CanonicalSampleRow
from zs32_inspection.models import (
    AnomalyDINOAdapter,
    AnomalibRuntimeBackend,
    AssetFile,
    CandidateStatus,
    DeviceSpec,
    EfficientADAdapter,
    ModelInput,
    ModelSlot,
    PatchCoreAdapter,
    RawModelScore,
    UltralyticsRuntimeBackend,
    UltralyticsYoloAdapter,
)
from zs32_inspection.models.base import AnomalyPredictor, YoloPredictor
from zs32_inspection.runtime.execution_receipt import (
    collect_execution_receipt,
    verify_execution_receipt_live,
)
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    canonical_json_bytes,
    sha256_file,
)
from zs32_inspection.template import OpenCvTemplatePredictor, TemplatePredictor, TemplateScore

from ._common import command_error, command_result, load_object
from ._deployment_assets import parse_candidate, parse_template_assets
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score only calibration/test canonical crops with frozen ZS32 models"
    )
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--template-assets", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--calibration-split-id", required=True)
    parser.add_argument("--test-split-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--score-run-id", required=True)
    parser.add_argument("--device", default="0")
    return parser


def _canonical_row(payload: Mapping[str, str], index: int) -> CanonicalSampleRow:
    context = f"canonical_manifest[{index}]"
    if set(payload) != set(CANONICAL_MANIFEST_COLUMNS):
        raise ValueError(f"{context} columns differ from the canonical schema")
    try:
        if payload["is_synthetic"] not in {"true", "false"}:
            raise ValueError("is_synthetic must be exactly true or false")
        return CanonicalSampleRow(
            dataset_release_id=payload["dataset_release_id"],
            part_instance_id=payload["part_instance_id"],
            capture_set_id=payload["capture_set_id"],
            hand=payload["hand"],
            view=payload["view"],
            source_path=payload["source_path"],
            source_sha256=payload["source_sha256"],
            crop_path=payload["crop_path"],
            crop_sha256=payload["crop_sha256"],
            roi_version=payload["roi_version"],
            roi_sha256=payload["roi_sha256"],
            roi_x1=int(payload["roi_x1"]),
            roi_y1=int(payload["roi_y1"]),
            roi_x2=int(payload["roi_x2"]),
            roi_y2=int(payload["roi_y2"]),
            crop_width=int(payload["crop_width"]),
            crop_height=int(payload["crop_height"]),
            label=payload["label"],
            defect_type=payload["defect_type"],
            split=payload["split"],
            is_synthetic=payload["is_synthetic"] == "true",
            annotation_state=payload["annotation_state"],
            source_label_sha256=payload["source_label_sha256"],
            crop_label_sha256=payload["crop_label_sha256"],
            source_box_count=int(payload["source_box_count"]),
            crop_box_count=int(payload["crop_box_count"]),
            clipped_box_count=int(payload["clipped_box_count"]),
            outside_roi_box_count=int(payload["outside_roi_box_count"]),
            dropped_box_count=int(payload["dropped_box_count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {context}: {error}") from error


def _load_canonical_rows(dataset_root: Path) -> tuple[CanonicalSampleRow, ...]:
    path = dataset_root / "canonical_manifest.csv"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"canonical manifest must be a regular non-symlink file: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(CANONICAL_MANIFEST_COLUMNS):
                raise ValueError("canonical manifest header differs from the frozen schema")
            rows = tuple(_canonical_row(row, index) for index, row in enumerate(reader, start=1))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"cannot read canonical manifest {path}: {error}") from error
    if not rows:
        raise ValueError("canonical manifest contains no rows")
    return rows


def _selected_rows(
    rows: Sequence[CanonicalSampleRow],
    *,
    required_slots: Sequence[ModelSlot],
    required_views: Sequence[str],
) -> tuple[CanonicalSampleRow, ...]:
    slots = tuple(required_slots)
    hands = {slot.hand for slot in slots}
    expected_slots = {
        ModelSlot(hand, view)
        for hand in hands
        for view in required_views
    }
    if set(slots) != expected_slots:
        raise ValueError("candidate slots must cover every dataset topology view for each enabled hand")
    selected = tuple(
        row
        for row in rows
        if row.hand in hands and row.split in {SplitRole.CALIBRATION.value, SplitRole.TEST.value}
    )
    if not selected:
        raise ValueError("dataset contains no calibration/test rows for the candidate hands")
    if {row.split for row in selected} != {SplitRole.CALIBRATION.value, SplitRole.TEST.value}:
        raise ValueError("both calibration and held-out test partitions are required before scoring")
    for split in (SplitRole.CALIBRATION.value, SplitRole.TEST.value):
        observed = {ModelSlot(row.hand, row.view) for row in selected if row.split == split}
        if observed != expected_slots:
            raise ValueError(f"{split} partition does not cover every required hand/view slot")
    by_capture: dict[str, list[CanonicalSampleRow]] = defaultdict(list)
    for row in selected:
        by_capture[row.capture_set_id].append(row)
    for capture_set_id, capture_rows in by_capture.items():
        identities = {
            (row.part_instance_id, row.hand, row.label, row.split)
            for row in capture_rows
        }
        if len(identities) != 1:
            raise ValueError(f"capture {capture_set_id} mixes part/hand/label/split identities")
        hand = capture_rows[0].hand
        expected = {ModelSlot(hand, view) for view in required_views}
        actual = {ModelSlot(row.hand, row.view) for row in capture_rows}
        if len(actual) != len(capture_rows) or actual != expected:
            raise ValueError(f"capture {capture_set_id} is incomplete or duplicated for calibration scoring")
    return selected


def _verified_crop(dataset_root: Path, row: CanonicalSampleRow) -> Path:
    relative = PurePosixPath(row.crop_path)
    root = dataset_root.resolve()
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"canonical crop path traverses a symlink: {row.crop_path}")
    path = root.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(f"canonical crop escapes dataset release: {row.crop_path}") from error
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"cannot stat canonical crop {row.crop_path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"canonical crop must be a private regular file: {row.crop_path}")
    actual = sha256_file(path)
    if actual != row.crop_sha256:
        raise ValueError(
            f"canonical crop SHA256 mismatch for {row.capture_set_id}/{row.view}: "
            f"{actual} != {row.crop_sha256}"
        )
    return path


def _model_inputs(
    dataset_root: Path,
    rows: Sequence[CanonicalSampleRow],
) -> tuple[ModelInput, ...]:
    return tuple(
        ModelInput(
            sample=row.to_roi_sample(),
            crop_path=_verified_crop(dataset_root, row),
            roi_digest=row.roi_sha256,
        )
        for row in rows
    )


def _result_map(
    results: Sequence[object],
    capture_rows: Sequence[CanonicalSampleRow],
    branch: str,
) -> dict[ModelSlot, object]:
    materialized = tuple(results)
    expected = {ModelSlot(row.hand, row.view) for row in capture_rows}
    mapped: dict[ModelSlot, object] = {}
    for result in materialized:
        try:
            slot = ModelSlot(result.hand, result.view)
        except AttributeError as error:
            raise ValueError(f"{branch} predictor returned an object without hand/view identity") from error
        if slot in mapped:
            raise ValueError(f"{branch} predictor returned duplicate slot {slot.key}")
        mapped[slot] = result
    if set(mapped) != expected:
        missing = sorted(slot.key for slot in expected - set(mapped))
        unexpected = sorted(slot.key for slot in set(mapped) - expected)
        raise ValueError(f"{branch} result slots mismatch; missing={missing}, unexpected={unexpected}")
    return mapped


def _common_result_identity(result: object, row: CanonicalSampleRow, inspection_id: str) -> None:
    expected = (
        inspection_id,
        row.part_instance_id,
        row.capture_set_id,
        row.hand,
        row.view,
        row.roi_version,
        row.roi_sha256,
        row.source_sha256,
        row.crop_sha256,
    )
    actual = (
        result.inspection_id,
        result.part_instance_id,
        result.capture_set_id,
        result.hand,
        result.view,
        result.roi_config_id,
        result.roi_digest,
        result.source_sha256,
        result.crop_sha256,
    )
    if actual != expected:
        raise ValueError(f"predictor changed canonical identity for {row.capture_set_id}/{row.view}")


def _score_payload(
    row: CanonicalSampleRow,
    *,
    branch: ScoreBranch,
    score: float,
    model_digest: str,
    calibration_split_id: str,
    test_split_id: str,
    ground_truth: GroundTruth,
    part_ground_truth: GroundTruth,
) -> dict[str, object]:
    split_role = SplitRole(row.split)
    split_id = calibration_split_id if split_role is SplitRole.CALIBRATION else test_split_id
    value = ScoreRow(
        dataset_release_id=row.dataset_release_id,
        part_instance_id=row.part_instance_id,
        capture_set_id=row.capture_set_id,
        hand=row.hand,
        view=row.view,
        branch=branch,
        score=score,
        ground_truth=ground_truth,
        split_role=split_role,
        split_id=split_id,
        model_digest=model_digest,
        roi_version=row.roi_version,
        roi_digest=row.roi_sha256,
        part_ground_truth=part_ground_truth,
    )
    return {
        "dataset_release_id": value.dataset_release_id,
        "part_instance_id": value.part_instance_id,
        "capture_set_id": value.capture_set_id,
        "hand": value.hand,
        "view": value.view,
        "branch": value.branch.value,
        "score": value.score,
        "ground_truth": value.ground_truth.value,
        "part_ground_truth": value.part_ground_truth.value,
        "split_role": value.split_role.value,
        "split_id": value.split_id,
        "model_digest": value.model_digest,
        "roi_version": value.roi_version,
        "roi_digest": value.roi_digest,
    }


def _score_all(
    rows: Sequence[CanonicalSampleRow],
    *,
    dataset_root: Path,
    score_run_id: str,
    calibration_split_id: str,
    test_split_id: str,
    template_predictor: TemplatePredictor,
    anomaly_predictor: AnomalyPredictor,
    yolo_predictor: YoloPredictor,
    template_assets: Mapping[ModelSlot, object],
    anomaly_artifacts: Mapping[ModelSlot, object],
    yolo_model_digest: str,
    anomaly_family: str,
    calibration_targets: Mapping[
        tuple[str, str, str, str, str], CalibrationTargetRecord
    ],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if (
        not calibration_split_id.strip()
        or not test_split_id.strip()
        or calibration_split_id != calibration_split_id.strip()
        or test_split_id != test_split_id.strip()
    ):
        raise ValueError("calibration/test split IDs must be non-empty")
    if calibration_split_id == test_split_id:
        raise ValueError("calibration and held-out test split IDs must differ")
    by_capture: dict[str, list[CanonicalSampleRow]] = defaultdict(list)
    for row in rows:
        if row.split not in {SplitRole.CALIBRATION.value, SplitRole.TEST.value}:
            raise ValueError("train rows are forbidden at the frozen-model scoring boundary")
        by_capture[row.capture_set_id].append(row)

    triples: dict[tuple[str, str], tuple[dict[str, object] | None, ...]] = {}
    audits: dict[tuple[str, str], tuple[dict[str, object], ...]] = {}
    for capture_set_id, capture_rows in by_capture.items():
        inspection_id = f"{score_run_id}:{capture_set_id}"
        inputs = _model_inputs(dataset_root, capture_rows)
        template_by_slot = _result_map(
            template_predictor.score_batch(inputs, inspection_id=inspection_id),
            capture_rows,
            "template",
        )
        anomaly_by_slot = _result_map(
            anomaly_predictor.predict_batch(inputs, inspection_id=inspection_id),
            capture_rows,
            "anomaly",
        )
        yolo_by_slot = _result_map(
            yolo_predictor.predict_batch(inputs, inspection_id=inspection_id),
            capture_rows,
            "yolo",
        )
        for row in capture_rows:
            slot = ModelSlot(row.hand, row.view)
            template = template_by_slot[slot]
            anomaly = anomaly_by_slot[slot]
            yolo = yolo_by_slot[slot]
            if not isinstance(template, TemplateScore):
                raise ValueError(f"template predictor returned an invalid score type for {slot.key}")
            if not isinstance(anomaly, RawModelScore) or not isinstance(yolo, RawModelScore):
                raise ValueError(f"model predictor returned an invalid score type for {slot.key}")
            for result in (template, anomaly, yolo):
                _common_result_identity(result, row, inspection_id)
            template_asset = template_assets[slot]
            anomaly_artifact = anomaly_artifacts[slot]
            if (
                template.model_digest != template_asset.model_digest
                or template.best_template_sha256 not in {item.sha256 for item in template_asset.templates}
            ):
                raise ValueError(f"template result asset identity mismatch for {slot.key}")
            if (
                anomaly.branch != ScoreBranch.ANOMALY.value
                or anomaly.model_family != anomaly_family
                or anomaly.model_digest != anomaly_artifact.model_digest
            ):
                raise ValueError(f"anomaly result model identity mismatch for {slot.key}")
            if (
                yolo.branch != ScoreBranch.YOLO.value
                or yolo.model_family != "yolo"
                or yolo.model_digest != yolo_model_digest
            ):
                raise ValueError(f"YOLO result model identity mismatch for {slot.key}")
            identity = (row.capture_set_id, row.view)
            target_triple: list[CalibrationTargetRecord] = []
            for branch in (ScoreBranch.TEMPLATE, ScoreBranch.ANOMALY, ScoreBranch.YOLO):
                key = (
                    row.capture_set_id,
                    row.part_instance_id,
                    row.hand,
                    row.view,
                    branch.value,
                )
                target = calibration_targets.get(key)
                if target is None:
                    raise ValueError(
                        f"approved calibration target is missing for {key}"
                    )
                if target.part_ground_truth != row.label:
                    raise ValueError(
                        f"calibration target part truth differs from canonical label for {key}"
                    )
                target_triple.append(target)

            def target_payload(
                branch: ScoreBranch,
                raw_score: float,
                model_digest: str,
                target: CalibrationTargetRecord,
            ) -> dict[str, object] | None:
                if target.target is CalibrationTargetValue.EXCLUDE:
                    return None
                return _score_payload(
                    row,
                    branch=branch,
                    score=raw_score,
                    model_digest=model_digest,
                    calibration_split_id=calibration_split_id,
                    test_split_id=test_split_id,
                    ground_truth=GroundTruth(target.target.value),
                    part_ground_truth=GroundTruth(target.part_ground_truth),
                )
            triples[identity] = (
                target_payload(
                    ScoreBranch.TEMPLATE,
                    template.risk_score,
                    template.model_digest,
                    target_triple[0],
                ),
                target_payload(
                    ScoreBranch.ANOMALY,
                    anomaly.score,
                    anomaly.model_digest,
                    target_triple[1],
                ),
                target_payload(
                    ScoreBranch.YOLO,
                    yolo.score,
                    yolo.model_digest,
                    target_triple[2],
                ),
            )
            audits[identity] = (
                {
                    "inspection_id": inspection_id,
                    "branch": "template",
                    "source_sha256": row.source_sha256,
                    "crop_sha256": row.crop_sha256,
                    "similarity": template.similarity,
                    "best_template_sha256": template.best_template_sha256,
                    "offset_xy": list(template.offset_xy),
                    "raw_score": template.risk_score,
                },
                {
                    "inspection_id": inspection_id,
                    "branch": "anomaly",
                    "source_sha256": anomaly.source_sha256,
                    "crop_sha256": anomaly.crop_sha256,
                    "model_family": anomaly.model_family,
                    "heatmap_sha256": anomaly.heatmap_sha256,
                    "raw_score": anomaly.score,
                },
                {
                    "inspection_id": inspection_id,
                    "branch": "yolo",
                    "source_sha256": yolo.source_sha256,
                    "crop_sha256": yolo.crop_sha256,
                    "detections": [dict(item) for item in yolo.detections],
                    "raw_score": yolo.score,
                },
            )
        for row in capture_rows:
            _verified_crop(dataset_root, row)

    scores: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for row in rows:
        identity = (row.capture_set_id, row.view)
        triple = triples.get(identity)
        audit_triple = audits.get(identity)
        if triple is None or audit_triple is None:
            raise ValueError(f"missing score triple for canonical row {identity}")
        target_triple = tuple(
            calibration_targets[
                (
                    row.capture_set_id,
                    row.part_instance_id,
                    row.hand,
                    row.view,
                    branch,
                )
            ]
            for branch in ("template", "anomaly", "yolo")
        )
        for score, audit, target in zip(
            triple, audit_triple, target_triple, strict=True
        ):
            split_role = SplitRole(row.split)
            split_id = (
                calibration_split_id
                if split_role is SplitRole.CALIBRATION
                else test_split_id
            )
            score_sequence_index = None
            if score is not None:
                score_sequence_index = len(scores)
                scores.append(score)
            audit_rows.append(
                {
                    "sequence_index": len(audit_rows),
                    "score_sequence_index": score_sequence_index,
                    "included_in_calibration": score is not None,
                    "dataset_release_id": row.dataset_release_id,
                    "part_instance_id": row.part_instance_id,
                    "capture_set_id": row.capture_set_id,
                    "hand": row.hand,
                    "view": row.view,
                    "split_role": split_role.value,
                    "split_id": split_id,
                    "roi_version": row.roi_version,
                    "roi_digest": row.roi_sha256,
                    "part_ground_truth": target.part_ground_truth,
                    "calibration_target": target.target.value,
                    "target_reason": target.reason,
                    "model_digest": (
                        template_assets[ModelSlot(row.hand, row.view)].model_digest
                        if target.branch == "template"
                        else anomaly_artifacts[ModelSlot(row.hand, row.view)].model_digest
                        if target.branch == "anomaly"
                        else yolo_model_digest
                    ),
                    **audit,
                }
            )
    if len(audit_rows) != 3 * len(rows):
        raise ValueError("every canonical calibration/test row must produce three audited branch scores")
    return tuple(scores), tuple(audit_rows)


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _adapter(family, runtime):
    return {
        "patchcore": PatchCoreAdapter,
        "efficientad": EfficientADAdapter,
        "anomalydino": AnomalyDINOAdapter,
    }[family.value](runtime_loader=runtime)


def _verify_asset_path(asset_root: Path, asset: AssetFile) -> None:
    root = asset_root.expanduser()
    if root.is_symlink():
        raise ValueError(f"asset root must not be a symlink: {root}")
    root = root.resolve()
    path = asset.path.expanduser()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"frozen asset escapes the explicit asset root: {path}") from error
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"frozen asset traverses a symlink: {path}")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"cannot stat frozen asset {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"frozen asset must be a private regular file: {path}")
    asset.verify()


def _verify_frozen_assets(
    candidate,
    template_assets: Mapping[ModelSlot, object],
    asset_root: Path,
) -> None:
    assets = [
        *(artifact.checkpoint for artifact in candidate.anomaly_artifacts.values()),
        *candidate.yolo.assets,
        *(template.model for template in template_assets.values()),
        *(
            reference
            for template in template_assets.values()
            for reference in template.templates
        ),
    ]
    for asset in assets:
        _verify_asset_path(asset_root, asset)
    candidate.yolo.verify()


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    if (
        not args.score_run_id.strip()
        or args.score_run_id != args.score_run_id.strip()
        or any(character.isspace() for character in args.score_run_id)
    ):
        raise ValueError("score-run-id must be a non-empty identifier without whitespace")
    if (
        not args.calibration_split_id.strip()
        or not args.test_split_id.strip()
        or args.calibration_split_id != args.calibration_split_id.strip()
        or args.test_split_id != args.test_split_id.strip()
    ):
        raise ValueError("calibration/test split IDs must be non-empty without outer whitespace")
    if args.calibration_split_id == args.test_split_id:
        raise ValueError("calibration and held-out test split IDs must differ")
    dataset_root = args.dataset_release.expanduser()
    dataset = verify_dataset_release(dataset_root)
    dataset_manifest_sha256 = sha256_file(dataset_root / "dataset_release.json")
    canonical_manifest_path = dataset_root / "canonical_manifest.csv"
    calibration_targets_path = dataset_root / "calibration_targets.json"
    if sha256_file(canonical_manifest_path) != dataset.canonical_manifest_sha256:
        raise ValueError("canonical manifest changed after dataset release verification")
    calibration_targets = load_calibration_targets(calibration_targets_path)
    if calibration_targets.sha256 != dataset.calibration_targets_sha256:
        raise ValueError("calibration target contract differs from dataset manifest")
    candidate_path = args.candidate.expanduser()
    template_path = args.template_assets.expanduser()
    candidate_descriptor_sha256 = sha256_file(candidate_path)
    template_descriptor_sha256 = sha256_file(template_path)
    candidate = parse_candidate(
        load_object(candidate_path, "model candidate"),
        asset_root=args.asset_root,
    )
    if candidate.status is not CandidateStatus.REGISTERED:
        raise ValueError("calibration scoring requires candidate status=registered")
    required_slots, template_assets = parse_template_assets(
        load_object(template_path, "template assets"),
        asset_root=args.asset_root,
        candidate=candidate,
    )
    if (
        candidate.dataset_release_id != dataset.dataset_release_id
        or candidate.dataset_manifest_digest != dataset_manifest_sha256
        or candidate.topology_digest != dataset.topology_sha256
        or candidate.roi_version != dataset.roi_version
        or candidate.roi_digest != dataset.roi_sha256
    ):
        raise ValueError("candidate provenance differs from the verified canonical dataset release")
    if any(
        candidate.anomaly_artifacts[slot].train_split_id != dataset.split_assignments_sha256
        or template_assets[slot].train_split_id != dataset.split_assignments_sha256
        for slot in required_slots
    ):
        raise ValueError("frozen model train_split_id differs from dataset split assignments")
    rows = _selected_rows(
        _load_canonical_rows(dataset_root),
        required_slots=required_slots,
        required_views=dataset.required_views,
    )
    for row in rows:
        if row.split != SplitRole.TEST.value:
            continue
        for branch in ("template", "anomaly", "yolo"):
            target = calibration_targets.by_key[
                (
                    row.capture_set_id,
                    row.part_instance_id,
                    row.hand,
                    row.view,
                    branch,
                )
            ]
            if target.target is CalibrationTargetValue.EXCLUDE:
                raise ValueError(
                    "held-out test targets cannot be exclude because strict fusion "
                    f"evaluation would be incomplete: {target.key}"
                )
    if (
        sha256_file(candidate_path) != candidate_descriptor_sha256
        or sha256_file(template_path) != template_descriptor_sha256
        or sha256_file(canonical_manifest_path) != dataset.canonical_manifest_sha256
        or sha256_file(calibration_targets_path) != dataset.calibration_targets_sha256
    ):
        raise ValueError("a frozen descriptor or canonical manifest changed while it was parsed")
    _verify_frozen_assets(candidate, template_assets, args.asset_root)
    device = DeviceSpec(accelerator="gpu", device=args.device)
    execution_parameters = {
        "calibration_split_id": args.calibration_split_id,
        "test_split_id": args.test_split_id,
        "device": args.device,
    }
    execution_receipt = collect_execution_receipt(
        operation="score_calibration",
        device=args.device,
        input_sha256_by_role={
            "dataset_manifest": dataset_manifest_sha256,
            "canonical_manifest": dataset.canonical_manifest_sha256,
            "calibration_targets": dataset.calibration_targets_sha256,
            "candidate_descriptor": candidate_descriptor_sha256,
            "template_assets_descriptor": template_descriptor_sha256,
        },
        parameters=execution_parameters,
    )
    template_predictor = OpenCvTemplatePredictor(template_assets)
    anomaly_predictor = _adapter(
        candidate.anomaly_family,
        AnomalibRuntimeBackend(),
    ).load(candidate.anomaly_artifacts, device)
    yolo_predictor = UltralyticsYoloAdapter(
        UltralyticsRuntimeBackend()
    ).load(candidate.yolo, device)
    scores, audit = _score_all(
        rows,
        dataset_root=dataset_root,
        score_run_id=args.score_run_id,
        calibration_split_id=args.calibration_split_id,
        test_split_id=args.test_split_id,
        template_predictor=template_predictor,
        anomaly_predictor=anomaly_predictor,
        yolo_predictor=yolo_predictor,
        template_assets=template_assets,
        anomaly_artifacts=candidate.anomaly_artifacts,
        yolo_model_digest=candidate.yolo.model_digest,
        anomaly_family=candidate.anomaly_family.value,
        calibration_targets=calibration_targets.by_key,
    )
    _verify_frozen_assets(candidate, template_assets, args.asset_root)
    if (
        sha256_file(candidate_path) != candidate_descriptor_sha256
        or sha256_file(template_path) != template_descriptor_sha256
        or sha256_file(dataset_root / "dataset_release.json") != dataset_manifest_sha256
        or sha256_file(canonical_manifest_path) != dataset.canonical_manifest_sha256
        or sha256_file(calibration_targets_path) != dataset.calibration_targets_sha256
    ):
        raise ValueError("frozen scoring provenance changed during model inference")
    verify_execution_receipt_live(execution_receipt, device=args.device)
    scores_bytes = _jsonl_bytes(scores)
    audit_bytes = _jsonl_bytes(audit)
    run_payload = {
        "schema": "zs32.calibration_score_run",
        "schema_version": 3,
        "score_run_id": args.score_run_id,
        "dataset_release_id": dataset.dataset_release_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "canonical_manifest_sha256": dataset.canonical_manifest_sha256,
        "calibration_targets_sha256": dataset.calibration_targets_sha256,
        "calibration_target_count": dataset.calibration_target_count,
        "split_assignments_sha256": dataset.split_assignments_sha256,
        "calibration_split_id": args.calibration_split_id,
        "test_split_id": args.test_split_id,
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.digest,
        "candidate_descriptor_sha256": candidate_descriptor_sha256,
        "candidate_status": candidate.status.value,
        "recipe_digest": candidate.recipe_digest,
        "anomaly_family": candidate.anomaly_family.value,
        "template_assets_descriptor_sha256": template_descriptor_sha256,
        "topology_id": dataset.topology_id,
        "topology_sha256": dataset.topology_sha256,
        "roi_version": dataset.roi_version,
        "roi_sha256": dataset.roi_sha256,
        "enabled_hands": sorted({slot.hand for slot in required_slots}),
        "required_slots": [slot.key for slot in required_slots],
        "model_digests": sorted(
            {
                candidate.yolo.model_digest,
                *(candidate.anomaly_artifacts[slot].model_digest for slot in required_slots),
                *(template_assets[slot].model_digest for slot in required_slots),
            }
        ),
        "template_models": [
            {
                "hand": slot.hand,
                "view": slot.view,
                "model_digest": template_assets[slot].model_digest,
                "reference_digests": [item.sha256 for item in template_assets[slot].templates],
            }
            for slot in required_slots
        ],
        "anomaly_models": [
            {
                "hand": slot.hand,
                "view": slot.view,
                "model_digest": candidate.anomaly_artifacts[slot].model_digest,
            }
            for slot in required_slots
        ],
        "yolo_model_digest": candidate.yolo.model_digest,
        "yolo_bundle_digest": candidate.yolo.bundle_digest,
        "device": {"accelerator": device.accelerator, "device": device.device},
        "branch_order": ["template", "anomaly", "yolo"],
        "canonical_row_count": len(rows),
        "score_row_count": len(scores),
        "score_audit_row_count": len(audit),
        "excluded_score_count": len(audit) - len(scores),
        "scores_sha256": hashlib.sha256(scores_bytes).hexdigest(),
        "score_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "execution_receipt": execution_receipt.as_dict(),
    }
    run_bytes = canonical_json_bytes(run_payload)

    def validate(staging: Path) -> None:
        expected = {
            "scores.jsonl": scores_bytes,
            "score_audit.jsonl": audit_bytes,
            "score_run.json": run_bytes,
        }
        for relative, content in expected.items():
            path = staging / relative
            if path.is_symlink() or path.read_bytes() != content:
                raise PublicationError(f"staged calibration score output changed: {relative}")

    with AtomicDirectoryPublisher(args.output_root, args.score_run_id) as publisher:
        publisher.write_bytes("scores.jsonl", scores_bytes)
        publisher.write_bytes("score_audit.jsonl", audit_bytes)
        publisher.write_bytes("score_run.json", run_bytes)
        published = publisher.finalize(
            validator=validate,
            required_paths=frozenset({"scores.jsonl", "score_audit.jsonl", "score_run.json"}),
        )
    command_result(
        "zs32-score-calibration",
        {
            "candidate_digest": candidate.digest,
            "canonical_row_count": len(rows),
            "score_row_count": len(scores),
            "scores_sha256": run_payload["scores_sha256"],
            "published_path": str(published),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-score-calibration", error)


if __name__ == "__main__":
    raise SystemExit(main())
