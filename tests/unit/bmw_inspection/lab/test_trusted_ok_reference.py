"""Regression tests for the manual BMW trusted-OK review package."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.trusted_ok_reference import (
    TrustedOkMatcher,
    prepare_review_package,
    publish_trusted_reference_index,
)


DATASET_FIELDS = (
    "sample_id",
    "physical_part_id",
    "session_id",
    "group_id",
    "view_id",
    "camera_serial",
    "source_path",
    "source_sha256",
    "source_class",
    "business_label",
    "split",
)


def _write_part(
    root: Path,
    rows: list[dict[str, str]],
    *,
    part_id: str,
    source_class: str = "normal",
    business_label: str = "OK",
    split: str = "train",
    views: tuple[str, ...] = VIEW_ORDER,
    bad_hash: bool = False,
) -> None:
    sample_id = f"{part_id}_000001"
    for index, view_id in enumerate(views):
        image_path = root / "images" / f"{part_id}_{view_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), (index * 15, 20, 30)).save(image_path)
        source_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        rows.append(
            {
                "sample_id": sample_id,
                "physical_part_id": part_id,
                "session_id": "20260810_210030_527506",
                "group_id": "group001",
                "view_id": view_id,
                "camera_serial": f"camera-{view_id}",
                "source_path": str(image_path),
                "source_sha256": "0" * 64 if bad_hash and index == 0 else source_hash,
                "source_class": source_class,
                "business_label": business_label,
                "split": split,
            }
        )


def _manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    path = root / "dataset_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _decisions(package: Path) -> list[tuple[str, str]]:
    with (package / "review_decisions.csv").open(newline="", encoding="utf-8") as stream:
        return [
            (row["physical_part_id"], row["decision"])
            for row in csv.DictReader(stream)
        ]


def _roi_config(root: Path) -> Path:
    """Write a fixed-setup ROI asset matching the small test images."""
    reference_manifest = root / "roi-reference.csv"
    reference_manifest.write_text("fixed setup reference\n", encoding="utf-8")
    payload = {
        "schema_version": 2,
        "coordinate_system": "pixel_xyxy_half_open",
        "representative_sample_id": "representative_000001",
        "image_width": 12,
        "image_height": 10,
        "part_rois": {view: [1, 1, 11, 9] for view in VIEW_ORDER},
        "binding_mode": "fixed_setup",
        "profile_id": "test-fixed-setup",
        "capture_scope": "right",
        "reference_manifest": str(reference_manifest),
        "reference_manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
    }
    path = root / "roi.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _review_package(root: Path, *, part_count: int = 2) -> Path:
    rows: list[dict[str, str]] = []
    for index in range(part_count):
        _write_part(root, rows, part_id=f"normal-train-{index + 1:03d}")
    review = root / "review-package"
    prepare_review_package(_manifest(root, rows), review, session_id="20260810_210030_527506")
    return review


def _set_decision(review: Path, part_id: str, decision: str) -> None:
    path = review / "review_decisions.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["physical_part_id"] == part_id:
            row.update({"decision": decision, "reviewer": "test-reviewer", "review_note": "reviewed"})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("physical_part_id", "sample_id", "decision", "reviewer", "review_note"))
        writer.writeheader()
        writer.writerows(rows)


def _set_candidate_manifest_hash(review: Path) -> None:
    """Model a producer package whose frozen candidate CSV has this exact content."""
    candidate_path = review / "candidate_manifest.csv"
    package_path = review / "review_package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["candidate_manifest_sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    package_path.write_text(json.dumps(package), encoding="utf-8")


def test_publish_trusted_reference_index_copies_only_complete_approved_parts(tmp_path: Path) -> None:
    review = _review_package(tmp_path, part_count=3)
    _set_decision(review, "normal-train-001", "APPROVED")
    _set_decision(review, "normal-train-002", "REJECTED")
    release = tmp_path / "trusted-release"

    summary = publish_trusted_reference_index(review, release, _roi_config(tmp_path))

    assert summary.approved_part_count == 1
    assert summary.reference_count_by_view == {view: 1 for view in VIEW_ORDER}
    whitelist = json.loads((release / "trusted_ok_whitelist.json").read_text(encoding="utf-8"))
    index = json.loads((release / "reference_index.json").read_text(encoding="utf-8"))
    assert whitelist["approved_part_ids"] == ["normal-train-001"]
    assert len(index["references"]) == len(VIEW_ORDER)
    for view in VIEW_ORDER:
        assert len(list((release / "references" / view / "full").iterdir())) == 1
        assert len(list((release / "references" / view / "roi").iterdir())) == 1
    with pytest.raises(FileExistsError, match="already exists"):
        publish_trusted_reference_index(review, release, _roi_config(tmp_path))


@pytest.mark.parametrize("decision", ["", "UNKNOWN"])
def test_publish_trusted_reference_index_rejects_blank_or_unknown_selected_decisions(
    tmp_path: Path, decision: str
) -> None:
    review = _review_package(tmp_path, part_count=1)
    _set_decision(review, "normal-train-001", decision)

    with pytest.raises(ValueError, match="decision"):
        publish_trusted_reference_index(review, tmp_path / "trusted-release", _roi_config(tmp_path))


def test_publish_trusted_reference_index_rejects_approved_source_sha_drift(tmp_path: Path) -> None:
    review = _review_package(tmp_path, part_count=1)
    _set_decision(review, "normal-train-001", "APPROVED")
    with (review / "candidate_manifest.csv").open(newline="", encoding="utf-8") as stream:
        source_path = Path(next(csv.DictReader(stream))["source_path"])
    source_path.write_bytes(b"changed after review")

    with pytest.raises(ValueError, match="source_sha256 mismatch"):
        publish_trusted_reference_index(review, tmp_path / "trusted-release", _roi_config(tmp_path))


def test_publish_trusted_reference_index_rejects_approved_part_missing_a_view(tmp_path: Path) -> None:
    review = _review_package(tmp_path, part_count=1)
    _set_decision(review, "normal-train-001", "APPROVED")
    candidate_path = review / "candidate_manifest.csv"
    with candidate_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[:-1]
    with candidate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _set_candidate_manifest_hash(review)

    with pytest.raises(ValueError, match="exactly eight views"):
        publish_trusted_reference_index(review, tmp_path / "trusted-release", _roi_config(tmp_path))


def test_publish_trusted_reference_index_rejects_when_no_part_is_approved(tmp_path: Path) -> None:
    review = _review_package(tmp_path, part_count=1)

    with pytest.raises(ValueError, match="no APPROVED"):
        publish_trusted_reference_index(review, tmp_path / "trusted-release", _roi_config(tmp_path))


def test_publish_trusted_reference_index_rejects_candidate_substitution_despite_unchanged_approval_ids(
    tmp_path: Path,
) -> None:
    review = _review_package(tmp_path, part_count=2)
    _set_decision(review, "normal-train-001", "APPROVED")
    candidate_path = review / "candidate_manifest.csv"
    with candidate_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    replacement_by_view = {
        row["view_id"]: row
        for row in rows
        if row["physical_part_id"] == "normal-train-002"
    }
    for row in rows:
        if row["physical_part_id"] == "normal-train-001":
            replacement = replacement_by_view[row["view_id"]]
            row["source_path"] = replacement["source_path"]
            row["source_sha256"] = replacement["source_sha256"]
    with candidate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="candidate_manifest_sha256"):
        publish_trusted_reference_index(review, tmp_path / "trusted-release", _roi_config(tmp_path))


def test_review_package_contains_only_complete_training_normals_and_pending_decisions(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    _write_part(tmp_path, rows, part_id="normal-train-001")
    _write_part(tmp_path, rows, part_id="no-streak-001", source_class="no_streak", business_label="NG")
    _write_part(tmp_path, rows, part_id="normal-calibration-001", split="calibration")
    _write_part(tmp_path, rows, part_id="normal-incomplete-001", views=VIEW_ORDER[:-1])

    output = tmp_path / "review-package"
    summary = prepare_review_package(
        _manifest(tmp_path, rows), output, session_id="20260810_210030_527506"
    )

    assert summary.candidate_part_count == 1
    assert summary.candidate_image_count == 8
    assert _decisions(output) == [("normal-train-001", "PENDING")]
    assert list((output / "review" / "contact_sheets").glob("*.png")) == [
        output / "review" / "contact_sheets" / "normal-train-001.png"
    ]
    with (output / "candidate_manifest.csv").open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream))
    assert len(candidate_rows) == 8
    assert [row["view_id"] for row in candidate_rows] == list(VIEW_ORDER)
    package = json.loads((output / "review_package.json").read_text(encoding="utf-8"))
    assert package["candidate_manifest_sha256"] == hashlib.sha256(
        (output / "candidate_manifest.csv").read_bytes()
    ).hexdigest()


def test_review_package_aborts_before_publication_when_a_candidate_hash_mismatches(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    _write_part(tmp_path, rows, part_id="normal-train-001", bad_hash=True)
    output = tmp_path / "review-package"

    with pytest.raises(ValueError, match="source_sha256"):
        prepare_review_package(
            _manifest(tmp_path, rows), output, session_id="20260810_210030_527506"
        )

    assert not output.exists()


def test_review_package_rejects_any_session_other_than_the_fixed_trusted_ok_source(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    _write_part(tmp_path, rows, part_id="normal-train-001")
    output = tmp_path / "review-package"

    with pytest.raises(ValueError, match="only supports session_id"):
        prepare_review_package(
            _manifest(tmp_path, rows), output, session_id="20260810_999999_000000"
        )

    assert not output.exists()


def test_review_package_rejects_a_symlinked_candidate_source_before_publication(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    _write_part(tmp_path, rows, part_id="normal-train-001")
    source = Path(rows[0]["source_path"])
    target = tmp_path / "target.png"
    target.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(target)
    output = tmp_path / "review-package"

    with pytest.raises(ValueError, match="regular file"):
        prepare_review_package(
            _manifest(tmp_path, rows), output, session_id="20260810_210030_527506"
        )

    assert not output.exists()


def test_review_package_rejects_unsafe_physical_part_ids_before_writing_contact_sheets(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    _write_part(tmp_path, rows, part_id="normal-train-001")
    for row in rows:
        row["physical_part_id"] = "../escape"
    output = tmp_path / "review-package"

    with pytest.raises(ValueError, match="physical_part_id"):
        prepare_review_package(
            _manifest(tmp_path, rows), output, session_id="20260810_210030_527506"
        )

    assert not output.exists()


def test_review_package_never_replaces_an_existing_output_directory(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    _write_part(tmp_path, rows, part_id="normal-train-001")
    output = tmp_path / "review-package"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_review_package(
            _manifest(tmp_path, rows), output, session_id="20260810_210030_527506"
        )


def test_review_cli_requires_a_session_and_defaults_to_the_immutable_review_location() -> None:
    script = Path(__file__).resolve().parents[4] / "pipeline/bmw_lab_prepare_trusted_ok_review.py"
    spec = importlib.util.spec_from_file_location("bmw_lab_prepare_trusted_ok_review", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module.build_parser()
    defaults = parser.parse_args(["--session-id", "20260810_210030_527506"])

    assert defaults.output.name == "bmw_right_20260810_21_train_normal_v2"
    assert defaults.manifest.name == "dataset_manifest.csv"
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_publish_cli_defaults_to_the_frozen_v2_review_and_reference_paths() -> None:
    script = Path(__file__).resolve().parents[4] / "pipeline/bmw_lab_publish_trusted_ok_reference.py"
    spec = importlib.util.spec_from_file_location("bmw_lab_publish_trusted_ok_reference", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    defaults = module.build_parser().parse_args([])

    assert defaults.review_dir.name == "bmw_right_20260810_21_train_normal_v2"
    assert defaults.output.name == "bmw_right_20260810_21_train_normal_approved_v2"


def _strict_matcher_fixture(tmp_path: Path) -> tuple[Path, str, list[np.ndarray]]:
    release = tmp_path / "bmw_right_20260810_21_train_normal_approved_v2"
    references: list[dict[str, object]] = []
    patterns: list[np.ndarray] = []
    for index in range(3):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.rectangle(
            image,
            (14 + index * 7, 18 + index * 5),
            (48 + index * 6, 78 + index * 4),
            (40 + index * 50, 180 - index * 30, 240 - index * 40),
            -1,
        )
        cv2.circle(image, (92 - index * 5, 42 + index * 12), 8 + index, (255, 255, 255), -1)
        patterns.append(image)
        relative = Path("assets") / f"pattern-{index}.png"
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), image)
    approved_part_ids = [f"part-{index:02d}" for index in range(50)]
    roi_config_sha256 = "a" * 64
    whitelist = {
        "schema_version": 1,
        "status": "published",
        "review_dir": str(tmp_path / "bmw_right_20260810_21_train_normal_v2"),
        "candidate_manifest_sha256": "b" * 64,
        "review_decisions_sha256": "c" * 64,
        "roi_config_path": str(tmp_path / "roi.json"),
        "roi_config_sha256": roi_config_sha256,
        "preprocessing_identity": "pil_rgb_crop_png_v1",
        "approved_part_ids": approved_part_ids,
        "approved_decisions": [
            {
                "decision": "APPROVED",
                "physical_part_id": part_id,
                "review_note": "synthetic fixture",
                "reviewer": "test-reviewer",
                "sample_id": f"sample-{index:02d}",
            }
            for index, part_id in enumerate(approved_part_ids)
        ],
    }
    whitelist_path = release / "trusted_ok_whitelist.json"
    whitelist_path.write_text(json.dumps(whitelist, sort_keys=True), encoding="utf-8")
    whitelist_sha256 = hashlib.sha256(whitelist_path.read_bytes()).hexdigest()
    for part_index, part_id in enumerate(approved_part_ids):
        pattern_index = part_index if part_index < 3 else 0
        relative = Path("assets") / f"pattern-{pattern_index}.png"
        digest = hashlib.sha256((release / relative).read_bytes()).hexdigest()
        for view in VIEW_ORDER:
            references.append(
                {
                    "business_label": "OK",
                    "camera_serial": f"camera-{view}",
                    "full_image_path": str(relative),
                    "full_image_sha256": digest,
                    "group_id": f"group-{part_index:02d}",
                    "physical_part_id": part_id,
                    "preprocessing_identity": "pil_rgb_crop_png_v1",
                    "roi_config_sha256": roi_config_sha256,
                    "roi_image_path": str(relative),
                    "roi_image_sha256": digest,
                    "roi_xyxy": [0, 0, 128, 128],
                    "sample_id": f"sample-{part_index:02d}",
                    "session_id": "20260810_210030_527506",
                    "source_class": "normal",
                    "source_path": f"/frozen/source/{part_id}/{view}.png",
                    "source_sha256": digest,
                    "split": "train",
                    "view_id": view,
                    "whitelist_sha256": whitelist_sha256,
                }
            )
    index_path = release / "reference_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "published",
                "whitelist_sha256": whitelist_sha256,
                "roi_config_path": str(tmp_path / "roi.json"),
                "roi_config_sha256": roi_config_sha256,
                "preprocessing_identity": "pil_rgb_crop_png_v1",
                "approved_part_count": 50,
                "reference_count_by_view": {view: 50 for view in VIEW_ORDER},
                "references": references,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return release, hashlib.sha256(index_path.read_bytes()).hexdigest(), patterns


def test_matcher_selects_deterministic_aligned_correlation_winner(tmp_path: Path) -> None:
    release, index_sha256, patterns = _strict_matcher_fixture(tmp_path)
    current = cv2.warpAffine(
        patterns[1],
        np.float32([[1, 0, 2], [0, 1, -1]]),
        (128, 128),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    matcher = TrustedOkMatcher(
        release,
        expected_index_sha256=index_sha256,
        max_shift=12,
    )
    matcher.preload()

    match = matcher.match("front", current, current, comparison_mode="roi")

    assert match.physical_part_id == "part-01"
    assert match.sample_id == "sample-01"
    assert match.similarity > 0.99
    assert abs(match.shift_x) <= 12
    assert abs(match.shift_y) <= 12
    assert match.difference_overlay.shape == (512, 512, 3)
    assert match.difference_overlay.flags.writeable is False
    assert match.index_sha256 == index_sha256
    assert matcher.match("front", current, current, comparison_mode="roi").physical_part_id == "part-01"


def test_matcher_requires_exact_approved_v2_identity_and_index_sha256(tmp_path: Path) -> None:
    release, index_sha256, _patterns = _strict_matcher_fixture(tmp_path)
    renamed = tmp_path / "synthetic_approved_v2"
    release.rename(renamed)

    with pytest.raises(ValueError, match="release identity"):
        TrustedOkMatcher(renamed, expected_index_sha256=index_sha256)


def test_matcher_rejects_index_sha256_mismatch(tmp_path: Path) -> None:
    release, _index_sha256, _patterns = _strict_matcher_fixture(tmp_path)

    with pytest.raises(ValueError, match="index SHA-256"):
        TrustedOkMatcher(release, expected_index_sha256="0" * 64)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("count", "reference_count_by_view"),
        ("session", "session_id"),
        ("source_hash", "source_sha256"),
        ("row_whitelist", "whitelist_sha256"),
        ("part_incomplete", "400 references"),
    ],
)
def test_matcher_rejects_noncanonical_v2_index(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    release, _index_sha256, _patterns = _strict_matcher_fixture(tmp_path)
    index_path = release / "reference_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if mutation == "count":
        index["reference_count_by_view"]["front"] = 49
    elif mutation == "session":
        index["references"][0]["session_id"] = "wrong-session"
    elif mutation == "source_hash":
        index["references"][0]["source_sha256"] = "d" * 64
    elif mutation == "row_whitelist":
        index["references"][0]["whitelist_sha256"] = "e" * 64
    else:
        index["references"].pop()
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    changed_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=message):
        TrustedOkMatcher(release, expected_index_sha256=changed_sha256)


def test_matcher_preload_is_complete_and_idempotent_without_second_file_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, index_sha256, _patterns = _strict_matcher_fixture(tmp_path)
    matcher = TrustedOkMatcher(release, expected_index_sha256=index_sha256)
    original = TrustedOkMatcher._load_verified
    loads: list[Path] = []

    def counting_load(path: Path, expected_sha256: str) -> np.ndarray:
        loads.append(path)
        return original(path, expected_sha256)

    monkeypatch.setattr(TrustedOkMatcher, "_load_verified", staticmethod(counting_load))

    matcher.preload()
    first_load_count = len(loads)
    matcher.preload()

    expected_keys = {(view, "roi") for view in VIEW_ORDER} | {("front_left", "full")}
    assert set(matcher._prepared) == expected_keys
    assert first_load_count == 9 * 50
    assert len(loads) == first_load_count


def test_matcher_requires_preload_before_matching(tmp_path: Path) -> None:
    release, index_sha256, patterns = _strict_matcher_fixture(tmp_path)
    matcher = TrustedOkMatcher(release, expected_index_sha256=index_sha256)

    with pytest.raises(RuntimeError, match="preload"):
        matcher.match("front", patterns[0], patterns[0], comparison_mode="roi")
