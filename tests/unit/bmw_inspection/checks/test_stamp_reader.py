# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for spatial code selection and evidence preservation."""

import json
import subprocess
import sys
from dataclasses import asdict, replace

import cv2
import numpy as np
import pytest

from bmw_inspection.checks.stamp_reader import StampReader, StampReaderConfig


@pytest.fixture
def config():
    return StampReaderConfig("left", "back", (100, 80), (10, 10, 90, 70), 90, (5, 5, 55, 35), 0.95, 2)


def line(text="5A9D6B3.02", score=0.99, y=10):
    return ([[10, y], [45, y], [45, y + 10], [10, y + 10]], text, score)


def reader(config, lines):
    return StampReader(config, backend=lambda image: (lines, [0.01, 0.02, 0.03]))


def test_preserves_punctuation_and_input(config):
    image = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
    before = image.copy()
    result = reader(config, [line(), line("BMW", y=50)]).read(image)
    assert result.payload["raw_code"] == "5A9D6B3.02"
    assert result.payload["normalized_code"] == "5A9D6B3.02"
    assert result.payload["removed_characters"] == []
    assert result.payload["state"] == "readable"
    assert result.payload["lines"][1]["text"] == "BMW"
    assert result.payload["source_kind"] == "unknown"
    np.testing.assert_array_equal(image, before)
    assert result.roi_upright.shape == (80, 60, 3)


def test_missing_code_does_not_fallback_to_brand(config):
    result = reader(config, [line("BMW", y=50)]).read(np.zeros((80, 100, 3), dtype=np.uint8))
    assert result.payload["raw_code"] is None
    assert result.payload["normalized_code"] is None
    assert result.payload["reasons"] == ["missing_code"]


@pytest.mark.parametrize("lines, reason", [
    ([line(score=0.4)], "low_score"),
    ([line(), line("02", y=20)], "ambiguous_code_lines"),
    ([line(y=5)], "code_box_near_region_border"),
    ([line(text="")], "empty_code"),
])
def test_uncertainty_does_not_become_product_verdict(config, lines, reason):
    result = reader(config, lines).read(np.zeros((80, 100, 3), dtype=np.uint8))
    assert result.payload["state"] == "review"
    assert reason in result.payload["reasons"]
    assert result.payload["business_rules_evaluated"] is False


@pytest.mark.parametrize("update", [
    {"roi_xyxy": (-1, 0, 50, 50)}, {"roi_xyxy": (10, 10, 10, 20)},
    {"code_region_xyxy": (0, 0, 61, 80)}, {"hand": "auto"}, {"view_id": "front"},
    {"reference_image_size": (100.0, 80)}, {"min_score": float("nan")},
    {"rotation_clockwise_degrees": 0}, {"border_margin_px": -1},
    {"normalization_policy": "alphanumeric"}, {"normalization_policy": None},
    {"normalization_policy": []},
])
def test_invalid_config_rejected(config, update):
    with pytest.raises(ValueError):
        replace(config, **update)


@pytest.mark.parametrize("image", [
    np.zeros((80, 100, 3), dtype=np.float32), np.zeros((80, 100), dtype=np.uint8),
    np.zeros((80, 101, 3), dtype=np.uint8),
])
def test_invalid_input_rejected_before_backend(config, image):
    def backend(image):
        pytest.fail("OCR must not run on invalid input")
    with pytest.raises(ValueError):
        StampReader(config, backend=backend).read(image)


def test_backend_mutation_cannot_change_evidence(config):
    def backend(image):
        image[:] = 255
        return [line()], None
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    result = StampReader(config, backend=backend).read(image)
    assert not image.any()
    assert not result.roi_original.any()
    assert not result.roi_upright.any()


def test_backend_errors_propagate(config):
    def backend(image):
        raise RuntimeError("backend failed")
    with pytest.raises(RuntimeError, match="backend failed"):
        StampReader(config, backend=backend).read(np.zeros((80, 100, 3), dtype=np.uint8))


def test_evidence_no_overwrite(config, tmp_path):
    result = reader(config, [line()]).read(np.zeros((80, 100, 3), dtype=np.uint8))
    output = tmp_path / "result"
    result.save(output, source="back.png")
    saved = json.loads((output / "result.json").read_text())
    assert saved["source"] == "back.png"
    assert saved["raw_code"] == "5A9D6B3.02"
    assert {p.name for p in output.iterdir()} == {"result.json", "roi_original.png", "roi_upright.png", "overlay.png"}
    with pytest.raises(FileExistsError):
        result.save(output)


def test_cli_help_without_ocr():
    result = subprocess.run(
        [sys.executable, "-m", "bmw_inspection.cli.read_stamp", "--help"], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_high_confidence_partial_span_is_review(config):
    partial = ([[10, 10], [30, 10], [30, 20], [10, 20]], "9D6B4 02", 0.999)
    result = reader(config, [partial]).read(np.zeros((80, 100, 3), dtype=np.uint8))
    assert result.payload["raw_code"] == "9D6B4 02"
    assert result.payload["state"] == "review"
    assert "incomplete_code_span" in result.payload["reasons"]


def test_cli_rejects_16bit_image(config, tmp_path, monkeypatch):
    from bmw_inspection.cli.read_stamp import main

    image = tmp_path / "back.png"
    cv2.imwrite(str(image), np.zeros((80, 100, 3), dtype=np.uint16))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "output"
    argv = ["read_stamp", "--image", str(image), "--config", str(config_path), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert not output.exists()


def test_cli_review_exit_code_and_evidence(config, tmp_path, monkeypatch, capsys):
    from bmw_inspection.cli.read_stamp import main

    image = tmp_path / "back.png"
    cv2.imwrite(str(image), np.zeros((80, 100, 3), dtype=np.uint8))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "output"
    argv = ["read_stamp", "--image", str(image), "--config", str(config_path), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(StampReader, "_engine", lambda self: lambda image: ([line(score=0.4)], None))
    assert main() == 3
    assert "low_score" in capsys.readouterr().out
    assert json.loads((output / "result.json").read_text())["code_score"] == 0.4


def test_alphanumeric_cleanup_preserves_raw_observation(config):
    config = replace(config, normalization_policy="ascii_alphanumeric")
    raw = " 5A9D6B3.02\t"
    payload = reader(config, [line(raw)]).read(np.zeros((80, 100, 3), dtype=np.uint8)).payload
    assert payload["normalized_code"] == "5A9D6B302"
    assert payload["raw_code"] == raw
    assert payload["lines"][0]["text"] == raw
    assert payload["removed_characters"] == [
        {"position": 0, "character": " ", "reason": "ascii_whitespace"},
        {"position": 8, "character": ".", "reason": "ascii_punctuation"},
        {"position": 11, "character": "\t", "reason": "ascii_whitespace"},
    ]
    assert payload["unsupported_characters"] == []
    assert payload["state"] == "readable"


def test_alphanumeric_cleanup_never_substitutes_characters(config):
    config = replace(config, normalization_policy="ascii_alphanumeric")
    payload = reader(config, [line("bB8oO0-aZ")]).read(np.zeros((80, 100, 3), dtype=np.uint8)).payload
    assert payload["normalized_code"] == "bB8oO0aZ"


@pytest.mark.parametrize("character", ["中", "é", "８", "©", "\x00"])
def test_unsupported_characters_remain_visible_and_require_review(config, character):
    config = replace(config, normalization_policy="ascii_alphanumeric")
    raw = f"A.{character}8"
    payload = reader(config, [line(raw)]).read(np.zeros((80, 100, 3), dtype=np.uint8)).payload
    assert payload["raw_code"] == raw
    assert payload["normalized_code"] == f"A{character}8"
    assert payload["unsupported_characters"] == [{"position": 2, "character": character}]
    assert payload["state"] == "review"
    assert "unsupported_characters" in payload["reasons"]


@pytest.mark.parametrize("raw", ["...", " \t-._", ""])
def test_empty_normalized_code_requires_review(config, raw):
    config = replace(config, normalization_policy="ascii_alphanumeric")
    payload = reader(config, [line(raw)]).read(np.zeros((80, 100, 3), dtype=np.uint8)).payload
    assert payload["normalized_code"] == ""
    assert payload["raw_code"] == raw
    assert payload["state"] == "review"
    assert "empty_code" in payload["reasons"]


def test_normalization_does_not_merge_ambiguous_candidates(config):
    config = replace(config, normalization_policy="ascii_alphanumeric")
    payload = reader(config, [line("AB."), line("02", y=20)]).read(np.zeros((80, 100, 3), dtype=np.uint8)).payload
    assert payload["raw_code"] is None
    assert payload["normalized_code"] is None
    assert payload["removed_characters"] == []
    assert payload["state"] == "review"


def test_cli_displays_normalized_and_raw_code(config, tmp_path, monkeypatch, capsys):
    from bmw_inspection.cli.read_stamp import main

    config = replace(config, normalization_policy="ascii_alphanumeric")
    image = tmp_path / "back.png"
    cv2.imwrite(str(image), np.zeros((80, 100, 3), dtype=np.uint8))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "output"
    argv = ["read_stamp", "--image", str(image), "--config", str(config_path), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(StampReader, "_engine", lambda self: lambda image: ([line()], None))
    assert main() == 0
    stdout = capsys.readouterr().out
    assert "normalized='5A9D6B302'" in stdout
    assert "raw='5A9D6B3.02'" in stdout
    saved = json.loads((output / "result.json").read_text())
    assert saved["normalized_code"] == "5A9D6B302"
    assert saved["raw_code"] == "5A9D6B3.02"


def test_unicode_punctuation_and_whitespace_cleanup(config):
    config = replace(config, normalization_policy="ascii_alphanumeric")
    payload = reader(config, [line("A。Ｂ，8\u00a0")]).read(np.zeros((80, 100, 3), dtype=np.uint8)).payload
    assert payload["normalized_code"] == "AＢ8"
    assert [item["reason"] for item in payload["removed_characters"]] == [
        "unicode_punctuation", "unicode_punctuation", "unicode_whitespace",
    ]
    assert payload["unsupported_characters"] == [{"position": 2, "character": "Ｂ"}]
    assert payload["state"] == "review"


def fallback_reader(config, primary, direct):
    config = replace(config, normalization_policy="ascii_alphanumeric", line_quad_xy=((6, 6), (54, 6), (54, 34), (6, 34)))
    responses = iter(direct)

    def backend(image, **kwargs):
        if kwargs:
            assert kwargs == {"use_det": False, "use_cls": False, "use_rec": True}
            image[:] = 0  # Backend mutation must not damage saved evidence.
            return next(responses), [0.01]
        return primary, None

    return StampReader(config, backend=backend)


def test_rectified_consensus_resolves_low_score_and_saves_evidence(config, tmp_path):
    image = np.full((80, 100, 3), 150, dtype=np.uint8)
    result = fallback_reader(config, [line("AB 80", score=0.91)], [[("AB.80", 0.92)], [("AB80", 0.96)]]).read(image)
    payload = result.payload
    assert payload["state"] == "readable"
    assert payload["normalized_code"] == "AB80"
    assert payload["raw_code"] == "AB.80"
    assert payload["code_score"] == 0.92
    assert payload["selected_source"] == "rectified_consensus"
    assert payload["primary_reading"]["raw_code"] == "AB 80"
    assert payload["primary_reading"]["reasons"] == ["low_score"]
    assert payload["lines"][0]["text"] == "AB 80"
    assert np.all(image == 150)
    assert result.line_original.any()
    assert result.line_clahe.any()
    output = tmp_path / "evidence"
    result.save(output)
    assert (output / "line_original.png").exists()
    assert (output / "line_clahe.png").exists()
    saved = json.loads((output / "result.json").read_text())
    assert len(saved["rectified_reading"]["variants"]) == 2
    assert np.asarray(saved["rectified_reading"]["upright_to_line_transform"]).shape == (3, 3)


@pytest.mark.parametrize("primary,direct,reason", [
    ([line("AB80", score=0.91)], [[("AB80", .96)], [("A880", .97)]], "rectified_disagreement"),
    ([line("AB80", score=0.91)], [[("A880", .96)], [("A880", .97)]], "rectified_primary_conflict"),
    ([line("AB80", score=0.91)], [[("AB80", .89)], [("AB80", .97)]], "rectified_low_score"),
    ([], [[("AB80", .96)], [("AB80", .97)]], "missing_code"),
    ([line("AB80", y=5)], [[("AB80", .96)], [("AB80", .97)]], "code_box_near_region_border"),
    ([line("AB80", score=.91)], [[], [("AB80", .97)]], "rectified_ambiguous_or_missing"),
    ([line("AB80", score=.91)], [[("AB。", .96)], [("AB。", .97)]], "rectified_primary_conflict"),
])
def test_rectified_consensus_keeps_uncertainty(config, primary, direct, reason):
    payload = fallback_reader(config, primary, direct).read(np.zeros((80, 100, 3), dtype=np.uint8)).payload
    assert payload["state"] == "review"
    assert payload["selected_source"] == "primary"
    assert reason in payload["reasons"]


def test_rectified_consensus_resolves_separated_fragments(config):
    primary = [([[32, 10], [45, 10], [45, 20], [32, 20]], "80", .99),
               ([[10, 10], [28, 10], [28, 20], [10, 20]], "AB", .99)]
    payload = fallback_reader(config, primary, [[("AB80", .96)], [("AB80", .97)]]).read(
        np.zeros((80, 100, 3), dtype=np.uint8),
    ).payload
    assert payload["state"] == "readable"
    assert payload["normalized_code"] == "AB80"
    assert payload["primary_reading"]["raw_code"] is None


def test_rectified_consensus_does_not_merge_overlapping_fragments(config):
    payload = fallback_reader(config, [line("AB"), line("80")], [[("AB80", .96)], [("AB80", .97)]]).read(
        np.zeros((80, 100, 3), dtype=np.uint8),
    ).payload
    assert payload["state"] == "review"
    assert "rectified_primary_conflict" in payload["reasons"]


@pytest.mark.parametrize("partial,expected_state", [("B80", "readable"), ("A880", "review")])
def test_rectified_consensus_partial_span_requires_matching_observation(config, partial, expected_state):
    primary = [([[10, 10], [30, 10], [30, 20], [10, 20]], partial, .999)]
    payload = fallback_reader(config, primary, [[("AB80", .96)], [("AB80", .97)]]).read(
        np.zeros((80, 100, 3), dtype=np.uint8),
    ).payload
    assert payload["state"] == expected_state


def test_readable_primary_never_runs_fallback(config):
    payload = fallback_reader(config, [line()], []).read(np.zeros((80, 100, 3), dtype=np.uint8)).payload
    assert payload["state"] == "readable"
    assert "rectified_reading" not in payload


def test_rectified_consensus_cannot_confirm_same_incomplete_code(config):
    primary = [([[10, 10], [25, 10], [25, 20], [10, 20]], "AB", .999)]
    payload = fallback_reader(config, primary, [[("AB", .99)], [("AB", .99)]]).read(
        np.zeros((80, 100, 3), dtype=np.uint8),
    ).payload
    assert payload["state"] == "review"
    assert "incomplete_code_span" in payload["reasons"]
    assert "rectified_primary_conflict" in payload["reasons"]


def test_rectified_consensus_cannot_confirm_narrow_fragment_union(config):
    primary = [([[10, 10], [15, 10], [15, 20], [10, 20]], "A", .999),
               ([[18, 10], [23, 10], [23, 20], [18, 20]], "B", .999)]
    payload = fallback_reader(config, primary, [[("AB", .99)], [("AB", .99)]]).read(
        np.zeros((80, 100, 3), dtype=np.uint8),
    ).payload
    assert payload["state"] == "review"
    assert "ambiguous_code_lines" in payload["reasons"]
    assert "rectified_primary_conflict" in payload["reasons"]


@pytest.mark.parametrize("update", [
    {"line_quad_xy": ((0, 0), (60, 0), (60, 20), (0, 20))},
    {"line_quad_xy": ((1, 1), (30, 20), (30, 1), (1, 20))},
    {"line_quad_xy": ((1, 1), (1, 20), (30, 20), (30, 1))},
    {"line_quad_xy": ((1, 1), (1, 1), (1, 1), (1, 1))},
    {"line_quad_xy": ((1, 1), (30, 1), (10, 10), (30, 20))},
    {"line_quad_xy": ((1, 1), (30, 1), (30, 20))},
    {"line_quad_xy": ((1, 1), (30, 1), (30, float("nan")), (1, 20))},
    {"line_image_size": (1, 104)}, {"consensus_min_score": float("nan")},
])
def test_invalid_rectified_config_rejected(config, update):
    with pytest.raises(ValueError):
        replace(config, **update)


def test_public_api_preserves_old_imports():
    from bmw_inspection.checks import StampReader as PublicReader
    from bmw_inspection.checks import StampReaderConfig as PublicConfig
    from bmw_inspection.checks import StampReadResult
    from bmw_inspection.checks.stamp_reader import StampReadResult as OldResult

    assert (PublicReader, PublicConfig, StampReadResult) == (StampReader, StampReaderConfig, OldResult)
    code = (
        "import sys; from bmw_inspection.checks import StampReader, StampReaderConfig, StampReadResult; "
        "assert 'rapidocr_onnxruntime' not in sys.modules; assert 'onnxruntime' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("field", ["intra_op_num_threads", "inter_op_num_threads"])
@pytest.mark.parametrize("value", [0, -1, True, 2.5, "4", None])
def test_thread_settings_reject_invalid_values(config, field, value):
    with pytest.raises(ValueError, match=field):
        replace(config, **{field: value})


def test_initialize_loads_once_without_inference(config, monkeypatch):
    from types import SimpleNamespace

    calls = []
    backend_calls = []

    def backend(image):
        backend_calls.append(image.shape)
        return [line()], None

    def build(**kwargs):
        calls.append(kwargs)
        return backend

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(disable_telemetry_events=lambda: None))
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=build))
    monkeypatch.setattr("bmw_inspection.checks.stamp_reader.version", lambda name: "test")
    subject = StampReader(replace(config, intra_op_num_threads=6, inter_op_num_threads=3))
    subject.initialize()
    subject.initialize()
    assert backend_calls == []
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    result = subject.read(image)
    subject.read(image)
    assert len(backend_calls) == 2
    assert calls == [{"intra_op_num_threads": 6, "inter_op_num_threads": 3, "text_score": 0.0}]
    assert result.payload["backend"]["intra_op_num_threads"] == 6
    assert result.payload["backend"]["inter_op_num_threads"] == 3
    assert (config.intra_op_num_threads, config.inter_op_num_threads) == (4, 2)


def test_result_properties_and_independent_json_snapshot(config, tmp_path):
    subject = reader(config, [line(score=0.4)])
    subject.initialize()
    result = subject.read(np.zeros((80, 100, 3), dtype=np.uint8))
    assert result.normalized_code == result.raw_code == "5A9D6B3.02"
    assert result.state == "review"
    assert result.code_score == 0.4
    assert result.reasons == ("low_score",)
    with pytest.raises(AttributeError):
        result.state = "readable"
    snapshot = result.to_dict()
    assert snapshot == json.loads(json.dumps(result.payload))
    snapshot["lines"][0]["text"] = "changed"
    snapshot["reasons"].append("changed")
    assert result.payload["lines"][0]["text"] != "changed"
    assert result.reasons == ("low_score",)
    result.save(tmp_path / "evidence", source="example.png")
    saved = json.loads((tmp_path / "evidence/result.json").read_text())
    assert saved == {**result.to_dict(), "source": "example.png"}
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "evidence")


@pytest.mark.parametrize("context", [
    {"capture_id": ""}, {"capture_id": " "}, {"capture_id": 4},
    {"inspection_id": ""}, {"inspection_id": "\t"}, {"inspection_id": False},
    {"source_kind": "hdr"}, {"source_kind": None},
])
def test_invalid_context_rejected_before_backend(config, context):
    def backend(image):
        pytest.fail("Invalid context must not run OCR")

    with pytest.raises(ValueError):
        StampReader(config, backend).read(np.zeros((80, 100, 3), dtype=np.uint8), **context)


def test_context_is_explicit_and_does_not_leak_between_reads(config):
    subject = reader(config, [line()])
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    tagged = subject.read(image, capture_id="capture-12", inspection_id="part-3", source_kind="fused_only")
    assert tagged.to_dict()["capture_id"] == "capture-12"
    assert tagged.to_dict()["inspection_id"] == "part-3"
    assert tagged.to_dict()["source_kind"] == "fused_only"
    untagged = subject.read(image).to_dict()
    assert untagged["capture_id"] is None
    assert untagged["inspection_id"] is None
    assert untagged["source_kind"] == "unknown"


def test_cli_forwards_capture_context(config, tmp_path, monkeypatch):
    from bmw_inspection.cli.read_stamp import main

    image_path = tmp_path / "back.png"
    cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", [
        "read-stamp", "--image", str(image_path), "--config", str(config_path), "--output", str(output),
        "--capture-id", "cap-1", "--inspection-id", "part-2", "--source-kind", "fused_only",
    ])
    monkeypatch.setattr(StampReader, "_engine", lambda self: lambda image: ([line()], None))
    assert main() == 0
    saved = json.loads((output / "result.json").read_text())
    assert (saved["capture_id"], saved["inspection_id"], saved["source_kind"]) == ("cap-1", "part-2", "fused_only")
