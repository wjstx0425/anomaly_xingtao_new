"""Linux-only exit-status contracts for the package-driven inspect CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zs32_inspection.cli import inspect as inspect_cli
from zs32_inspection.capture.errors import CaptureDataIntegrityError
from zs32_inspection.domain.decisions import InspectionDecision, InspectionStatus
from zs32_inspection.domain.identity import Hand


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


@pytest.fixture(autouse=True)
def _verified_runtime_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep inspect CLI unit tests independent from the checkout running pytest."""
    monkeypatch.setattr(inspect_cli, "verify_runtime_code_identity", lambda _payload: object())
    monkeypatch.setattr(
        inspect_cli,
        "_verify_release_runtime_environment",
        lambda _release, *, device: object(),
    )


def _decision(status: InspectionStatus) -> InspectionDecision:
    released = status if status in {
        InspectionStatus.OK,
        InspectionStatus.NG_TEMPLATE,
        InspectionStatus.NG_ANOMALY,
        InspectionStatus.NG_YOLO,
    } else None
    evidence = status if status in {
        InspectionStatus.NG_TEMPLATE,
        InspectionStatus.NG_ANOMALY,
        InspectionStatus.NG_YOLO,
    } else None
    return InspectionDecision(
        inspection_id="inspection-cli-v1",
        evidence_status=evidence,
        inspection_status=status,
        review_status="operator_review" if status is InspectionStatus.REVIEW else None,
        released_status=released,
        reason_codes=(f"status:{status.value}",),
    )


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [
        (InspectionStatus.OK, 0),
        (InspectionStatus.NG_TEMPLATE, 0),
        (InspectionStatus.NG_ANOMALY, 0),
        (InspectionStatus.NG_YOLO, 0),
        (InspectionStatus.REVIEW, 0),
        (InspectionStatus.RETAKE, 2),
        (InspectionStatus.INVALID_CAPTURE, 2),
        (InspectionStatus.SYSTEM_ERROR, 2),
    ],
)
def test_run_exit_code_distinguishes_completed_decisions_from_operational_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: InspectionStatus,
    expected_exit: int,
) -> None:
    topology = object()
    contract = SimpleNamespace(topology=topology, allowed_hands=(Hand.RIGHT,))
    release = SimpleNamespace(
        manifest=SimpleNamespace(release_id="release-cli-v1"),
        contract=contract,
        files=SimpleNamespace(read_json=lambda _path: {"code": "identity"}),
    )
    request = SimpleNamespace(capture=SimpleNamespace(part=SimpleNamespace(hand=Hand.RIGHT)))
    result_payloads: list[tuple[str, object]] = []

    class FakeRequest:
        @classmethod
        def from_capture_publication(cls, **kwargs: object) -> object:
            assert kwargs["release_id"] == "release-cli-v1"
            assert kwargs["topology"] is topology
            return request

    class FakeOrchestrator:
        def __init__(self, *, cropper: object, sink: object) -> None:
            assert cropper == "cropper"
            assert sink == "sink"

        def inspect(self, actual_request: object, runtime: object) -> object:
            assert actual_request is request
            assert runtime == "runtime"
            return SimpleNamespace(
                run=SimpleNamespace(decision=_decision(status)),
                published_path=tmp_path / "output" / "inspection-cli-v1",
            )

    monkeypatch.setattr(inspect_cli, "load_verified_deployment_release", lambda _path: release)
    monkeypatch.setattr(inspect_cli, "InspectionRequest", FakeRequest)
    monkeypatch.setattr(inspect_cli, "build_loaded_runtime", lambda *_args, **_kwargs: "runtime")
    monkeypatch.setattr(inspect_cli, "create_opencv_runtime_cropper", lambda _root: "cropper")
    monkeypatch.setattr(inspect_cli, "FilesystemInspectionSink", lambda _root: "sink")
    monkeypatch.setattr(inspect_cli, "InspectionOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(
        inspect_cli,
        "command_result",
        lambda command, payload: result_payloads.append((command, payload)),
    )

    exit_code = inspect_cli._run(
        [
            "--release",
            str(tmp_path / "release"),
            "--capture-set-root",
            str(tmp_path / "capture"),
            "--inspection-id",
            "inspection-cli-v1",
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == expected_exit
    assert result_payloads[0][0] == "zs32-inspect"
    assert result_payloads[0][1]["inspection_status"] == status.value
    assert result_payloads[0][1]["released_status"] == (
        status.value if _decision(status).released_status is not None else None
    )


def test_main_normalizes_release_or_runtime_exception_to_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[tuple[str, Exception]] = []
    monkeypatch.setattr(inspect_cli, "require_linux_nvidia", lambda: object())
    monkeypatch.setattr(
        inspect_cli,
        "_run",
        lambda _argv: (_ for _ in ()).throw(RuntimeError("broken release")),
    )
    monkeypatch.setattr(
        inspect_cli,
        "command_error",
        lambda command, error: errors.append((command, error)) or 2,
    )

    assert inspect_cli.main([]) == 2
    assert errors[0][0] == "zs32-inspect"
    assert str(errors[0][1]) == "broken release"


def test_capture_hand_not_enabled_is_never_inspected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = SimpleNamespace(
        manifest=SimpleNamespace(release_id="release-cli-v1"),
        contract=SimpleNamespace(topology=object(), allowed_hands=(Hand.RIGHT,)),
        files=SimpleNamespace(read_json=lambda _path: {"code": "identity"}),
    )
    request = SimpleNamespace(capture=SimpleNamespace(part=SimpleNamespace(hand=Hand.LEFT)))
    monkeypatch.setattr(inspect_cli, "load_verified_deployment_release", lambda _path: release)
    monkeypatch.setattr(
        inspect_cli,
        "InspectionRequest",
        SimpleNamespace(from_capture_publication=lambda **_kwargs: request),
    )
    called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("model runtime must not load for an unsupported capture hand")

    monkeypatch.setattr(inspect_cli, "build_loaded_runtime", forbidden_load)
    published: list[dict[str, object]] = []
    monkeypatch.setattr(
        inspect_cli,
        "publish_preflight_failure",
        lambda **kwargs: published.append(kwargs) or tmp_path / "output" / "inspection-cli-v1",
    )
    monkeypatch.setattr(inspect_cli, "command_result", lambda *_args, **_kwargs: None)

    exit_code = inspect_cli._run(
        [
            "--release",
            str(tmp_path / "release"),
            "--capture-set-root",
            str(tmp_path / "capture"),
            "--inspection-id",
            "inspection-cli-v1",
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 2
    assert called is False
    assert published[0]["status"] is InspectionStatus.INVALID_CAPTURE
    assert str(published[0]["reason_code"]).startswith(
        "capture_load_failed:CaptureDataIntegrityError:"
    )


@pytest.mark.parametrize(
    "error",
    [
        CaptureDataIntegrityError("missing required view"),
        CaptureDataIntegrityError("capture identity conflict"),
    ],
)
def test_explicit_capture_integrity_failure_is_invalid_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    release = SimpleNamespace(
        manifest=SimpleNamespace(release_id="release-cli-v1"),
        contract=SimpleNamespace(topology=object(), allowed_hands=(Hand.RIGHT,)),
        files=SimpleNamespace(read_json=lambda _path: {"code": "identity"}),
    )
    monkeypatch.setattr(inspect_cli, "load_verified_deployment_release", lambda _path: release)
    monkeypatch.setattr(
        inspect_cli.InspectionRequest,
        "from_capture_publication",
        lambda **_kwargs: (_ for _ in ()).throw(error),
    )
    published: list[dict[str, object]] = []
    monkeypatch.setattr(
        inspect_cli,
        "publish_preflight_failure",
        lambda **kwargs: published.append(kwargs) or tmp_path / "output" / "inspection-cli-v1",
    )
    monkeypatch.setattr(inspect_cli, "command_result", lambda *_args, **_kwargs: None)

    exit_code = inspect_cli._run(
        [
            "--release", str(tmp_path / "release"),
            "--capture-set-root", str(tmp_path / "capture"),
            "--inspection-id", "inspection-cli-v1",
            "--output-root", str(tmp_path / "output"),
        ]
    )

    assert exit_code == 2
    assert published[0]["status"] is InspectionStatus.INVALID_CAPTURE


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("capture permission denied"),
        OSError("capture filesystem unavailable"),
        json.JSONDecodeError("malformed capture JSON", "{", 1),
        RuntimeError("unexpected capture loader bug"),
    ],
)
def test_capture_load_runtime_parser_and_io_failures_are_system_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    release = SimpleNamespace(
        manifest=SimpleNamespace(release_id="release-cli-v1"),
        contract=SimpleNamespace(topology=object(), allowed_hands=(Hand.RIGHT,)),
        files=SimpleNamespace(read_json=lambda _path: {"code": "identity"}),
    )
    monkeypatch.setattr(inspect_cli, "load_verified_deployment_release", lambda _path: release)
    monkeypatch.setattr(
        inspect_cli.InspectionRequest,
        "from_capture_publication",
        lambda **_kwargs: (_ for _ in ()).throw(error),
    )
    published: list[dict[str, object]] = []
    monkeypatch.setattr(
        inspect_cli,
        "publish_preflight_failure",
        lambda **kwargs: published.append(kwargs) or tmp_path / "output" / "inspection-cli-v1",
    )
    monkeypatch.setattr(inspect_cli, "command_result", lambda *_args, **_kwargs: None)

    exit_code = inspect_cli._run(
        [
            "--release", str(tmp_path / "release"),
            "--capture-set-root", str(tmp_path / "capture"),
            "--inspection-id", "inspection-cli-v1",
            "--output-root", str(tmp_path / "output"),
        ]
    )

    assert exit_code == 2
    assert published[0]["status"] is InspectionStatus.SYSTEM_ERROR


def test_runtime_code_identity_failure_is_system_error_before_capture_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"code": "identity"}
    release = SimpleNamespace(
        manifest=SimpleNamespace(release_id="release-cli-v1"),
        contract=SimpleNamespace(topology=object(), allowed_hands=(Hand.RIGHT,)),
        files=SimpleNamespace(
            read_json=lambda path: (
                payload if path == "provenance/code_version.json" else None
            )
        ),
    )
    monkeypatch.setattr(inspect_cli, "load_verified_deployment_release", lambda _path: release)
    monkeypatch.setattr(
        inspect_cli,
        "verify_runtime_code_identity",
        lambda actual: (_ for _ in ()).throw(RuntimeError("dirty runtime checkout"))
        if actual is payload
        else None,
    )
    capture_called = False

    def forbidden_capture_load(**_kwargs: object) -> object:
        nonlocal capture_called
        capture_called = True
        raise AssertionError("capture must not load after runtime identity failure")

    monkeypatch.setattr(
        inspect_cli,
        "InspectionRequest",
        SimpleNamespace(from_capture_publication=forbidden_capture_load),
    )
    published: list[dict[str, object]] = []
    monkeypatch.setattr(
        inspect_cli,
        "publish_preflight_failure",
        lambda **kwargs: published.append(kwargs) or tmp_path / "output" / "inspection-cli-v1",
    )
    monkeypatch.setattr(inspect_cli, "command_result", lambda *_args, **_kwargs: None)

    exit_code = inspect_cli._run(
        [
            "--release", str(tmp_path / "release"),
            "--capture-set-root", str(tmp_path / "capture"),
            "--inspection-id", "inspection-cli-v1",
            "--output-root", str(tmp_path / "output"),
        ]
    )

    assert exit_code == 2
    assert capture_called is False
    assert published[0]["status"] is InspectionStatus.SYSTEM_ERROR
    assert str(published[0]["reason_code"]).startswith(
        "runtime_code_identity_failed:RuntimeError:dirty runtime checkout"
    )


def test_runtime_environment_failure_is_system_error_before_capture_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = SimpleNamespace(
        manifest=SimpleNamespace(release_id="release-cli-v1"),
        contract=SimpleNamespace(topology=object(), allowed_hands=(Hand.RIGHT,)),
        files=SimpleNamespace(read_json=lambda _path: {"code": "identity"}),
    )
    monkeypatch.setattr(inspect_cli, "load_verified_deployment_release", lambda _path: release)
    monkeypatch.setattr(
        inspect_cli,
        "_verify_release_runtime_environment",
        lambda _release, *, device: (_ for _ in ()).throw(
            RuntimeError(f"driver mismatch on device {device}")
        ),
    )
    capture_called = False

    def forbidden_capture_load(**_kwargs: object) -> object:
        nonlocal capture_called
        capture_called = True
        raise AssertionError("capture must not load after runtime environment failure")

    monkeypatch.setattr(
        inspect_cli,
        "InspectionRequest",
        SimpleNamespace(from_capture_publication=forbidden_capture_load),
    )
    published: list[dict[str, object]] = []
    monkeypatch.setattr(
        inspect_cli,
        "publish_preflight_failure",
        lambda **kwargs: published.append(kwargs) or tmp_path / "output" / "inspection-cli-v1",
    )
    monkeypatch.setattr(inspect_cli, "command_result", lambda *_args, **_kwargs: None)

    exit_code = inspect_cli._run(
        [
            "--release", str(tmp_path / "release"),
            "--capture-set-root", str(tmp_path / "capture"),
            "--inspection-id", "inspection-cli-v1",
            "--output-root", str(tmp_path / "output"),
            "--device", "0",
        ]
    )

    assert exit_code == 2
    assert capture_called is False
    assert published[0]["status"] is InspectionStatus.SYSTEM_ERROR
    assert str(published[0]["reason_code"]).startswith(
        "runtime_environment_failed:RuntimeError:driver mismatch on device 0"
    )
