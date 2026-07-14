"""Topology-driven capture business services for ZS32."""

from .contracts import (
    CameraBinding,
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    CaptureResult,
    CaptureRoundPlan,
    RoundConfirmation,
)
from .gates import CaptureGateResult
from .gate_policy import (
    CaptureGatePolicy,
    CaptureGateProvenance,
    HandCaptureGatePolicy,
    load_capture_gate_policy_bytes,
)
from .gate_publication import (
    VerifiedCaptureGatePublication,
    load_verified_capture_gate_publication,
)
from .errors import (
    CaptureDataIntegrityError,
    CaptureRetakeRequired,
    CaptureSystemError,
    InvalidCaptureError,
)
from .hikvision import (
    DeviceDescription,
    HikvisionCameraAdapter,
    HikvisionCaptureConfig,
    HikvisionCaptureError,
    HikvisionSdkUnavailableError,
    select_devices_by_serial,
)
from .opencv_quality import OpenCvQualityGate, QualityGateProfile, QualityViewThresholds
from .opencv_registration import (
    OpenCvRegistrationGate,
    RegistrationGateProfile,
    RegistrationReference,
    RegistrationThresholds,
    RegistrationViewSpec,
)
from .quality import QualityGate
from .round_coordinator import ConsoleRoundCoordinator
from .reader import discover_complete_capture_sets, load_capture_bundle
from .registration import RegistrationGate
from .service import (
    CaptureService,
    CaptureSource,
    IncompleteCaptureError,
    PartialRoundCaptureError,
    RoundCoordinator,
)
from .storage import AtomicCaptureStore

__all__ = [
    "AtomicCaptureStore",
    "CameraBinding",
    "CaptureFrame",
    "CaptureDataIntegrityError",
    "CaptureGateResult",
    "CaptureGatePolicy",
    "CaptureGateProvenance",
    "CapturePlan",
    "CaptureRequest",
    "CaptureResult",
    "CaptureRoundPlan",
    "RoundConfirmation",
    "CaptureService",
    "CaptureSource",
    "CaptureRetakeRequired",
    "CaptureSystemError",
    "ConsoleRoundCoordinator",
    "DeviceDescription",
    "HikvisionCameraAdapter",
    "HikvisionCaptureConfig",
    "HikvisionCaptureError",
    "HikvisionSdkUnavailableError",
    "HandCaptureGatePolicy",
    "IncompleteCaptureError",
    "InvalidCaptureError",
    "OpenCvQualityGate",
    "OpenCvRegistrationGate",
    "PartialRoundCaptureError",
    "QualityGate",
    "QualityGateProfile",
    "QualityViewThresholds",
    "RegistrationGateProfile",
    "RegistrationReference",
    "RegistrationGate",
    "RegistrationThresholds",
    "RegistrationViewSpec",
    "RoundCoordinator",
    "VerifiedCaptureGatePublication",
    "discover_complete_capture_sets",
    "load_capture_bundle",
    "load_capture_gate_policy_bytes",
    "load_verified_capture_gate_publication",
    "select_devices_by_serial",
]
