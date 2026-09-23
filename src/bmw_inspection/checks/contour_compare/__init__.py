"""Independent development-only full-perimeter inspection API."""
from .comparison import compare_outline
from .extraction import extract_full_outline
from .reference import load_reference, save_reference, teach_reference
from .registration import estimate_rigid_pose
from .api import FrontContourInspector, FrontContourResult
from .contracts import ContourInputError

__all__ = ["compare_outline", "extract_full_outline", "load_reference", "save_reference", "teach_reference", "estimate_rigid_pose"]
__all__ += ["FrontContourInspector", "FrontContourResult", "ContourInputError"]
