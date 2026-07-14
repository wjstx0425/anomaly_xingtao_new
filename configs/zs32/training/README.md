# ZS32 Anomalib training parameters

These files are parameter templates for `zs32-train-anomaly`; they are not
approved release recipes. Training always consumes one explicitly selected
`hand/view` slot from a verified materialized Anomalib export and publishes a
new immutable candidate. It does not update a production release.
`--train-split-id` is the verified canonical dataset manifest's
`split_assignments_sha256`; arbitrary run labels are rejected as provenance.

`image_size` is `[height, width]`. The backend builds only Resize and optional
ImageNet Normalize transforms. AnomalyDINO dimensions must be divisible by 14,
otherwise its upstream patch encoder would spatially crop the resized input and
the job is rejected.

All three examples use backend schema v2. The auxiliary hashes in all three
examples are deliberate placeholders. Replace
them with hashes measured on the Linux training host. PatchCore requires a raw
`state_dict` for the exact Anomalib feature extractor selected by `backbone` and
`layers`; the backend constructs PatchCore with `pre_trained=false`, strictly
loads that local file, and never permits an implicit download/cache choice. Its
scope is exactly `anomalib_timm_feature_extractor`, meaning the serialized keys
must come from the Anomalib `TimmFeatureExtractor` wrapper rather than the inner
timm backbone. `backbone_weights_asset_id` is an opaque audited ID, not a path.
EfficientAD additionally binds the complete ImageNette auxiliary tree;
AnomalyDINO binds the cached pretrained encoder. Missing or changed auxiliary
assets fail before training and are rechecked after training.
The ImageNette directory is training-only: published EfficientAD metadata keeps
its tree digest and the teacher-weight digest, but removes the Linux host path.

On Linux, compute the EfficientAD tree digest with:

```bash
uv run python - <<'PY'
from pathlib import Path
from zs32_inspection.models import sha256_auxiliary_tree

print(sha256_auxiliary_tree(Path("/DATA/ljl/zs32_auxiliary/imagenette2")))
PY
```

Use `sha256sum` for the PatchCore state_dict, pretrained teacher, or DINOv2
encoder file. The PatchCore file must be exported on Linux as the raw
`model.model.feature_extractor.state_dict()` of the exact architecture; wrapped
checkpoints, inner-backbone-only keys, and partial/non-strict key sets are rejected.
The training execution receipt binds both the canonical raw parameter digest and
the verified backbone file digest. Published metadata removes the Linux host path;
the self-contained checkpoint is the only runtime model file.

`template_opencv.example.json` is consumed by `zs32-train-template`. It selects
deterministic train/normal reference crops only; the binary template threshold
is still fitted later by `zs32-calibrate` and is never embedded in training.
