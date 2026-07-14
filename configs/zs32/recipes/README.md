# ZS32 recipes

`zs32_right_patchcore_training_v1.json` is a **training profile**, not a
deployment recipe.  Its zero SHA-256 asset references are deliberate unbound
sentinels and its threshold arrays are empty.  It therefore must never be
used for training, compiled, or assembled as a release. In particular,
`capture_gate_policy.sha256` must be replaced with the real digest emitted by
`zs32-publish-gate-policy` before the first model is trained; otherwise later
binding would change `recipe_sha256` and invalidate all training provenance.
The gate policy digest also transitively binds the exact Hikvision acquisition
asset bytes through its schema-v2 `acquisition_config` ArtifactRef.

Create a Linux-local `.local.json` copy with the real policy artifact id,
version, and SHA256. All training/register/calibration commands consume that
bound local recipe. Zero is a visible sentinel, never a runnable default.

On the Linux + NVIDIA host, train/import every frozen asset and pass their
aggregate `zs32.candidate_registration` document to
`zs32-register-candidate`.  That command derives the candidate digest and
writes `bound_recipe.json` with real model/template/YOLO hashes while
preserving the stable `recipe_sha256`.  `zs32-finalize-recipe` later inserts
only thresholds tied to the immutable calibration artifact.

Left-hand deployment remains forbidden until the left ROI is measured,
reviewed, and changed from `pending` to `ready`; do not mirror the right ROI.
