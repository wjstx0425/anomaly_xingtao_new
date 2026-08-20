# ZS32 Fusion Config Memory

- `zs32_eight_view_24group_template_0727_v12_bundle_source.json` is a commissioning-only
  v12 source declaration. It keeps the v11 ROI, PatchCore summary, YOLO weights, and YOLO
  image size unchanged; only the bundle/profile identities and Template model directory differ.
- `zs32_right_eight_view_24_group_commissioning_template_0727_v12.json` remains an unbound
  24-group profile (`expected_versions=[]`) until Stage37 publishes hash-bound assets.
- Both `front_secondary` and `back_secondary` require Template, view-specific PatchCore, and YOLO.
- Published v12 candidate chain:
  `results/zs32_runtime_assets_eight_view_template_0727_v12` ->
  `results/zs32_24group_template_0727_v12_commissioning` ->
  `results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json`.
- The bundle keeps all v11 non-Template bytes and remains commissioning-only/non-production. The user reviewed the
  final-test visualization, accepted 4/16 normal false rejects, and authorized Stage35/Dashboard to default to v12.
