# BMW independent runtime environment

This uv project isolates BMW runtime dependencies from repository development
and test extras while reusing the local `anomalib` and `bmw_inspection` source.

```bash
uv sync --project bmw_runtime

uv run --project bmw_runtime bmw-inspect \
  --config configs/bmw/experiments/bmw_eight_view_demo_left_0820_mixed_v1.json
```

Use the right profile for right-hand parts. There is no automatic side choice.

For the camera host, install Hikvision MVS separately and set the path when it
is not installed at the default location:

```bash
export BMW_MVS_SDK_PATH=/opt/MVS/Samples/64/Python/MvImport
uv run --project bmw_runtime bmw-collect --list-devices
```

All BMW JSON input paths are resolved relative to the JSON file. Copying the
repository or a complete BMW asset tree to another absolute location therefore
does not require editing paths.

## External BMW assets

GitHub contains the runtime code and small JSON configuration only. A fresh
clone is not a complete inspection station: copy the following BMW-only trees
from the accepted station backup into the same repository-relative locations:

```text
dataset/bmw_lab_prepared/bmw_left_0820_v1/
dataset/bmw_lab_prepared/bmw_right_0820_v1/
dataset/bmw_trusted_ok_reference/bmw_left_0820_train_normal_v1/
dataset/bmw_trusted_ok_reference/bmw_right_0820_train_normal_v1/
results/bmw_bright_streak_rotated_roi/
results/bmw_efficientad_manual_ignore_masks/
results/bmw_lab_one_click/
```

The active left/right profile is the asset inventory: startup stops and names
the first missing model, ROI, mask, or trusted-OK index. The prepared dataset
manifest is needed only for `--sample-id`; live cameras and `--capture-set` do
not require it. Generated inspection output remains under the profile's
`result_root`.
