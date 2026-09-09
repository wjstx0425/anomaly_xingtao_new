# BMW pipeline

The product-specific pipeline is now BMW-only. The supported camera contract is
four serial-bound cameras, two operator rounds, and eight semantic views.

## Runtime

Use one explicit side profile; there is deliberately no implicit left/right
default:

```bash
uv run bmw-inspect \
  --config configs/bmw/experiments/bmw_eight_view_demo_left_0820_mixed_v1.json

uv run bmw-inspect \
  --config configs/bmw/experiments/bmw_eight_view_demo_right_0820_mixed_v1.json
```

Offline validation:

```bash
uv run bmw-inspect \
  --config configs/bmw/experiments/bmw_eight_view_demo_left_0820_mixed_v1.json \
  --capture-set /path/to/eight-view-part \
  --no-gui
```

The capture-set directory must contain one image for each of `front`,
`front_left`, `front_right`, `front_secondary`, `back`, `back_left`,
`back_right`, and `back_secondary`.

## Collection

```bash
BMW_MVS_SDK_PATH=/opt/MVS/Samples/64/Python/MvImport \
uv run bmw-collect --list-devices

uv run bmw-collect \
  --hand right \
  --label normal \
  --part-id bmw_right_normal \
  --group-count 1
```

The collector saves fused, short-exposure, and long-exposure images. The CSV
manifest records relative image paths and does not create hashes or deployment
receipts.

The files under `pipeline/bmw_lab_*` are compatibility wrappers. Packaged entry
points live under `bmw_inspection.cli`.

The retired numbered pipeline, C789/FX11 tools, and their `capture_data`
implementations were removed during the BMW cleanup. Their source remains in
Git history at commit `56f429c462536f9a45a6a247845c371448e6aa3c`.
For environment setup and the required external assets, see
[the BMW runtime guide](../bmw_runtime/README.md).
