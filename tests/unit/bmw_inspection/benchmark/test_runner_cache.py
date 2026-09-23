"""Regression checks for reusing source-bound, parameter-bound contour evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('bmw_comparison_runner', ROOT / 'tools/bmw/run_algorithm_comparison.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def cache(tmp_path):
    run = tmp_path / 'run'
    reference = run / 'reference_draft'
    sample = run / 'samples' / 'one'
    reference.mkdir(parents=True)
    sample.mkdir(parents=True)
    for name in ('reference.png', 'reference_foreground_mask.png', 'contour.npz'):
        (reference / name).write_bytes(name.encode())
    base = {'comparison': {'outward_px': 4}, 'extraction': {'edge_selection_mode': 'continuous_v3'}, 'reference_version': 'base'}
    actual = base | {'reference_version': 'repaired_fold_mask'}
    recipe = actual | {'asset_sha256': {p.name: runner.digest(p) for p in reference.iterdir()}}
    (reference / 'recipe.json').write_text(json.dumps(recipe))
    result = {'image_sha256': 'expected', 'parameter_snapshot': actual, 'status': 'REVIEW',
              'test_to_reference': {'max_px': 0.0}, 'reason_codes': ['unknown'], 'events': [],
              'observed_required_fraction': .8, 'unobserved_required_length_px': 20,
              'full_perimeter_pass': False, 'timing_ms': {'algorithm_total': 100}}
    (sample / 'result.json').write_text(json.dumps(result))
    for name in ('measured_outline_overlay.png', 'contour_samples.csv'):
        (sample / name).write_bytes(b'evidence')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(base))
    output = tmp_path / 'out'
    output.mkdir()
    return {'contour_cached_runs': [str(run)], 'contour_config': str(config)}, {'image_id': 'one', 'sha256': 'expected'}, output, reference, sample


def test_cache_allows_recorded_teaching_version_but_not_inference_change(cache):
    plan, row, output, reference, sample = cache
    result = runner.contour_cache(plan, row, output)
    assert result['decision'] == 'REVIEW' and result['cache_used']
    assert result['teaching_metadata_differences']['reference_version']['resolved_reference_recipe'] == 'repaired_fold_mask'
    config = Path(plan['contour_config'])
    data = json.loads(config.read_text())
    data['comparison']['outward_px'] = 8
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='parameter mismatch'):
        runner.contour_cache(plan, row, output)


def test_cache_rejects_changed_source_and_reference_bytes(cache):
    plan, row, output, reference, sample = cache
    with pytest.raises(ValueError, match='source SHA mismatch'):
        runner.contour_cache(plan, row | {'sha256': 'other'}, output)
    (reference / 'contour.npz').write_bytes(b'different reference')
    with pytest.raises(ValueError, match='asset hash mismatch'):
        runner.contour_cache(plan, row, output)


def test_cache_rejects_result_metadata_unbound_to_actual_recipe(cache):
    plan, row, output, reference, sample = cache
    path = sample / 'result.json'
    data = json.loads(path.read_text())
    data['parameter_snapshot']['reference_version'] = 'unbound'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='actual reference recipe'):
        runner.contour_cache(plan, row, output)
