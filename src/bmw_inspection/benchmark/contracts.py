"""Small offline experiment contracts; no model or image access."""
from __future__ import annotations

import copy

ALGORITHMS = frozenset({'baseline_system', 'baseline_template', 'baseline_efficientad', 'baseline_yolo26n',
                       'R01_contour', 'R02a_morph_lines', 'R02b_multiscale_lines', 'R03_reference_residual',
                       'R04_color_regions', 'R05_texture', 'D01_yolo_input_ablation', 'D01_yolo_high',
                       'D01_yolo_tiles', 'D02_detector', 'S01_segmentation', 'S02_segmentation', 'S03_segmentation'})
TRACKS = {'development_only', 'deployment_replay', 'clean_split'}


def validate_plan(plan: dict) -> dict:
    """Validate a resolved E0-E2 offline plan and return a defensive copy."""
    p = copy.deepcopy(plan)
    if p.get('stage') not in {'E0', 'E1', 'E2'}:
        raise ValueError('Only E0-E2 stages are supported')
    if p.get('evaluation_track') not in TRACKS:
        raise ValueError('Invalid evaluation_track')
    algorithms = p.get('algorithms')
    if not isinstance(algorithms, list) or not algorithms or any(not isinstance(a, str) or a not in ALGORITHMS for a in algorithms):
        raise ValueError('Unregistered algorithm; EfficientAD is the only allowed normal-only deep model')
    if len(set(algorithms)) != len(algorithms):
        raise ValueError('Duplicate algorithm IDs')
    budget = p.get('max_new_candidate_runs', 6)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 6:
        raise ValueError('New candidate budget must be an integer from 0 to 6')
    if sum(not a.startswith('baseline_') for a in algorithms) > budget:
        raise ValueError('New candidate budget exceeded')
    for field in ('allow_training', 'locked_test_access', 'allow_efficientad_retraining', 'allow_new_unsupervised_deep_training'):
        if p.get(field, False) is not False:
            raise ValueError(f'{field} is not authorized')
    for field in ('normal_only_deep_model_allowlist', 'new_unsupervised_models'):
        if field in p and (not isinstance(p[field], list) or any(model != 'EfficientAD' for model in p[field])):
            raise ValueError('EfficientAD is the only allowed normal-only deep model')
    return p


def validate_manifest(rows: list[dict], evaluation_track='development_only') -> dict:
    """Check image identities and split leakage without reading image pixels."""
    if evaluation_track not in TRACKS:
        raise ValueError('Invalid evaluation_track')
    if not rows:
        raise ValueError('Manifest is empty')
    seen = set()
    groups = {}
    missing = []
    fitted_dev = []
    for row in rows:
        identifier = row.get('image_id')
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError('Missing or duplicate image_id')
        seen.add(identifier)
        split = row.get('split')
        if split not in {'train', 'dev', 'locked_test'}:
            raise ValueError('Invalid split')
        # This runner is intentionally limited to E0-E2, including evaluation.
        if split == 'locked_test':
            raise ValueError('locked_test access is blocked during E0-E2')
        truth = row.get('ground_truth')
        if truth not in {'normal', 'defect', 'unknown'}:
            raise ValueError('Invalid ground_truth')
        req = row.get('requirement_ids', [])
        if not isinstance(req, list) or any(isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= 10 for i in req) or len(set(req)) != len(req):
            raise ValueError('requirement_ids must be unique integers 1..10')
        if truth == 'unknown' and req:
            raise ValueError('Unknown labels cannot imply known requirement truth')
        for field in ('physical_part_id', 'lineage_id'):
            value = row.get(field)
            if value is None or value == '':
                missing.append({'image_id': identifier, 'field': field})
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Invalid identity value')
            key = (field, value)
            if key in groups and groups[key] != split:
                raise ValueError(f'{field} crosses splits: {value}')
            groups[key] = split
        fit = row.get('reference_fit', False)
        if not isinstance(fit, bool):
            raise ValueError('reference_fit must be boolean')
        if fit and truth != 'normal':
            raise ValueError('Reference fitting requires confirmed normal label')
        if fit and split != 'train':
            fitted_dev.append(identifier)
    blockers = []
    if missing:
        blockers.append('missing_physical_or_lineage_identity')
    if fitted_dev:
        blockers.append('reference_fit_outside_train')
    if evaluation_track != 'development_only' and blockers:
        raise ValueError('Manifest lacks clean identity/fit separation: ' + ','.join(blockers))
    return {'evaluation_track': evaluation_track, 'image_count': len(rows), 'block_clean_leaderboard': bool(blockers) or evaluation_track != 'clean_split',
            'blockers': blockers, 'missing_identity': missing, 'reference_fit_outside_train': fitted_dev,
            'unknown_is_negative': False, 'locked_test_access': False}
