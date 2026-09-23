"""Post-prediction label joins and image-level counts, never prediction inputs."""
from __future__ import annotations

import math

from .contracts import validate_manifest

DECISIONS = ('NG', 'PASS', 'REVIEW', 'ERROR')


def _counts(decisions):
    n = len(decisions)
    return {decision: {'k': decisions.count(decision), 'n': n,
                       'rate': decisions.count(decision)/n if n else None} for decision in DECISIONS}


def summarize_predictions(rows, predictions, *, algorithm_ids=None, evaluation_track='development_only') -> dict:
    """Count every planned image; missing/invalid model outputs become ERROR.

    Requirement strata count image interception, not localization or independent
    physical defects. Normal labels provide image-level nonrelease only; no
    absent requirement is silently treated as supervised negative truth.
    """
    flags = validate_manifest(rows, evaluation_track)
    image_ids = {r['image_id'] for r in rows}
    algorithms = list(algorithm_ids) if algorithm_ids is not None else sorted({p['algorithm_id'] for p in predictions})
    if not algorithms or any(not isinstance(a, str) or not a for a in algorithms) or len(algorithms) != len(set(algorithms)):
        raise ValueError('Explicit nonempty unique algorithm IDs required when no predictions exist')
    indexed = {}
    for p in predictions:
        key = (p.get('algorithm_id'), p.get('image_id'))
        if key[0] not in algorithms or key[1] not in image_ids or key in indexed:
            raise ValueError('Unknown or duplicate prediction identity')
        indexed[key] = p
    results = {}
    for algorithm in algorithms:
        joined = []
        for row in rows:
            p = indexed.get((algorithm, row['image_id']))
            decision = p.get('decision') if p else 'ERROR'
            reason = None if p else 'missing_prediction'
            if p:
                if decision not in DECISIONS or str(p.get('execution_status', '')).lower() not in {'ok', 'success', 'completed'}:
                    decision, reason = 'ERROR', 'invalid_or_failed_execution'
                score = p.get('raw_score')
                if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
                    decision, reason = 'ERROR', 'invalid_raw_score'
            joined.append({'image_id': row['image_id'], 'decision': decision, 'evaluation_error': reason,
                           'ground_truth': row['ground_truth'], 'requirement_ids': row.get('requirement_ids', [])})
        normal = [r['decision'] for r in joined if r['ground_truth'] == 'normal']
        strata = {}
        for req in range(1, 11):
            defect = [r['decision'] for r in joined if r['ground_truth'] == 'defect' and req in r['requirement_ids']]
            strata[str(req)] = {'defect_image_decisions': _counts(defect), 'localization': None, 'independent_parts': None,
                                'metric_scope': 'image_interception_not_instance_detection', 'requirement_coverage_confirmed': False}
        results[algorithm] = {'all_image_decisions': _counts([r['decision'] for r in joined]),
                              'normal_image_decisions': _counts(normal),
                              'normal_nonrelease': {'k': sum(v != 'PASS' for v in normal), 'n': len(normal),
                                                    'rate': sum(v != 'PASS' for v in normal)/len(normal) if normal else None},
                              'unknown_image_decisions': _counts([r['decision'] for r in joined if r['ground_truth'] == 'unknown']),
                              'per_requirement': strata, 'localization': None, 'independent_parts': None, 'image_records': joined}
    return {'evaluation_track': evaluation_track, 'manifest_flags': flags, 'algorithms': results,
            'localization': None, 'independent_parts': None, 'clean_leaderboard': False}
