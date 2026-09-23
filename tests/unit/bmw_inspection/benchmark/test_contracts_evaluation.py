"""Leakage and denominator safeguards for the offline development comparison."""
import copy
import pytest

from bmw_inspection.benchmark.contracts import validate_manifest, validate_plan
from bmw_inspection.benchmark.evaluation import summarize_predictions


def rows():
    return [{'image_id': k, 'split': 'dev', 'ground_truth': truth, 'requirement_ids': req,
             'physical_part_id': None, 'lineage_id': None, 'reference_fit': k == 'reference'}
            for k, truth, req in [('reference','normal',[]), ('normal001','normal',[]), ('normal003','unknown',[]), ('defect002','defect',[5])]]


def plan():
    return {'stage':'E2','evaluation_track':'development_only','algorithms':['baseline_efficientad','R01_contour'],
            'max_new_candidate_runs':6,'allow_training':False,'locked_test_access':False}


@pytest.mark.parametrize('change', [{'stage':'E4'}, {'allow_training':True}, {'locked_test_access':True},
                                  {'algorithms':['PatchCore']}, {'new_unsupervised_models':['PaDiM']},
                                  {'max_new_candidate_runs':7}, {'max_new_candidate_runs':False}])
def test_disallowed_plan(change):
    p=plan();p.update(change)
    with pytest.raises(ValueError):validate_plan(p)


def test_plan_budget_counts_new_not_baselines():
    p=plan();p['max_new_candidate_runs']=1
    assert validate_plan(p)==p
    p['algorithms'].append('R02a_morph_lines')
    with pytest.raises(ValueError):validate_plan(p)


def test_missing_identity_dev_only_and_fit_flag():
    out=validate_manifest(rows())
    assert out['block_clean_leaderboard']
    assert out['reference_fit_outside_train']==['reference']
    with pytest.raises(ValueError):validate_manifest(rows(),'clean_split')


@pytest.mark.parametrize('field',['lineage_id','physical_part_id'])
def test_cross_split_identity_rejected(field):
    r=rows();r[0]['split']='train';r[0][field]=r[1][field]='same-part'
    with pytest.raises(ValueError,match='crosses splits'):validate_manifest(r)


def test_locked_test_never_enters_evaluation():
    r=rows();r[-1]['split']='locked_test'
    with pytest.raises(ValueError,match='locked_test'):summarize_predictions(r,[],algorithm_ids=['R01_contour'])


def test_unknown_not_negative_and_missing_prediction_keeps_denominator():
    r=rows();ps=[{'algorithm_id':'a','image_id':'normal001','decision':'REVIEW','execution_status':'ok'},
                {'algorithm_id':'a','image_id':'defect002','decision':'NG','execution_status':'success'}]
    before=copy.deepcopy(ps)
    out=summarize_predictions(r,ps,algorithm_ids=['a','entirely_missing'])['algorithms']
    a=out['a']
    assert a['normal_nonrelease']=={'k':2,'n':2,'rate':1.}
    assert a['all_image_decisions']['ERROR']['k']==2
    assert a['unknown_image_decisions']['ERROR']['n']==1
    assert a['per_requirement']['5']['defect_image_decisions']['NG']=={'k':1,'n':1,'rate':1.}
    assert a['per_requirement']['1']['defect_image_decisions']['NG']['n']==0
    assert a['localization'] is None and a['independent_parts'] is None
    assert out['entirely_missing']['all_image_decisions']['ERROR']['k']==4
    assert ps==before


@pytest.mark.parametrize('patch',[{'raw_score':float('nan')},{'execution_status':'OOM'},{'decision':'UNKNOWN'}])
def test_invalid_predictions_retained_as_error(patch):
    p={'algorithm_id':'a','image_id':'defect002','decision':'NG','execution_status':'ok'};p.update(patch)
    out=summarize_predictions(rows(),[p])['algorithms']['a']
    assert out['per_requirement']['5']['defect_image_decisions']['ERROR']['k']==1


def test_duplicates_are_not_silently_overwritten():
    p={'algorithm_id':'a','image_id':'defect002','decision':'NG','execution_status':'ok'}
    with pytest.raises(ValueError):summarize_predictions(rows(),[p,p])
