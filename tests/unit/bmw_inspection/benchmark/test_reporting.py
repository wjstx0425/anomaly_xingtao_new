"""Report labels preserve interception, unknowns and failure denominators."""
import csv
import json

from bmw_inspection.benchmark.evaluation import summarize_predictions
from bmw_inspection.benchmark.reporting import write_reports


def test_reports_do_not_turn_review_into_hit_or_unknown_into_normal(tmp_path):
    algorithms=['baseline_template','R01_contour','R02a_morph_lines']
    rows=[{'image_id':i,'ground_truth':gt,'requirement_ids':req,'split':'dev'} for i,gt,req in
          [('n','normal',[]),('u','unknown',[]),('d','defect',[5])]]
    predictions=[]
    for algorithm in algorithms:
        for row in rows:
            decision='NG' if algorithm=='R01_contour' else 'REVIEW'
            predictions.append({'image_id':row['image_id'],'algorithm_id':algorithm,'decision':decision,
                                'execution_status':'SUCCESS','raw_score':2.,'wall_ms':10.,'timing_ms':{'inference':3.},
                                'evidence_overlay':'evidence/overlay.jpg'})
    metrics=summarize_predictions(rows,predictions,algorithm_ids=algorithms)
    result=write_reports(tmp_path,{'algorithms':algorithms},rows,predictions,metrics)
    assert all((tmp_path/f).exists() for f in result['report_files'])
    def read(name):return list(csv.DictReader(open(tmp_path/name)))
    assert len(read('new_hits.csv'))==1
    assert read('new_hits.csv')[0]['candidate_id']=='R01_contour'
    assert len(read('added_false_alarms.csv'))==1
    assert read('added_false_alarms.csv')[0]['image_id']=='n'
    assert read('shared_misses.csv')[0]['candidate_decision']=='REVIEW'
    assert all(r['correct_localization']=='unavailable' for r in read('new_hits.csv'))
    req1=[r for r in read('metrics_per_requirement.csv') if r['requirement_id']=='1']
    assert all(r['n']=='0' for r in req1)
    latency=json.load(open(tmp_path/'latency.json'))
    assert latency['algorithms']['R01_contour']['measurements']['wall_ms']['median_ms']==10.
    page=(tmp_path/'comparison.html').read_text()
    assert 'data-gt="unknown"' in page and 'data-req5="true"' in page
