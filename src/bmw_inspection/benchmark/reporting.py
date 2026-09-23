"""Static development reports with image interception and localization separated."""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from statistics import median
from urllib.parse import quote


def _csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def write_reports(output_dir, plan, manifest, predictions, metrics):
    """Write review artifacts; never infer localization from image-level NG.

    Input pixels and overlays are referenced by relative paths, not reprocessed.
    Known normal means a confirmed label; unknown is never a negative example.
    No production, part-level or system-wide inference is made from this report.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    algorithms = list(plan['algorithms'])
    raw = {(p['algorithm_id'], p['image_id']): p for p in predictions}
    decisions = {}
    for algorithm in algorithms:
        for record in metrics.get('algorithms', {}).get(algorithm, {}).get('image_records', []):
            decisions[(algorithm, record['image_id'])] = record['decision']
    def decision(algorithm, row):
        return decisions.get((algorithm, row['image_id']), 'ERROR')
    states = ('NG', 'PASS', 'REVIEW', 'ERROR')
    req_rows = []
    for algorithm in algorithms:
        for req in range(1,11):
            selected = [r for r in manifest if r['ground_truth']=='defect' and req in r.get('requirement_ids', [])]
            counts = {s: sum(decision(algorithm,r)==s for r in selected) for s in states}
            req_rows.append({'algorithm_id':algorithm,'requirement_id':req,'metric_scope':'image_interception',
                             'n':len(selected), **counts, 'localization':'unavailable',
                             'coverage':'not_validated' if selected else 'no_confirmed_requirement_samples'})
    _csv(out/'metrics_per_requirement.csv',req_rows,['algorithm_id','requirement_id','metric_scope','n',*states,'localization','coverage'])
    view_rows=[]
    for algorithm in algorithms:
        for view in sorted({r.get('view_id','unknown') for r in manifest}):
            for truth in ('normal','defect','unknown'):
                selected=[r for r in manifest if r.get('view_id','unknown')==view and r['ground_truth']==truth]
                view_rows.append({'algorithm_id':algorithm,'view_id':view,'ground_truth':truth,'n':len(selected),
                                  **{s:sum(decision(algorithm,r)==s for r in selected) for s in states},
                                  'normal_nonrelease':sum(decision(algorithm,r)!='PASS' for r in selected) if truth=='normal' else '',
                                  'unit':'image_not_independent_part'})
    _csv(out/'metrics_per_view.csv',view_rows,['algorithm_id','view_id','ground_truth','n',*states,'normal_nonrelease','unit'])
    unavailable=[{'algorithm_id':a,'metric':'unavailable','value':'','reason':'No complete instance localization ground truth or approved matching protocol'} for a in algorithms]
    _csv(out/'localization_metrics.csv',unavailable,['algorithm_id','metric','value','reason'])
    _csv(out/'metrics_per_part.csv',[{'algorithm_id':a,'metric':'unavailable','value':'','reason':'Physical identities and complete required-view coverage not established'} for a in algorithms],['algorithm_id','metric','value','reason'])
    warmup_path = out/'warmup.json'
    warmups = json.loads(warmup_path.read_text()) if warmup_path.is_file() else []
    warmup_by_algorithm = {record['algorithm_id']: record for record in warmups}
    planned_warmup = plan.get('resources', {}).get('warmup_iterations')
    latency={}
    for algorithm in algorithms:
        records=[p for p in predictions if p['algorithm_id']==algorithm]
        groups={}
        for p in records:
            values={'wall_ms':p.get('wall_ms')}
            for k,v in (p.get('timing_ms') or {}).items():
                values[k if k.endswith('_ms') else k+'_ms']=v
            for k,v in values.items():
                if not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and v>=0:
                    groups.setdefault(k,[]).append(v)
        latency[algorithm]={'cache_used':any(p.get('cache_used',False) for p in records),
                            'timing_scopes':sorted({str(p.get('timing_scope','unspecified')) for p in records}),
                            'warmup':warmup_by_algorithm.get(algorithm, {'status':'NOT_RECORDED'}),
                            'same_session_fair_comparison':False if any(p.get('cache_used',False) for p in records) else None,
                            'cache_timing_note':'Cached original contour timing is from a previous session; current wall_ms measures cache import/evidence handling, not contour inference. Do not rank against freshly executed CPU branches.' if any(p.get('cache_used',False) for p in records) else None,
                            'record_count':len(records),'execution_status_counts':{s:sum(str(p.get('execution_status'))==s for p in records) for s in sorted({str(p.get('execution_status')) for p in records})},
                            'measurements':{k:{'n':len(v),'median_ms':median(v),'p95_ms':_percentile(v,.95)} for k,v in groups.items()}}
    (out/'latency.json').write_text(json.dumps({'device':'cpu','scope':'small_sample_development_execution_including_reported_failures',
        'production_benchmark':False,'planned_warmup_iterations':planned_warmup,
        'warmup_protocol':'Per-algorithm warmup.json records execution status; successful warmup samples are excluded from predictions and these statistics. Missing/error warmup is not treated as successful.',
        'warmup_records':warmups,
        'note':'wall_ms includes wrapper work when provided; phase timings retain adapter scope and must not be treated as GPU/kernel latency',
        'algorithms':latency},ensure_ascii=False,indent=2)+'\n')
    baselines=[a for a in algorithms if a in ('baseline_template','baseline_efficientad','baseline_yolo26n')]
    candidates=[a for a in algorithms if not a.startswith('baseline_')]
    differences={name:[] for name in ('baseline_misses','new_hits','added_false_alarms','shared_misses')}
    for baseline in baselines:
        for row in manifest:
            bd=decision(baseline,row)
            common={'baseline_id':baseline,'image_id':row['image_id'],'ground_truth':row['ground_truth'],
                    'requirement_ids':','.join(map(str,row.get('requirement_ids',[]))), 'baseline_decision':bd,
                    'correct_localization':'unavailable','metric_scope':'image_interception_only'}
            if row['ground_truth']=='defect' and bd!='NG':
                differences['baseline_misses'].append(common|{'candidate_id':'','candidate_decision':'','interpretation':'not_automatic_NG; see PASS/REVIEW/ERROR separately'})
            for candidate in candidates:
                cd=decision(candidate,row)
                record=common|{'candidate_id':candidate,'candidate_decision':cd}
                if row['ground_truth']=='defect' and bd!='NG' and cd=='NG':
                    differences['new_hits'].append(record|{'interpretation':'additional_automatic_image_interception_not_localized_hit'})
                if row['ground_truth']=='normal' and bd!='NG' and cd=='NG':
                    differences['added_false_alarms'].append(record|{'interpretation':'added_normal_NG; baseline_may_already_be_REVIEW_or_ERROR'})
                if row['ground_truth']=='defect' and bd!='NG' and cd!='NG':
                    differences['shared_misses'].append(record|{'interpretation':'neither_automatic_NG; REVIEW/ERROR_are_not_defect_escape'})
    fields=['baseline_id','candidate_id','image_id','ground_truth','requirement_ids','baseline_decision','candidate_decision','correct_localization','metric_scope','interpretation']
    for name,records in differences.items():
        _csv(out/(name+'.csv'),records,fields)
    lines=['# BMW 离线算法对照实验摘要','',
           '本报告仅统计图像级拦截。未知标签不作为正常负样本；NG不等于正确定位真实缺陷。缺少完整实例标注、实体关联与必检视图，因此定位和独立工件指标不可用。',
           '', '当前结果属于开发实验，不代表干净泛化排行榜、需求完整覆盖、生产节拍或全站系统效果。', '',
           '| 算法 | 图像 NG/PASS/REVIEW/ERROR | 正常非自动放行 k/n |', '|---|---|---|']
    for algorithm in algorithms:
        counts=[sum(decision(algorithm,r)==s for r in manifest) for s in states]
        normal=[r for r in manifest if r['ground_truth']=='normal']
        lines.append(f"| {algorithm} | {'/'.join(map(str,counts))} | {sum(decision(algorithm,r)!='PASS' for r in normal)}/{len(normal)} |")
    lines+=['','第5项按已确认缺陷图像单独分层；其他缺少需求真值的项目不填造检出率。各算法覆盖对象不同，不可视为同一分类器的直接替代。',
            '', '差集分别比较每个新增分支与三个冻结基线。new_hits仅表示新增自动NG图像拦截，不表示正确定位；shared_misses明确保留PASS、REVIEW、ERROR，不能把复核和执行失败混称流出。',
            '', f'计划预热次数：{planned_warmup if planned_warmup is not None else "未记录"}。实际执行见warmup.json；成功预热样本不计入预测计时。预热失败或未记录不视为预热成功。',
            '', 'R01若复用历史轮廓结果，其算法时间来自原会话，当前wall仅包含缓存读取及证据处理，不是重新推理；不能与本次CPU实跑算法公平排名。其他计时仍须结合适配器记录的范围及小样本数量解释。',
            '', '[同图并列查看](comparison.html) · [需求指标](metrics_per_requirement.csv) · [延迟与预热记录](latency.json)']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    esc=lambda v:html.escape(str(v),quote=True)
    cards=[]
    def picture(path):
        if not path:
            return '<p>No overlay available</p>'
        # Only local relative evidence is admitted into the HTML artifact.
        p=Path(str(path))
        if p.is_absolute() or '..' in p.parts:
            return '<p>Evidence path rejected: requires relative local path</p>'
        url=quote(p.as_posix(),safe='/')
        return f'<a href="{esc(url)}"><img loading="lazy" src="{esc(url)}" alt="Evidence image"></a>'
    for row in manifest:
        panels=[f'<article><h3>Input</h3>{picture("evidence/input/"+row["image_id"]+".jpg")}</article>']
        for algorithm in algorithms:
            p=raw.get((algorithm,row['image_id']),{})
            details={k:p[k] for k in ('raw_score','threshold','observed_required_fraction','candidate_ng_count','candidate_review_count','candidate_count','error') if k in p}
            panels.append(f'<article><h3>{esc(algorithm)} · {esc(decision(algorithm,row))}</h3>{picture(p.get("evidence_overlay"))}<pre>{esc(json.dumps(details,ensure_ascii=False,indent=2))}</pre></article>')
        reqs=row.get('requirement_ids',[])
        cards.append(f'<section data-gt="{esc(row["ground_truth"])}" data-req5="{str(5 in reqs).lower()}"><h2>{esc(row["image_id"])}</h2><p>GT: {esc(row["ground_truth"])}; requirement IDs: {esc(reqs)}. Text labels only; no truth boxes.</p><div class="panels">'+''.join(panels)+'</div></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BMW development comparison</title><style>body{font:15px system-ui;margin:24px;color:#17212b;background:#f3f5f7}section{margin:24px 0;padding:16px;background:white;border:1px solid #ccd3db}.panels{display:flex;gap:14px;overflow:auto}article{flex:0 0 290px}img{width:100%;height:240px;object-fit:contain;background:#111}h3{font-size:14px}pre{white-space:pre-wrap;font-size:12px}select{padding:8px}a{color:#174d87}</style><h1>BMW image-level development comparison</h1><p>NG is image interception, not verified localization. REVIEW and ERROR are distinct. Unknown labels are not negative truth. No independent-part or production acceptance claim.</p><label>Filter <select id="filter"><option value="all">All</option><option value="req5">Requirement 5</option><option value="normal">Confirmed normal</option><option value="unknown">Unknown label</option></select></label>'''+''.join(cards)+'''<script>document.querySelector('#filter').addEventListener('change',e=>{const f=e.target.value;document.querySelectorAll('section').forEach(s=>{s.hidden=!(f==='all'||(f==='req5'?s.dataset.req5==='true':s.dataset.gt===f));});});</script></html>'''
    (out/'comparison.html').write_text(page,encoding='utf-8')
    return {'report_files':['comparison.html','summary.md','metrics_per_requirement.csv','metrics_per_view.csv','localization_metrics.csv','metrics_per_part.csv','latency.json',*[n+'.csv' for n in differences]]}
