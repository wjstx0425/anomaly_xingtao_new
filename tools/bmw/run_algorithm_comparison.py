"""Replay the bounded BMW E0-E2 development plan using local frozen assets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.benchmark.contracts import validate_manifest, validate_plan
from bmw_inspection.benchmark.evaluation import summarize_predictions
from bmw_inspection.benchmark.frozen_baseline import FrozenBaselineAdapter
from bmw_inspection.benchmark.line_rule import predict_lines

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_pixels(row):
    path = resolve(row['image_path'])
    if digest(path) != row['sha256']:
        raise ValueError('Input SHA differs from frozen development manifest')
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (row['height'], row['width'], 3) or image.dtype != np.uint8:
        raise ValueError('Unexpected input shape, channels or bit depth')
    return image


def contour_cache(plan, row, folder):
    matches = []
    for run in plan['contour_cached_runs']:
        path = resolve(run) / 'samples' / row['image_id'] / 'result.json'
        if path.is_file():
            matches.append(path)
    if len(matches) != 1:
        raise ValueError('Exactly one contour cache entry is required')
    path = matches[0]
    result = json.loads(path.read_text())
    if result['image_sha256'] != row['sha256']:
        raise ValueError('Contour cache source SHA mismatch')
    config = json.loads(resolve(plan['contour_config']).read_text())
    parameters = result['parameter_snapshot']
    teaching_fields = {'reference_bundle', 'reference_version', 'reference_identity',
                       'reference_normal_unconfirmed_sample_ids', 'parameter_provenance'}
    for key, value in config.items():
        if key not in teaching_fields and parameters.get(key) != value:
            raise ValueError('Contour cache parameter mismatch: ' + key)
    source_dir = path.parents[2] / 'reference_draft'
    recipe = json.loads((source_dir / 'recipe.json').read_text())
    for name, sha in recipe.pop('asset_sha256').items():
        if digest(source_dir / name) != sha:
            raise ValueError('Cached reference asset hash mismatch: ' + name)
    if parameters != recipe:
        raise ValueError('Cached result differs from actual reference recipe')
    shutil.copy2(path, folder / 'contour_result.json')
    source_overlay = path.parent / 'measured_outline_overlay.png'
    shutil.copy2(source_overlay, folder / 'overlay.png')
    source_dir = path.parents[2] / 'reference_draft'
    source_files = [path, source_overlay, path.parent / 'contour_samples.csv', source_dir / 'recipe.json', source_dir / 'contour.npz']
    return {
        'decision': result['status'], 'execution_status': 'SUCCESS',
        'raw_score': result['test_to_reference']['max_px'], 'score_units': 'pixel_distance',
        'threshold': config['comparison'], 'reason': result['reason_codes'],
        'candidate_ng_count': sum(e['status'] == 'NG' for e in result['events']),
        'candidate_review_count': sum(e['status'] == 'REVIEW' for e in result['events']),
        'observed_required_fraction': result['observed_required_fraction'],
        'unobserved_required_length_px': result['unobserved_required_length_px'],
        'full_perimeter_pass': result['full_perimeter_pass'],
        'timing_ms': result['timing_ms'], 'timing_scope': 'cached_actual_contour_replay; not comparable same-session timing',
        'cache_provenance': {str(p): digest(p) for p in source_files}, 'cache_used': True,
        'teaching_metadata_differences': {key: {'base_config': config.get(key), 'resolved_reference_recipe': parameters.get(key)} for key in teaching_fields if config.get(key) != parameters.get(key)},
    }


def predict_one(plan, row, image, algorithm, adapter, folder):
    if algorithm == 'R01_contour':
        return contour_cache(plan, row, folder)
    if algorithm == 'R02a_morph_lines':
        result = predict_lines(image, plan['line_config'])
        arrays = {key: result.pop(key) for key in ('response_dark', 'response_bright', 'candidate_mask')}
        np.savez_compressed(folder / 'line_responses.npz', **arrays)
        overlay = image.copy()
        candidate = arrays['candidate_mask'] > 0
        overlay[candidate] = (0.4 * overlay[candidate] + 0.6 * np.array([0, 190, 255])).astype(np.uint8)
        for box in result['boxes']:
            x1, y1, x2, y2 = box['xyxy']
            cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 200, 255), 1)
        cv2.imwrite(str(folder / 'overlay.png'), overlay)
        cv2.imwrite(str(folder / 'candidate_mask.png'), arrays['candidate_mask'])
        result['candidate_count'] = len(result['boxes'])
        result['timing_ms'] = {phase: result.pop(phase + '_ms') for phase in ('preprocess', 'inference', 'postprocess')}
        result['timing_scope'] = 'CPU native input ROI morphology plus candidate measurements; wall includes evidence export'
        return result
    result = adapter.predict(image, algorithm, view='front')
    overlay = result.pop('overlay')
    anomaly_map = result.pop('raw_map')
    if overlay is not None:
        full = image.copy()
        x1, y1, x2, y2 = result['mapping']['roi_xyxy']
        if result['mapping']['overlay_coordinate_system'] == 'template_aligned_target_pixels':
            cv2.imwrite(str(folder / 'raw_overlay.png'), overlay)
            matrix = np.asarray(result['mapping']['overlay_to_full_3x3'], dtype=np.float64)
            left, top, right, bottom = result['mapping']['target_content_xyxy']
            target_valid = np.zeros(overlay.shape[:2], np.uint8)
            target_valid[top:bottom, left:right] = 255
            size = (image.shape[1], image.shape[0])
            remapped = cv2.warpPerspective(overlay, matrix, size, flags=cv2.INTER_LINEAR)
            valid = cv2.warpPerspective(target_valid, matrix, size, flags=cv2.INTER_NEAREST) > 0
            roi_valid = np.zeros(image.shape[:2], bool)
            roi_valid[y1:y2, x1:x2] = True
            valid &= roi_valid
            full[valid] = remapped[valid]
            result['overlay_note'] = 'Template aligned target mapped back for display only; reflected padding excluded; raw 512 overlay retained'
        else:
            if overlay.shape != full[y1:y2, x1:x2].shape:
                raise ValueError('Frozen runtime overlay coordinate shape mismatch')
            full[y1:y2, x1:x2] = overlay
        cv2.imwrite(str(folder / 'overlay.png'), full)
    if anomaly_map is not None:
        np.save(folder / 'model_output_map.npy', anomaly_map, allow_pickle=False)
        result['anomaly_map_note'] = 'Engine normalized model output before runtime ignore; internally upsampled; native feature map unavailable'
    result['decision'] = 'PASS' if result['status'] == 'OK' else result['status']
    result['execution_status'] = 'SUCCESS'
    result['raw_score'] = result['score']
    return result


def audit(output, plan, rows, flags):
    names = ['开裂', '隐裂（微裂）', '划痕', '毛刺', '变形', '压伤', '油污', '表面氧化', '成型挤料', '多料/少料']
    with (output / 'requirement_coverage.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['requirement_id', 'name', 'confirmed_defect_images', 'confirmed_normal_images_global', 'unknown_images', 'independent_normal_parts', 'independent_defect_parts', 'visible_view_count', 'complete_boxes', 'true_masks', 'repeat_capture_count', 'status'])
        writer.writeheader()
        for req, name in enumerate(names, 1):
            count = sum(r['ground_truth'] == 'defect' and req in r['requirement_ids'] for r in rows)
            writer.writerow(dict(requirement_id=req, name=name, confirmed_defect_images=count,
                confirmed_normal_images_global=sum(r['ground_truth'] == 'normal' for r in rows),
                unknown_images=sum(r['ground_truth'] == 'unknown' for r in rows), independent_normal_parts='UNKNOWN',
                independent_defect_parts='UNKNOWN', visible_view_count=1 if count else 0, complete_boxes=0, true_masks=0,
                repeat_capture_count='UNKNOWN', status='IMAGE_LEVEL_ONLY' if count else 'BLOCKED_CATEGORY_TRUTH'))
    with (output / 'split_manifest.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'data_audit.md').write_text('''# E0 数据审计

本轮复用现有8张左件 front 开发图，仅检查清单中的原图；4024×3036、uint8、fused_only。三通道并非完全相同，但未确认相机色彩来源。reference和normal_extra_001为用户确认正常，normal_extra_003为unknown；5张defect由用户确认轮廓变形，仅对应第5项图像级真值。未把通用defect目录转成十类标签。

物理零件ID、谱系、型号、独立缺陷实例、完整bbox和mask缺失，不能构造独立实体划分或正式排行榜。全部保留dev；reference在开发集用于建立参考，属于已知污染。历史YOLO训练元数据中007/015/020为train、002/005为calibration；当前模型的完整训练重合无法凭无SHA旧清单排除。EfficientAD原训练报告使用各视图90张正常图，pending_external_validation。模型内部模板银行不计为新增验证样本。

现用ROI配置SHA与历史训练/ignore index记录不同，本次冻结现用资产，同时保留历史来源；不回退配置。详细权重、模板、mask、阈值、训练args/data/report及代码SHA见baseline_snapshot.json。left/front映射运行时front，不是front_left。

只回放front分支，未获得同件完整8视图，不复现25检查整站结果。零件级指标、隐裂质量确认、深度/厚度指标均不可评估。未打开locked_test图像、未训练、未修改现场配方。
''')
    (output / 'blocked_items.md').write_text('''# 尚未解决

- 第1—4、6—10项缺确认类别样本；第2项还需要独立质量确认及表面可见性。R02a响应不能命名为裂纹。
- 缺完整定位标注：新增正确定位、AP、Dice、实例检出率不可评估；new_hits只记录自动NG图像拦截差集。
- 缺物理实体/谱系关联和干净训练划分；本轮development_only，禁止正式排名。
- 参考边界仍为draft，存在UNKNOWN，R01不能全周PASS；正常001仍有公差冲突。
- 缺同件8视图，baseline_system与完整工件耗时不可用。
- CUDA不可用，本轮CPU离线计时，包含文件、校验和可视化的wall与模型计时分开记录；不是产线节拍。
- 缺mask，本轮不启动监督分割；E0—E2不训练或下载新网络。
''')
    write_json(output / 'manifest_validation.json', flags)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('configs/bmw/benchmark/left_front_e2_development.json'))
    parser.add_argument('--output', type=Path, help='New output directory; existing directories are refused')
    args = parser.parse_args()
    plan_path = resolve(args.plan)
    plan = validate_plan(json.loads(plan_path.read_text()))
    if plan.get('status') != 'development_ready' or plan['evaluation_track'] != 'development_only':
        raise ValueError('This runner executes only resolved development_ready development_only plans')
    if plan['resources']['device'] != 'cpu':
        raise ValueError('This frozen replay adapter requires an explicit CPU plan')
    supported = set(FrozenBaselineAdapter.ALGORITHMS) | {'R01_contour', 'R02a_morph_lines'}
    if not set(plan['algorithms']) <= supported:
        raise ValueError('Plan requests an unimplemented branch')
    rows = json.loads(resolve(plan['manifest']).read_text())
    flags = validate_manifest(rows, plan['evaluation_track'])
    if any(row['hand'] != 'left' or row['view_id'] != 'front' for row in rows):
        raise ValueError('This development plan is restricted to left/front')
    output = resolve(args.output or plan['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        load_pixels(row)
    import torch
    torch.set_num_threads(plan['resources']['torch_num_threads'])
    torch.set_num_interop_threads(plan['resources']['torch_num_interop_threads'])
    cv2.setNumThreads(plan['resources']['torch_num_threads'])
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'opencv-python', 'ultralytics', 'numpy', 'scikit-image')}
    environment = {'python': platform.python_version(), 'platform': platform.platform(), 'versions': versions,
                   'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count(), 'resources': plan['resources']}
    write_json(output / 'environment.json', environment)
    (output / 'environment.txt').write_text(json.dumps(environment, ensure_ascii=False, indent=2) + '\n')
    write_json(output / 'resolved_config.json', plan)
    write_json(output / 'image_manifest.json', rows)
    audit(output, plan, rows, flags)
    adapter = FrozenBaselineAdapter(resolve(plan['baseline_profile']))
    write_json(output / 'baseline_snapshot.json', adapter.snapshot())
    files = [plan_path, resolve(plan['manifest']), resolve(plan['contour_config']), Path(__file__)]
    files += list((ROOT / 'src/bmw_inspection/benchmark').glob('*.py'))
    files += list((ROOT / 'src/bmw_inspection/checks/contour_compare').glob('*.py'))
    frozen = {str(p): digest(p) for p in files}
    manifest = {'status': 'RUNNING', 'evaluation_track': 'development_only', 'source_sha256': frozen,
                'planned_images': len(rows), 'planned_algorithms': plan['algorithms'], 'planned_prediction_count': len(rows) * len(plan['algorithms']),
                'training_executed': False, 'locked_test_access': False, 'full_station_replayed': False, 'source_validation': 'each image SHA verified before every measured prediction'}
    write_json(output / 'run_manifest.json', manifest)
    inputs = output / 'evidence/input'
    inputs.mkdir(parents=True)
    for row in rows:
        cv2.imwrite(str(inputs / (row['image_id'] + '.jpg')), cv2.resize(load_pixels(row), (1006, 759)))
    predictions, warmups = [], []
    for algorithm in plan['algorithms']:
        if algorithm != 'R01_contour':
            image = load_pixels(rows[0])
            started = time.perf_counter()
            try:
                for _ in range(plan['resources']['warmup_iterations']):
                    if algorithm == 'R02a_morph_lines':
                        predict_lines(image, plan['line_config'])
                    else:
                        adapter.predict(image, algorithm, view='front')
                warmups.append({'algorithm_id': algorithm, 'status': 'SUCCESS', 'wall_ms': (time.perf_counter() - started) * 1000})
            except Exception:
                warmups.append({'algorithm_id': algorithm, 'status': 'ERROR', 'error': traceback.format_exc(), 'wall_ms': (time.perf_counter() - started) * 1000})
            write_json(output / 'warmup.json', warmups)
        for row in rows:
            folder = output / 'evidence' / algorithm / row['image_id']
            folder.mkdir(parents=True)
            started = time.perf_counter()
            try:
                image = load_pixels(row)
                result = predict_one(plan, row, image, algorithm, adapter, folder)
            except Exception:
                result = {'decision': 'ERROR', 'execution_status': 'ERROR', 'raw_score': None, 'error': traceback.format_exc(), 'timing_ms': {}}
            result.update(image_id=row['image_id'], algorithm_id=algorithm, image_sha256=row['sha256'], device='cpu', evaluation_track='development_only')
            result['wall_ms'] = (time.perf_counter() - started) * 1000
            result['wall_scope'] = 'input_hash_read + algorithm + evidence_export; R01 uses cache import wall, see cached algorithm_total separately'
            overlay_path = folder / 'overlay.png'
            result['evidence_overlay'] = str(overlay_path.relative_to(output)) if overlay_path.exists() else None
            if overlay_path.exists():
                cv2.imwrite(str(folder / 'preview.jpg'), cv2.resize(cv2.imread(str(overlay_path)), (1006, 759)))
            write_json(folder / 'prediction.json', result)
            predictions.append(result)
            with (output / 'predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            print(json.dumps({key: result.get(key) for key in ('algorithm_id', 'image_id', 'decision', 'raw_score', 'candidate_count', 'candidate_ng_count', 'wall_ms', 'error')}, ensure_ascii=False), flush=True)
    adapter.verify_frozen()
    changed = [p for p, sha in frozen.items() if digest(p) != sha]
    if changed:
        raise RuntimeError('Experiment code/config changed during execution: ' + str(changed))
    for row in rows:
        load_pixels(row)
    metrics = summarize_predictions(rows, predictions, algorithm_ids=plan['algorithms'])
    write_json(output / 'metrics.json', metrics)
    from bmw_inspection.benchmark.reporting import write_reports
    write_reports(output, plan, rows, predictions, metrics)
    manifest.update(status='COMPLETED_WITH_ERRORS' if any(p['decision'] == 'ERROR' for p in predictions) else 'COMPLETED', actual_prediction_count=len(predictions), frozen_assets_unchanged=True, input_sha_unchanged=True)
    write_json(output / 'run_manifest.json', manifest)
    print(str(output / 'summary.md'), flush=True)


if __name__ == '__main__':
    main()
