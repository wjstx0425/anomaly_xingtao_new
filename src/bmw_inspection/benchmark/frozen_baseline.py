"""Image-only CPU replay of frozen BMW runtime branches, without fitting."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from bmw_inspection.lab.eight_view_demo import load_demo_config
from bmw_inspection.lab import eight_view_demo_models as runtime


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _template_overlay_mapping(crop_shape, target_shape, shift_xy, roi_origin):
    """Map aligned template pixels to full image, excluding reflected padding."""
    height, width = crop_shape
    target_height, target_width = target_shape
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, min(target_width, int(round(width * scale))))
    resized_height = max(1, min(target_height, int(round(height * scale))))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    sx, sy = resized_width / width, resized_height / height
    dx, dy = shift_xy
    x0, y0 = roi_origin
    matrix = [[1 / sx, 0, x0 + (dx - left + .5) / sx - .5],
              [0, 1 / sy, y0 + (dy - top + .5) / sy - .5], [0, 0, 1]]
    content = [max(0, left - dx), max(0, top - dy),
               min(target_width, left + resized_width - dx), min(target_height, top + resized_height - dy)]
    return {'overlay_coordinate_system': 'template_aligned_target_pixels',
            'overlay_to_full_3x3': matrix, 'target_content_xyxy': content,
            'target_shape_hw': [target_height, target_width],
            'resized_shape_hw': [resized_height, resized_width],
            'letterbox_padding_left_top': [left, top], 'best_shift_xy': [dx, dy],
            'mapping_scope': 'display_only; discard reflected padding outside target_content_xyxy'}


def _digest(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_efficient_factory(path):
    from anomalib.engine import Engine
    predictor = runtime._AnomalibEfficientPredictor(path)
    predictor._engine = Engine(logger=False, accelerator='cpu', devices=1)
    return predictor


def _cpu_yolo_factory(path):
    from ultralytics import YOLO
    model = YOLO(str(path))
    model.to('cpu')
    return model


class FrozenBaselineAdapter:
    """Reuse runtime scoring while loading only requested neural branches.

    Factories are injection points for offline tests. ``predict`` accepts only
    pixels and a view name: no annotation, split or label enters inference.
    Exceptions propagate to the experiment runner, which must record ERROR.
    ``raw_map`` is the normalized model output before runtime ignore masking.
    EfficientAD internally upsamples this map; it is not a native feature map.
    Model preprocessing inside Engine/Ultralytics is included in backend timing;
    timing fields explicitly describe these limits and are not kernel timings.
    """

    ALGORITHMS = ('baseline_template', 'baseline_efficientad', 'baseline_yolo26n', 'D01_yolo_high')

    def __init__(self, profile_path, *, efficient_factory=None, yolo_factory=None):
        self.config = load_demo_config(Path(profile_path))
        self.rois = runtime.load_part_rois(self.config.roi_config)
        roi_path = Path(self.config.roi_config)
        roi_metadata = json.loads(roi_path.read_text()) if roi_path.is_file() else {}
        self._expected_shape = (roi_metadata.get('image_height'), roi_metadata.get('image_width'))
        self._efficient_factory = efficient_factory or _cpu_efficient_factory
        self._yolo_factory = yolo_factory or _cpu_yolo_factory
        self._predictors = {}
        self._efficient_loaded = {}
        self._shared_yolo = None
        self._backend_ms = 0.0
        self._raw_map = None
        self._yolo_speed = None
        self._snapshot = self._build_snapshot()

    def _build_snapshot(self):
        c = self.config
        paths = {Path(__file__), c.path, c.roi_config, c.yolo_checkpoint, *c.template_models.values(), *c.efficientad_checkpoints.values(), Path(runtime.__file__)}
        for model_path in c.template_models.values():
            model = json.loads(model_path.read_text())
            paths.update((model_path.parent / item['path']).resolve() for item in model['templates'])
        for index in (c.template_ignore_mask_index, c.efficientad_ignore_mask_index):
            if index is not None:
                paths.add(index)
                payload = json.loads(index.read_text())
                paths.update((index.parent / item['mask_path']).resolve() for item in payload['views'].values())
        if c.template_weighted_regions is not None:
            paths.add(c.template_weighted_regions.roi_config)
        if c.efficientad_component_filter_config is not None:
            paths.add(c.efficientad_component_filter_config)
        # Bind implementation of preprocessing and postprocessing as well.
        for name in ('eight_view_demo.py', 'efficientad_component_filter.py', 'efficientad_ignore_mask.py', 'template_region_weighting.py'):
            paths.add(Path(runtime.__file__).parent / name)
        provenance = {}
        for label, path in [('efficientad_metrics', c.efficientad_checkpoints['front'].parent / 'metrics.json'),
                            ('efficientad_run_report', c.efficientad_checkpoints['front'].parents[2] / 'run_report.json'),
                            ('yolo_training_args', c.yolo_checkpoint.parents[1] / 'args.yaml')]:
            if path.is_file():
                paths.add(path)
                provenance[label] = {'path': str(path), 'text': path.read_text()}
        args_path = c.yolo_checkpoint.parents[1] / 'args.yaml'
        if args_path.is_file():
            import yaml
            args = yaml.safe_load(args_path.read_text())
            data_path = Path(args.get('data', ''))
            if data_path.is_file():
                paths.add(data_path)
                provenance['yolo_training_data'] = {'path': str(data_path), 'text': data_path.read_text()}
        return {'schema_version': 1, 'device': 'cpu', 'evaluation_track': 'deployment_replay',
                'profile_path': str(c.path), 'profile': json.loads(c.path.read_text()),
                'files': {str(p.resolve()): _digest(p) for p in sorted(paths)},
                'parameters': {'view_mapping': 'hand selects profile; left/front maps to front',
                               'roi_xyxy': _plain(self.rois), 'baseline_yolo_imgsz': c.yolo_imgsz,
                               'D01_yolo_high_imgsz': 1280, 'high_variant_changes': ['imgsz'],
                               'candidate_conf': c.yolo_candidate_conf, 'final_threshold': c.yolo_final_threshold},
                'training_provenance': provenance,
                'limitations': ['Physical part identity and clean training separation not established.',
                                'Existing all-normal EfficientAD is deployment replay, not clean-split evidence.',
                                'Historical ROI hashes may differ from frozen current profile; preserve both.',
                                'Single-view branch replay does not represent the complete 25-check runtime.']}

    def snapshot(self):
        """Return an owned, JSON-compatible frozen asset and parameter record."""
        return deepcopy(self._snapshot)

    def verify_frozen(self):
        """Raise before replay if a dependency disappeared or its bytes changed."""
        changed = [path for path, digest in self._snapshot['files'].items() if not Path(path).is_file() or _digest(path) != digest]
        if changed:
            raise RuntimeError('Frozen baseline dependencies changed: ' + ', '.join(changed))
        return True

    def _masks(self, index):
        if index is None:
            return None
        shapes = {view: (box[3] - box[1], box[2] - box[0]) for view, box in self.rois.items()}
        return runtime._load_demo_ignore_masks(index, shapes)

    def _lazy_efficient(self, path):
        def predict(image):
            if path not in self._efficient_loaded:
                self._efficient_loaded[path] = self._efficient_factory(path)
            start = perf_counter()
            output = self._efficient_loaded[path](image)
            self._backend_ms += (perf_counter() - start) * 1000
            self._raw_map = np.asarray(output[2]).copy()
            return output
        return predict

    def _shared_yolo_factory(self, path):
        if self._shared_yolo is None:
            self._shared_yolo = self._yolo_factory(path)
        owner = self
        class TimedModel:
            names = owner._shared_yolo.names
            def predict(self, **kwargs):
                start = perf_counter()
                output = owner._shared_yolo.predict(**kwargs, device='cpu')
                owner._backend_ms += (perf_counter() - start) * 1000
                owner._yolo_speed = _plain(getattr(output[0], 'speed', None)) if output else None
                return output
        return TimedModel()

    def _get_predictor(self, algorithm):
        if algorithm in self._predictors:
            return self._predictors[algorithm]
        c = self.config
        if algorithm == 'baseline_template':
            weighted = c.template_weighted_regions
            shapes = {v: (b[3]-b[1], b[2]-b[0]) for v, b in self.rois.items()}
            regions = runtime.load_template_weighted_regions(weighted.roi_config, expected_shapes=shapes) if weighted is not None and weighted.enabled else None
            predictor = runtime.EightViewTemplatePredictor(c.template_models, thresholds=c.template_thresholds,
                ignore_masks=self._masks(c.template_ignore_mask_index), weighted_regions=regions,
                weighted_region_weight=weighted.weight if regions is not None else 1,
                weighted_outside_weight=weighted.outside_weight if regions is not None else 1,
                weighted_thresholds=weighted.thresholds if regions is not None else None)
        elif algorithm == 'baseline_efficientad':
            predictor = runtime.EightViewEfficientAdPredictor(c.efficientad_checkpoints, thresholds=c.efficientad_thresholds,
                base_thresholds=c.efficientad_base_thresholds, threshold_margin=c.efficientad_threshold_margin,
                predictor_factory=self._lazy_efficient, ignore_masks=self._masks(c.efficientad_ignore_mask_index),
                component_policies=c.efficientad_component_policies, threshold_source=c.efficientad_threshold_source,
                validation_status=c.efficientad_validation_status)
        else:
            predictor = runtime.EightViewYoloPredictor(c.yolo_checkpoint, candidate_conf=c.yolo_candidate_conf,
                final_threshold=c.yolo_final_threshold, imgsz=1280 if algorithm == 'D01_yolo_high' else c.yolo_imgsz,
                ignore_regions=c.yolo_ignore_regions, model_factory=self._shared_yolo_factory)
        self._predictors[algorithm] = predictor
        return predictor

    def predict(self, image, algorithm_id, view='front'):
        """Return runtime output, owned arrays, crop mapping and CPU timings."""
        if algorithm_id not in self.ALGORITHMS or view not in self.rois:
            raise ValueError('Unsupported frozen algorithm or view')
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Frozen replay requires uint8 BGR image')
        if all(value is not None for value in self._expected_shape) and image.shape[:2] != self._expected_shape:
            raise ValueError('Input shape differs from frozen ROI configuration')
        self.verify_frozen()
        begin = perf_counter()
        x1, y1, x2, y2 = self.rois[view]
        if x2 > image.shape[1] or y2 > image.shape[0]:
            raise ValueError('Frozen ROI lies outside input image')
        crop = image[y1:y2, x1:x2].copy()
        preprocess_ms = (perf_counter()-begin)*1000
        load_start = perf_counter()
        predictor = self._get_predictor(algorithm_id)
        if algorithm_id == 'baseline_efficientad':
            path = self.config.efficientad_checkpoints[view]
            if path not in self._efficient_loaded:
                self._efficient_loaded[path] = self._efficient_factory(path)
        load_ms = (perf_counter()-load_start)*1000
        self._raw_map = None
        self._backend_ms = 0
        self._yolo_speed = None
        started = perf_counter()
        output = predictor.predict(view, crop)
        runtime_ms = (perf_counter()-started)*1000
        post_start = perf_counter()
        details = _plain(output.details)
        boxes = [{**item, 'xyxy_full': [item['xyxy'][0]+x1, item['xyxy'][1]+y1, item['xyxy'][2]+x1, item['xyxy'][3]+y1]} for item in details.get('boxes', [])]
        backend_ms = runtime_ms if algorithm_id == 'baseline_template' else self._backend_ms
        result = {'algorithm_id': algorithm_id, 'view_id': view, 'device': 'cpu', 'status': output.status.value,
                  'score': output.score, 'threshold': output.threshold, 'reason': output.reason,
                  'raw_pred_label': output.raw_pred_label, 'details': details,
                  'overlay': None if output.overlay is None else output.overlay.copy(), 'raw_map': self._raw_map,
                  'map_space': 'model_output_before_runtime_ignore' if self._raw_map is not None else None,
                  'native_feature_map_available': False,
                  'model_internal_upsampling': self._raw_map is not None,
                  'mapping': {'roi_xyxy': [x1,y1,x2,y2], 'crop_shape_hw': list(crop.shape[:2]),
                              'input_shape_hw': list(image.shape[:2]), 'crop_to_full': [[1,0,x1],[0,1,y1],[0,0,1]],
                              'overlay_coordinate_system': 'roi_pixels', 'boxes': boxes,
                              'raw_map_shape_hw': None if self._raw_map is None else list(self._raw_map.shape)},
                  'timing_ms': {'preprocess': preprocess_ms, 'inference': backend_ms,
                                'postprocess': max(0.0,runtime_ms-backend_ms)+(perf_counter()-post_start)*1000,
                                'model_load': load_ms, 'runtime_total': runtime_ms},
                  'timing_scope': 'preprocess=ROI copy; inference=backend including internal transforms; postprocess=runtime scoring/rendering plus output mapping; template inference includes its complete runtime',
                  'backend_reported_speed_ms': self._yolo_speed}
        if algorithm_id == 'baseline_template' and output.overlay is not None:
            result['mapping'].update(_template_overlay_mapping(
                crop.shape[:2], output.overlay.shape[:2],
                (details['best_shift_x'], details['best_shift_y']), (x1, y1)))
        return result
