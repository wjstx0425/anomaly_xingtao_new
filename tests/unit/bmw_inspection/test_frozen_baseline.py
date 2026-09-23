"""Frozen adapter contract tests with injected model backends."""

from types import SimpleNamespace

import numpy as np
import pytest

from bmw_inspection.benchmark.frozen_baseline import FrozenBaselineAdapter, _digest
from bmw_inspection.lab.eight_view_demo import BranchStatus
from bmw_inspection.lab.eight_view_demo_models import ModelOutput


@pytest.fixture
def adapter(monkeypatch):
    import bmw_inspection.benchmark.frozen_baseline as module
    config = SimpleNamespace(roi_config='roi', yolo_checkpoint='weights', yolo_candidate_conf=.1,
        yolo_final_threshold=.25, yolo_imgsz=640, yolo_ignore_regions={})
    monkeypatch.setattr(module, 'load_demo_config', lambda p: config)
    monkeypatch.setattr(module.runtime, 'load_part_rois', lambda p: {'front': (2,3,12,13)})
    monkeypatch.setattr(FrozenBaselineAdapter, '_build_snapshot', lambda self: {'files': {}, 'parameters': {'imgsz':640}})
    calls=[]
    class FakeYolo:
        names={0:'defect'}
        def predict(self, **kwargs):
            calls.append(kwargs)
            kwargs['source'][:]=0
            return [SimpleNamespace(speed={'inference':1.}, boxes=SimpleNamespace(xyxy=np.array([[1,2,5,6]]), conf=np.array([.8]), cls=np.array([0])))]
    loads=[]
    def factory(path):
        loads.append(path)
        return FakeYolo()
    # Avoid filesystem checkpoint validation while preserving actual prediction.
    def init(self,path,**kwargs):
        self._candidate_conf=kwargs['candidate_conf']; self._final_threshold=kwargs['final_threshold']
        self._imgsz=kwargs['imgsz']; self._ignore_regions={}
        self._model=kwargs['model_factory'](path)
        from threading import Lock
        self._lock=Lock()
    monkeypatch.setattr(module.runtime.EightViewYoloPredictor,'__init__',init)
    result=FrozenBaselineAdapter('profile',yolo_factory=factory)
    return result,calls,loads


def test_yolo_shared_cpu_frozen_threshold_and_coordinate_mapping(adapter):
    model,calls,loads=adapter
    image=np.full((20,20,3),80,np.uint8); before=image.copy()
    a=model.predict(image,'baseline_yolo26n')
    b=model.predict(image,'D01_yolo_high')
    assert len(loads)==1
    assert [c['imgsz'] for c in calls]==[640,1280]
    assert all(c['device']=='cpu' and c['conf']==.1 for c in calls)
    assert a['threshold']==b['threshold']==.25
    assert a['mapping']['boxes'][0]['xyxy_full']==[3,5,7,9]
    np.testing.assert_array_equal(image,before)
    assert a['status']=='NG'
    assert a['raw_map'] is None


def test_snapshot_is_owned_and_drift_is_rejected(adapter,tmp_path):
    model,_,_=adapter
    snapshot=model.snapshot(); snapshot['parameters']['imgsz']=1
    assert model.snapshot()['parameters']['imgsz']==640
    asset=tmp_path/'asset'; asset.write_text('first')
    model._snapshot['files']={str(asset):_digest(asset)}
    assert model.verify_frozen()
    asset.write_text('changed')
    with pytest.raises(RuntimeError,match='dependencies changed'):
        model.predict(np.zeros((20,20,3),np.uint8),'baseline_yolo26n')


def test_efficient_lazy_factory_preserves_model_output_map_and_loads_once(adapter):
    model,_,_=adapter
    loads=[]
    def factory(path):
        loads.append(path)
        return lambda image:(.6,True,np.full((3,4),.6,np.float32))
    model._efficient_factory=factory
    first=model._lazy_efficient('front_weights')
    model._lazy_efficient('other_weights')
    assert not loads
    first(np.zeros((10,10,3),np.uint8)); first(np.zeros((10,10,3),np.uint8))
    assert loads==['front_weights']
    assert model._raw_map.shape==(3,4)


def test_invalid_inputs_do_not_enter_model(adapter):
    model,calls,_=adapter
    with pytest.raises(ValueError,match='Unsupported'):
        model.predict(np.zeros((20,20,3),np.uint8),'other')
    with pytest.raises(ValueError,match='BGR'):
        model.predict(np.zeros((20,20),np.uint8),'baseline_yolo26n')
    with pytest.raises(ValueError,match='outside'):
        model.predict(np.zeros((5,5,3),np.uint8),'baseline_yolo26n')
    assert not calls


def test_efficient_predict_returns_model_output_map_and_runtime_details(adapter):
    model,_,_=adapter
    model.config.efficientad_checkpoints={'front':'front_weights'}
    model._efficient_factory=lambda path: lambda image:(.7,True,np.full((3,4),.7,np.float32))
    lazy=model._lazy_efficient('front_weights')
    class Predictor:
        def predict(self,view,crop):
            score,label,raw=lazy(crop)
            return ModelOutput(BranchStatus.NG,score,.39,'frozen scoring',crop.copy(),label,{'score_source':'accepted_component_max_p95'})
    model._predictors['baseline_efficientad']=Predictor()
    result=model.predict(np.ones((20,20,3),np.uint8),'baseline_efficientad')
    assert result['raw_map'].shape==(3,4)
    assert result['map_space']=='model_output_before_runtime_ignore'
    assert result['native_feature_map_available'] is False
    assert result['model_internal_upsampling'] is True
    assert result['mapping']['raw_map_shape_hw']==[3,4]
    assert result['details']['score_source']=='accepted_component_max_p95'
    assert result['threshold']==.39


def test_template_mapping_handles_non_square_roi_padding_and_nonzero_shift():
    from bmw_inspection.benchmark.frozen_baseline import _template_overlay_mapping
    result=_template_overlay_mapping((100,60),(40,40),(3,-2),(1000,2000))
    # Resize to 24x40, pad x=8. Query coordinate is overlay+(3,-2).
    assert result['target_content_xyxy']==[5,2,29,40]
    assert result['overlay_coordinate_system']=='template_aligned_target_pixels'
    matrix=np.asarray(result['overlay_to_full_3x3'])
    point=matrix@np.array([5,2,1])
    np.testing.assert_allclose(point,[1000.75,2000.75,1])
    assert result['resized_shape_hw']==[40,24]
    np.testing.assert_allclose(matrix[:2,:2],np.eye(2)*2.5)
