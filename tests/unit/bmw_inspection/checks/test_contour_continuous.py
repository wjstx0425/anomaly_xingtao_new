"""Integration contracts for the explicit continuous-edge development strategy."""
import cv2
import numpy as np
import pytest
from test_contour_extraction import scene, extract


def continuous_scene():
    image, reference, config = scene()
    config['extraction']['edge_selection_mode'] = 'continuous_v3'
    return image, reference, config


def test_continuous_self_has_image_supported_zero_displacement():
    image, ref, config = continuous_scene()
    original = image.copy()
    result = extract(image, ref, config)
    assert result['valid'].all()
    np.testing.assert_allclose(result['normal_offset_u_px'], 0, atol=1e-10)
    np.testing.assert_array_equal(image, original)
    assert result['diagnostics']['edge_selection_method'] == 'continuous_v3'


def test_continuous_notch_keeps_thirteen_pixel_depth():
    image, ref, config = continuous_scene()
    image[65:88, 45:58] = 20
    result = extract(image, ref, config)
    selected = (ref['sample_xy'][:,0] == 45) & (ref['sample_xy'][:,1]>=68) & (ref['sample_xy'][:,1]<=84)
    assert result['valid'][selected].all()
    np.testing.assert_allclose(result['normal_offset_u_px'][selected],13,atol=1e-9)


def test_continuous_reference_mask_offset_preserves_subpixel_motion():
    image, ref, config = continuous_scene()
    warp = lambda im,dx: cv2.warpAffine(im,np.array([[1,0,dx],[0,1,0]],float),(180,150),borderValue=(20,20,20))
    ref['image'] = warp(image,3)
    current = warp(image,3.25)
    result = extract(current,ref,config)
    left = ref['sample_xy'][:,0] == 45
    assert result['valid'][left].all()
    np.testing.assert_allclose(result['normal_offset_u_px'][left],.25,atol=1e-8)


def test_continuous_missing_image_evidence_stays_unknown():
    image, ref, config = continuous_scene()
    image[:] = 20
    result = extract(image,ref,config)
    assert not result['valid'].any()
    assert np.isnan(result['sample_xy']).all()


@pytest.mark.parametrize('key,value',[('edge_selection_mode','bad'),('reference_edge_search_px',float('nan')),('path_ambiguity_margin',0),('path_continuity_weight',True)])
def test_continuous_invalid_settings_rejected(key,value):
    image, ref, config = continuous_scene()
    config['extraction'][key]=value
    with pytest.raises(ValueError):
        extract(image,ref,config)


def test_continuous_tracks_real_corners_displaced_six_pixels():
    from bmw_inspection.checks.contour_compare.geometry import resample_closed
    image, ref, config = continuous_scene()
    ref.update(resample_closed(ref['dense_xy'],1.))
    current=cv2.warpAffine(image,np.array([[1,0,6],[0,1,0]],float),(180,150),borderValue=(20,20,20))
    result=extract(current,ref,config)
    corners=ref['corner_mask']
    assert corners.any()
    assert result['valid'][corners].all()
    np.testing.assert_allclose(result['sample_xy'][corners]-ref['sample_xy'][corners],np.tile([6.,0.],(corners.sum(),1)),atol=1e-8)


def test_continuous_unconfirmed_reference_cells_break_paths_and_have_reason():
    image, ref, config = continuous_scene()
    ref['reference_valid']=np.ones(len(ref['sample_xy']),bool)
    ref['reference_valid'][10:13]=False
    result=extract(image,ref,config)
    assert not result['valid'][10:13].any()
    assert np.isnan(result['sample_xy'][10:13]).all()
    assert set(result['invalid_reason'][10:13]) == {'reference_material_normal_unconfirmed'}
    assert result['diagnostics']['reference_path_selection']['missing_row_count']>=3


def test_continuous_evidence_saves_actual_measured_overlay(tmp_path):
    from bmw_inspection.checks.contour_compare.geometry import resample_closed
    from bmw_inspection.checks.contour_compare.comparison import compare_outline
    from bmw_inspection.checks.contour_compare.evidence import write_evidence
    image, ref, config = continuous_scene()
    ref.update(resample_closed(ref['dense_xy'],1.))
    n=len(ref['sample_xy'])
    ref.update(arc_id=np.zeros(n,int),required_mask=np.ones(n,bool))
    observation=extract(image,ref,config)
    observation['registration']['T_reference_to_test']=np.eye(3)
    result=compare_outline(ref,observation,{'comparison':{'default_inward_tolerance_px':4.,'default_outward_tolerance_px':4.,'min_exceedance_arc_px':15.}})
    output=write_evidence(image,ref,observation,result,tmp_path/'evidence')
    assert output['evidence']['measured_outline_overlay']=='measured_outline_overlay.png'
    assert cv2.imread(str(tmp_path/'evidence/measured_outline_overlay.png')).shape==image.shape
