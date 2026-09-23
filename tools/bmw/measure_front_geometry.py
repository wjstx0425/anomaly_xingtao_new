"""Automatically locate existing front images and record independent geometry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
from pathlib import Path
import time

import cv2
import numpy as np

from bmw_inspection.checks.contour_compare.contracts import check_image, read_json, write_json, validate_config
from bmw_inspection.checks.contour_compare import FrontContourInspector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-mask', required=True)
    parser.add_argument('--acquire-contour', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest, config = read_json(args.manifest), validate_config(read_json(args.config), draft=True)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json',manifest);write_json(output/'config.json',config)
    reference_row = next(r for r in manifest['samples'] if r['sample_id'] == manifest['reference_sample_id'])
    def load(row):
        path = root/row['image_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        image = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if image is None or list(image.shape) != row['shape']:
            raise ValueError(f'Image shape mismatch: {path}')
        check_image(image, config)
        if any(row.get(key) != config[key] for key in ('hand', 'view_id')) or row.get('channel') != config['image']['channel']:
            raise ValueError(f'Image hand/view/channel mismatch: {path}')
        if Path(row['sample_id']).name != row['sample_id'] or row['sample_id'] in ('.', '..'):
            raise ValueError('sample_id must be a directory basename')
        return image
    reference = load(reference_row)
    mask_path = Path(args.reference_mask)
    reference_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if reference_mask is None or reference_mask.shape != reference.shape[:2]:
        raise ValueError('Reference mask shape mismatch')
    write_json(output/'reference_mask_source.json',{'path':str(mask_path.resolve()),'sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest()})
    inspector = FrontContourInspector(config, reference, reference_mask)
    reference_lines = inspector.reference_features
    write_json(output/'reference_features.json',reference_lines)
    summaries, tiles, cards = [], [], []
    for row in manifest['samples']:
        started = time.perf_counter();image = load(row)
        result = inspector.process(image, hand=row['hand'], view_id=row['view_id'],
                                   channel=row['channel'], acquire_contour=args.acquire_contour)
        record = result.record
        record.update(sample_id=row['sample_id'], image_path=row['image_path'], image_sha256=row['sha256'])
        pose = record['pose']
        folder = output/row['sample_id'];folder.mkdir()
        overlay=image[:,:,:3].copy()
        if pose['valid']:
            if args.acquire_contour:
                acquired = result.contour
                contour_overlay = image[:,:,:3].copy()
                candidate_overlay = image[:,:,:3].copy()
                for i,j in enumerate(acquired['segment_end_index']):
                    if j < 0:
                        continue
                    a,b=np.rint(acquired['candidate_xy'][[i,j]]).astype(int)
                    cv2.line(candidate_overlay,tuple(a),tuple(b),(220,0,220),2,cv2.LINE_AA)
                    if acquired['segment_valid'][i]:
                        a,b=np.rint(acquired['observed_xy'][[i,j]]).astype(int)
                        cv2.line(contour_overlay,tuple(a),tuple(b),(255,220,0),2,cv2.LINE_AA)
                    else:
                        cv2.circle(contour_overlay,tuple(a),2,(0,140,255),-1)
                assert cv2.imwrite(str(folder/'acquired_outline_overlay.png'),contour_overlay)
                assert cv2.imwrite(str(folder/'candidate_outline_overlay.png'),candidate_overlay)
                assert cv2.imwrite(str(folder/'current_mask.png'),result.current_mask*255)
                np.savez_compressed(folder/'acquired_outline.npz',**{k:v for k,v in acquired.items() if isinstance(v,np.ndarray) and v.dtype.kind!='O'})
                with (folder/'acquired_outline.csv').open('w') as stream:
                    writer=csv.writer(stream);writer.writerow(['index','candidate_x','candidate_y','observed_x','observed_y','valid','reason','next_index','segment_valid'])
                    for i,(candidate,observed) in enumerate(zip(acquired['candidate_xy'],acquired['observed_xy'])):
                        writer.writerow([i,*[v if np.isfinite(v) else '' for v in candidate],*[v if np.isfinite(v) else '' for v in observed],bool(acquired['valid'][i]),acquired['invalid_reason'][i],acquired['segment_end_index'][i],bool(acquired['segment_valid'][i])])
                record['contour_acquisition']={k:v for k,v in acquired.items() if not isinstance(v,np.ndarray)}
                write_json(folder/'acquisition.json',record['contour_acquisition'])
            holes=np.array([a['test']['center_xy'] for a in pose['anchors']])
            for k,point in enumerate(holes):
                cv2.drawMarker(overlay,tuple(np.rint(point).astype(int)),(0,0,255),cv2.MARKER_CROSS,30,3)
                cv2.putText(overlay,f'H{k+1}',tuple(np.rint(point+[15,-15]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            for spec, line in zip(config['geometry_features'], record['features']):
                for point in line['points_xy'][line['point_valid']]:
                    cv2.circle(overlay,tuple(np.rint(point).astype(int)),2,(255,255,0),-1)
                if line['valid']:
                    a,b=np.rint(line['observed_support_endpoints_xy']).astype(int)
                    cv2.putText(overlay,spec['name'],tuple(a+[10,-10]),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,255),2)
                    for index in spec.get('display_hole_indices',range(len(holes))):
                        hole,distance=holes[index],line['hole_distances'][index]
                        foot=distance['perpendicular_foot_xy']
                        cv2.line(overlay,tuple(np.rint(hole).astype(int)),tuple(np.rint(foot).astype(int)),(0,180,255),1)
        record['elapsed_ms']=(time.perf_counter()-started)*1000
        assert cv2.imwrite(str(folder/'geometry_overlay.png'),overlay)
        write_json(folder/'geometry.json',record)
        brief={'sample_id':row['sample_id'],'pose_valid':pose['valid'],'independent_check_consistent':record['independent_check_consistent'],
               'hole_spacing_px':record.get('hole_spacing_px'),'elapsed_ms':record['elapsed_ms'],
               'features':[{'name':f['name'],'valid':f['valid'],'support_fraction':f['support_fraction'],
                            'fit_rms_px':f.get('fit_rms_px'),'delta_center_to_line_px':f.get('delta_center_to_line_px'),
                            'hole_distances_px':[d['distance_px'] for d in f['hole_distances']]} for f in record['features']]}
        brief['contour_acquisition']=record.get('contour_acquisition')
        summaries.append(brief);write_json(output/'summary.json',summaries)
        x1,y1,x2,y2=config['part_roi_xyxy']
        crop=cv2.resize(overlay[y1:y2,x1:x2],(436,590))
        tile=cv2.copyMakeBorder(crop,40,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(tile,row['sample_id']+' pose='+str(pose['valid']),(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        tiles.append(tile)
        cards.append(f'<section><h2>{html.escape(row["sample_id"])}</h2><p>双孔定位：{pose["valid"]}；独立边段一致性：{record["independent_check_consistent"]}；开发级 REVIEW</p><a href="{row["sample_id"]}/geometry_overlay.png"><img src="{row["sample_id"]}/geometry_overlay.png"></a><p><a href="{row["sample_id"]}/geometry.json">完整几何测量 JSON</a></p></section>')
        if args.acquire_contour and pose['valid']:
            cards.append(f'<p><a href="{row["sample_id"]}/candidate_outline_overlay.png">完整分割候选</a> · <a href="{row["sample_id"]}/acquired_outline_overlay.png">实际图像支持轮廓</a> · <a href="{row["sample_id"]}/current_mask.png">当前mask</a> · <a href="{row["sample_id"]}/acquired_outline.csv">原图轮廓坐标CSV</a></p>')
        print({k:v for k,v in brief.items() if k != 'contour_acquisition'},flush=True)
    if len(tiles)==8:
        assert cv2.imwrite(str(output/'geometry_contact_sheet.jpg'),np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
    (output/'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>BMW 自动定位与几何测量</title><style>body{font-family:sans-serif;margin:24px}img{width:min(100%,950px)}section{margin-bottom:40px}</style><h1>BMW front：双孔自动定位与独立几何测量</h1><p>红色十字为孔中心，青色为实际直边采样点，黄色线为孔到拟合无限直线的垂线。垂足可能位于观测线段外，JSON明确记录此情况。小脚不参与定位，线段拟合不用于补全轮廓。全部为像素投影测量和开发级 REVIEW。</p>'+''.join(cards)+'</html>',encoding='utf-8')


if __name__=='__main__':
    main()
