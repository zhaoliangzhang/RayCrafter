"""Audit source splits/priors, freeze a no-resampling test manifest and link bundles."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import shutil
from eval_shared import *
import numpy as np
from PIL import Image

def progress(phase,**values):
    record={'time':now(),'phase':phase,**values}
    write_json(OUT/'runtime/preparation.json',record)
    print(json.dumps(record),flush=True)

def pose_check(row):
    poses=np.asarray([v['w2c'] for v in row['references']+row['targets']])
    assert poses.shape==(5,4,4) and np.isfinite(poses).all()
    assert np.allclose(poses[:,3],[0,0,0,1],atol=1e-5)
    rot=poses[:,:3,:3]
    assert np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=2e-3)
    assert np.allclose(np.linalg.det(rot),1,atol=2e-3)

def perspective_row(item):
    row=copy.deepcopy(item)
    directory=PERSPECTIVE/'priors/samples'/item['id']
    assert (directory/'_SUCCESS').is_file()
    meta=json.loads((directory/'metadata.json').read_text())
    assert meta['set_id']==item['id'] and meta['source_sample']==item['source_sample']
    assert meta['scene_name']==item['scene_name'] and meta['split']==item['split']=='test'
    assert meta['distortion_applied'] and not meta['target_rgb_used_for_reconstruction']
    assert meta['num_reference_images']==2 and meta['num_query_renders']==3
    for role,views in [('references',row['references']),('targets',row['targets'])]:
        assert len(views)==(2 if role=='references' else 3)
        for j,(view,actual) in enumerate(zip(views,meta[role],strict=True)):
            assert all(view[k]==actual[k] for k in ('image','frame_index','w2c','camera'))
            view['original_source_image']=view['image']
            view['image']=str(directory/f'{"ref" if role=="references" else "gt"}_{j}.png')
            if role=='targets':
                view['noisy']=str(directory/f'prior_{j}.png')
                assert actual['prior']==view['noisy'] and actual['gt']==view['image']
    row.update(id='dl3dv_'+item['id'],source_id=item['id'],dataset='DL3DV',
               scene_name='DL3DV::'+item['scene_name'],projection='perspective',height=288,width=512,
               source_manifest=str(PERSPECTIVE/'benchmark/manifest.json'),prior_metadata=str(directory/'metadata.json'))
    return row

def audit_images(row):
    pose_check(row)
    digests={}
    expected=(row['width'],row['height'])
    for view in row['references']+row['targets']:
        native=expected if row['projection']=='panorama' else (view['camera']['width'],view['camera']['height'])
        for key in ('image','noisy'):
            if key not in view:
                continue
            path=Path(view[key])
            with Image.open(path) as im:
                assert im.mode=='RGB' and im.size==native,(path,im.mode,im.size,native)
                im.verify()
            digests[str(path)]=sha256(path)
    for j,target in enumerate(row['targets']):
        destination=OUT/'metric_ground_truth'/row['dataset']/row['id']/f'query_{j}.png'
        if row['projection']=='panorama':
            link(target['image'],destination)
        elif not destination.exists():
            with Image.open(target['image']) as im:
                atomic_png(im.convert('RGB').resize(expected,Image.Resampling.BILINEAR),destination)
        with Image.open(destination) as im:
            assert im.size==expected and im.mode=='RGB'
            im.verify()
        target['metric_gt']=str(destination)
        target['reference_overlap']=target['frame_index'] in {r['frame_index'] for r in row['references']}
        digests[str(destination)]=sha256(destination)
    for task in TASKS:
        dest=bundle(row,task)
        (dest/'predictions').mkdir(parents=True,exist_ok=True)
        for j,ref in enumerate(row['references']):
            link(ref['image'],dest/'references'/f'ref_{j}.png')
        for j,target in enumerate(row['targets']):
            link(target['metric_gt'],dest/'ground_truth'/f'query_{j}.png')
            link(target['image'],dest/'ground_truth_original'/f'query_{j}.png')
            if task=='cleanup':
                link(target['noisy'],dest/'noisy_inputs'/f'query_{j}.png')
        if task=='cleanup':
            link(row['prior_metadata'],dest/'noisy_inputs/provenance.json')
        write_json(dest/'input_metadata.json',{**row,'task':task,
            'input_query_rgb':'all_black' if task=='nvs' else 'cached_raw_point_render',
            'online_pixel_dropout':False,'resize':'PIL bilinear to requested inference resolution',
            'GT_access_during_inference':False},immutable=True)
    return row,digests

def main():
    imports()
    from safetensors import safe_open
    from mixed_camera import camera_features
    from evaluate_framecrafter_model2_replica import load_and_audit_benchmark
    OUT.mkdir(parents=True,exist_ok=True)
    progress('checking_checkpoint_and_source_manifests')
    with safe_open(str(CHECKPOINT),framework='pt',device='cpu') as state:
        assert len(state.keys())==802
        assert state.get_slice('patch_embedding.weight').get_shape()==[5120,420,1,2,2]
        checkpoint_meta=state.metadata()
        assert checkpoint_meta['step']=='22000' and checkpoint_meta['camera_injection']=='concat'
        assert checkpoint_meta['manifest_sha256']==json.loads((TRAIN/'data_manifest/summary.json').read_text())['sha256']
    source_hashes={str(CHECKPOINT):sha256(CHECKPOINT)}
    old=json.loads((PREVIOUS/'provenance.json').read_text())
    assert sha256(PREVIOUS/'benchmark/manifest.json')==old['benchmark_sha256']
    pano,items,audit=load_and_audit_benchmark(PREVIOUS/'benchmark')
    for d in pano['dataset_audits']:
        assert sha256(d['source_manifest'])==d['source_manifest_sha256']
        source_hashes[d['source_manifest']]=d['source_manifest_sha256']
    expected_priors={(r['set_id'],r['target_number']):r['sha256'] for r in json.loads((PREVIOUS/'panovggt_cache_hashes.json').read_text())}
    rows=[]
    for item in items:
        row=copy.deepcopy(item)
        directory=PRIORS/'samples'/item['id']
        assert (directory/'_SUCCESS').is_file()
        meta=json.loads((directory/'metadata.json').read_text())
        assert meta['source_sample']==row['source_sample']
        assert meta['input_reference_images']==[v['image'] for v in row['references']]
        assert meta['num_reference_images']==2 and meta['num_query_renders']==3
        assert meta['render_resolution']==[256,512] and not meta['target_rgb_accessed']
        for j,t in enumerate(row['targets']):
            mt=meta['targets'][j]
            assert mt['frame_index']==t['frame_index'] and mt['target_number']==j
            assert mt['target_pose_used'] and not mt['target_rgb_accessed']
            t['noisy']=str(directory/f'prior_{j}.png')
            assert sha256(t['noisy'])==expected_priors[item['id'],j]
        row.update(id=row['dataset'].lower()+'_'+item['id'],source_id=item['id'],
                   projection='panorama',height=256,width=512,prior_metadata=str(directory/'metadata.json'))
        rows.append(row)
    pm_path=PERSPECTIVE/'benchmark/manifest.json'
    pm=json.loads(pm_path.read_text());contract=json.loads((PERSPECTIVE/'priors/run_contract.json').read_text())
    assert pm['logical_sets']==len(pm['sets'])==2800 and pm['projection']=='perspective'
    assert pm['pose_convention']=='OpenCV world-to-camera'
    assert contract['manifest_sha256']==sha256(pm_path) and not contract['ignore_distortion']
    source_hashes[str(pm_path)]=sha256(pm_path)
    for shard in range(4):
        assert (PERSPECTIVE/f'priors/_SUCCESS.shard{shard:02d}').is_file()
        config=json.loads((PERSPECTIVE/f'priors/configuration_shard_{shard:02d}.json').read_text())
        assert not config['skipped_degenerate_sets']
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows.extend(pool.map(perspective_row,pm['sets']))
    assert Counter(r['dataset'] for r in rows)==COUNTS and len({r['id'] for r in rows})==sum(COUNTS.values())
    for i,row in enumerate(rows):
        row.update(index=i,seed=BASE_SEED+i)
    train_scenes={}
    for dataset in ('HM3D','JHU'):
        train_scenes[dataset]=set(json.loads((TRAIN/f'data_manifest/{dataset.lower()}_scene_selection.json').read_text())['selected'])
    train_scenes['DL3DV']=set(json.loads((TRAIN/'data_manifest/perspective_source_audit.json').read_text())['available_sets_per_scene'])
    overlaps={d:sorted(train_scenes[d]&{r['scene_name'] for r in rows if r['dataset']==d}) for d in train_scenes}
    assert not any(overlaps.values()),overlaps
    assert {r['scene_name'].split('::')[-1] for r in rows if r['dataset']=='JHU'}==set(old['jhu_heldout_scenes_verified_against_training_summary'])
    # Validate every distinct perspective calibration at the actual inference grid.
    calibrations={}
    for row in rows:
        if row['projection']=='perspective':
            key=json.dumps([v['camera'] for v in row['references']+row['targets']],sort_keys=True)
            calibrations.setdefault(key,row)
    for row in [rows[0],*calibrations.values()]:
        rays,scale=camera_features(row,'concat')
        assert tuple(rays['raymap'].shape)==(5,384,row['height']//8,64) and scale>0
    progress('auditing_images_and_creating_linked_bundles',sets=len(rows),perspective_calibrations=len(calibrations))
    final=[];digests={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(row,hashes) in enumerate(pool.map(audit_images,rows),1):
            final.append(row);digests.update(hashes)
            if i%100==0 or i==len(rows):
                progress('auditing_images_and_creating_linked_bundles',completed=i,total=len(rows))
    overlap_counts={d:sum(t['reference_overlap'] for r in final if r['dataset']==d for t in r['targets']) for d in COUNTS}
    assert overlap_counts=={'HM3D':21,'Replica':0,'JHU':0,'DL3DV':0},overlap_counts
    data={'sets':final,'logical_sets':len(final),'target_images':3*len(final),'datasets':COUNTS,
          'pose_convention':'OpenCV world-to-camera','resolutions_wh':{'panorama':[512,256],'perspective':[512,288]}}
    write_json(OUT/'benchmark/manifest.json',data,immutable=True)
    write_json(OUT/'input_hashes.json',digests,immutable=True)
    for path,digest in source_hashes.items():
        assert sha256(path)==digest
    protocol={'checkpoint':str(CHECKPOINT),'checkpoint_sha256':source_hashes[str(CHECKPOINT)],'checkpoint_metadata':checkpoint_meta,
        'manifest_sha256':sha256(OUT/'benchmark/manifest.json'),'source_hashes':source_hashes,'source_panorama_audit':audit,
        'datasets':COUNTS,'total_sets':len(final),'generated_targets_both_tasks':6*len(final),'tasks':TASKS,
        'resolutions_wh':data['resolutions_wh'],'joint_queries':3,'reference_count':2,'inference_steps':20,
        'cfg_scale':1.0,'sigma_shift':5.0,'base_seed':BASE_SEED,'matched_task_noise':True,
        'online_pixel_dropout':False,'training_scene_overlaps':overlaps,'reference_target_overlaps':overlap_counts,
        'panorama_prior':'cached PanoVGGT, two reference images, GT-reference-pose alignment, raw target splats',
        'perspective_prior':'supplied VGGT-Omega, two reference images, GT-reference-pose alignment, distorted OpenCV raw target splats',
        'camera_normalization':'last-query origin, mean camera-center distance scale; same code as training',
        'metrics':['psnr','ssim','lpips_vgg','dreamsim_ensemble','fid_clean_2048'],
        'GT_resize':'PIL bilinear to requested output resolution before every metric; no prediction resize',
        'fid_note':'Dataset-level Clean-FID; pooled category FID computed from pooled features, not averaged per-dataset FIDs. Small correlated test sets, especially Replica 204, limit FID reliability.',
        'strict_HM3D':'exclude 21 target/reference overlaps for extra strict scores; retain original inference groups',
        'code_sha256':code_hashes()}
    write_json(OUT/'provenance.json',protocol,immutable=True)
    write_json(OUT/'benchmark/_SUCCESS',{'passed':True,'time':now(),'sets':len(final)})
    progress('complete',sets=len(final),targets_per_task=3*len(final))

if __name__=='__main__':
    main()
