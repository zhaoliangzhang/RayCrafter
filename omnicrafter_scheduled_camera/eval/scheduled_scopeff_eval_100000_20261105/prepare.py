"""Preserve full validation rows exactly and verify existing image hashes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import math
import shutil
from eval_shared import *

def main():
    imports()
    from safetensors import safe_open
    from contract import check_contract,verify_vendor,CAMERA_CONTRACT
    verify_vendor()
    for name in ('runtime','logs','metrics','benchmark'):(OUT/name).mkdir(parents=True,exist_ok=True)
    old=json.loads((MATCHED/'provenance.json').read_text())
    assert (MATCHED/'_SUCCESS').is_file()
    source=MATCHED/'benchmark/manifest.json'
    assert sha256(source)==old['manifest_sha256']
    data=json.loads(source.read_text());rows=data['sets']
    assert Counter(r['dataset'] for r in rows)==COUNTS
    assert len(rows)==3791 and len({r['id'] for r in rows})==3791
    assert [r['index'] for r in rows]==list(range(len(rows)))
    with safe_open(str(CHECKPOINT),framework='pt',device='cpu') as f:
        metadata=f.metadata();check_contract(metadata)
        assert int(metadata['step'])==EXPECTED_STEP and len(f.keys())==EXPECTED_KEYS
        assert f.get_slice('patch_embedding.weight').get_shape()==[5120,420,1,2,2]
    schedule=json.loads(metadata['injection_schedule'])
    assert schedule==dict(base_step=50000,completed_updates=50000,ramp_steps=10000,type='linear',weight=1.)
    depth_source=COMBINED/'benchmark/near_depth_resolved.json'
    depth=json.loads(depth_source.read_text())
    for row in rows:
        assert row['seed']==BASE_SEED+row['index']
        assert len(row['references'])==2 and len(row['targets'])==3
        assert (row['width'],row['height'])==(512,288 if row['projection']=='perspective' else 256)
        assert math.isfinite(depth[row['scene_name']]['near_depth']) and depth[row['scene_name']]['near_depth']>0
    for src,dst in [(source,OUT/'benchmark/manifest.json'),(depth_source,OUT/'benchmark/near_depth_resolved.json')]:
        if dst.exists():assert sha256(src)==sha256(dst)
        else:shutil.copy2(src,dst)
    known=json.loads((MATCHED/'input_hashes.json').read_text())
    required={r['image'] for row in rows for r in row['references']}
    required.update(t[k] for row in rows for t in row['targets'] for k in ('image','noisy','metric_gt'))
    assert required<=known.keys()
    expected={p:known[p] for p in sorted(required)}
    def verify(item):
        p,h=item;assert sha256(p)==h,f'Input changed: {p}'
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(verify,expected.items()),1):
            if i%1000==0:
                record=dict(time=now(),stage='verify_inputs',checked=i,total=len(expected))
                print(json.dumps(record),flush=True);write_json(OUT/'runtime/preparation.json',record)
    write_json(OUT/'input_hashes.json',expected,immutable=True)
    prior_hashes={p:sha256(p) for p in sorted({r['prior_metadata'] for r in rows})}
    write_json(OUT/'benchmark/prior_metadata_hashes.json',prior_hashes,immutable=True)
    link(MATCHED/'metric_ground_truth',OUT/'metric_ground_truth')
    for i,row in enumerate(rows,1):
        for task in TASKS:
            b=visual_links(row,task)
            write_json(b/'input_metadata.json',dict(row,task=task,near_depth=near_depth_for(row),
                online_pixel_dropout=False,reference_model_order='unchanged manifest order'),immutable=True)
        if i%250==0:print(json.dumps(dict(time=now(),stage='bundles',sets=i,total=len(rows))),flush=True)
    protocol={k:old[k] for k in ('GT_resize','fid_note','strict_HM3D')}
    protocol.update(checkpoint=str(CHECKPOINT),checkpoint_sha256=sha256(CHECKPOINT),checkpoint_metadata=metadata,
        code_sha256=code_hashes(),manifest_sha256=sha256(OUT/'benchmark/manifest.json'),camera_injection=MODEL,
        camera_contract=CAMERA_CONTRACT,pose_normalization='Third query full SE3 origin for every branch; unchanged mean-distance concat scale; scene-near-depth SCoPE-FF.',
        inference_weight=schedule['weight'],diffusion_steps=20,cfg_scale=1.,sigma_shift=5.,
        matched_evaluation=str(MATCHED),matched_manifest_byte_identical=True,source_manifest_sha256=old['manifest_sha256'],
        near_depth_sha256=sha256(OUT/'benchmark/near_depth_resolved.json'),
        near_depth_note='Same verified scene-near-depth values as prior camera evaluations, including approved reference-only PanoVGGT fallbacks; no target GT RGB/depth.',
        selection='All four datasets; all original samples, indices, seeds, reference/target ordering, poses, and priors retained.',
        tasks=TASKS,server=HOST,num_shards=NUM_SHARDS,gpu_uuids=GPUS,gpu_indices=GPU_INDICES,
        inference_vram_budget_gib=VRAM_LIMIT_GIB,gpu_sharing_authorized=False,
        notes='Scheduled concat+SCoPE-FF checkpoint-specific camera contract; no target GT RGB/depth used for inference. NVS all black; cleanup cached prior without pixel dropout. User confirmed Replica 204 (not 304).')
    write_json(OUT/'provenance.json',protocol,immutable=True)
    write_json(OUT/'benchmark/_SUCCESS',dict(time=now(),passed=True,sets=len(rows),input_files=len(expected)))
    write_json(OUT/'runtime/preparation.json',dict(time=now(),stage='complete',sets=len(rows)))

if __name__=='__main__':main()
