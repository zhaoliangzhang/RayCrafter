"""Fail closed on missing/duplicate outputs; aggregate per-dataset and category."""
import csv
from eval_shared import *
import numpy as np
sys.path.insert(0,str(FID_PACKAGES))
from cleanfid.fid import frechet_distance

def main():
    data=manifest();rows=data['sets'];contract=json.loads((OUT/'provenance.json').read_text())
    assert code_hashes()==contract['code_sha256']
    records=[];arrays={key:[] for key in ('keys','gt','nvs','cleanup')};versions=None
    for rank in range(NUM_SHARDS):
        part=json.loads((OUT/f'metrics/shard_{rank:02d}.json').read_text())
        assert part['rank']==rank and part['num_shards']==NUM_SHARDS
        assert part['manifest_sha256']==contract['manifest_sha256'] and part['checkpoint_sha256']==contract['checkpoint_sha256']
        assert versions is None or versions==part['versions'];versions=part['versions']
        path=OUT/f'metrics/features_shard_{rank:02d}.npz'
        assert sha256(path)==part['feature_sha256']
        with np.load(path,allow_pickle=False) as feature:
            for key in arrays: arrays[key].append(feature[key])
        records.extend(part['records'])
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    expected={f'{row["id"]}:{j}' for row in rows for j in range(3)}
    assert set(arrays['keys'])==expected and len(arrays['keys'])==len(expected)
    expected_records={(row['id'],task,j) for row in rows for task in TASKS for j in range(3)}
    assert len(records)==len(expected_records) and {(r['set_id'],r['task'],r['target_number']) for r in records}==expected_records
    lookup={f'{r["id"]}:{j}':(r,t) for r in rows for j,t in enumerate(r['targets'])}
    masks={dataset:np.array([lookup[k][0]['dataset']==dataset for k in arrays['keys']]) for dataset in COUNTS}
    masks['panorama']=np.array([lookup[k][0]['projection']=='panorama' for k in arrays['keys']])
    masks['HM3D_strict']=masks['HM3D']&np.array([not lookup[k][1]['reference_overlap'] for k in arrays['keys']])
    masks['panorama_strict']=masks['panorama']&np.array([not lookup[k][1]['reference_overlap'] for k in arrays['keys']])
    results={}
    for task in TASKS:
        results[task]={}
        for group,mask in masks.items():
            keys=set(arrays['keys'][mask]);part=[r for r in records if r['task']==task and f'{r["set_id"]}:{r["target_number"]}' in keys]
            assert len(part)==int(mask.sum())
            pred=arrays[task][mask].astype(np.float64);gt=arrays['gt'][mask].astype(np.float64)
            score=float(frechet_distance(pred.mean(0),np.cov(pred,rowvar=False),gt.mean(0),np.cov(gt,rowvar=False)))
            assert np.isfinite(score) and score>=-1e-5
            value={'targets':len(part),'sets':len({r['set_id'] for r in part}),
                **{m:float(np.mean([r[m] for r in part])) for m in METRICS},'fid':max(0.,score)}
            results[task][group]=value
            write_json(OUT/'metrics/summary_partial.json',{'complete':False,'time':now(),'results':results})
            print(json.dumps({'task':task,'group':group,**value}),flush=True)
        results[task]['perspective']=dict(results[task]['DL3DV'])
    for row in rows:
        a=json.loads(completion(row,'nvs').read_text());b=json.loads(completion(row,'cleanup').read_text())
        assert a['noise_sha256']==b['noise_sha256'] and a['seed']==b['seed']==row['seed']
    summary={'complete':True,'time':now(),'checkpoint':str(CHECKPOINT),'checkpoint_sha256':contract['checkpoint_sha256'],
        'manifest_sha256':contract['manifest_sha256'],'results':results,'resolutions_wh':data['resolutions_wh'],
        'reference_views':2,'joint_queries':3,'inference_steps':20,'paired_task_noise_verified':True,
        'metric_versions':versions,'metric_protocol':{'lpips':'VGG','dreamsim':'official ensemble distance',
        'fid':'Clean-FID clean mode, Inception 2048-D, dataset-level','gt_resize':contract['GT_resize']},
        'fid_caveat':contract['fid_note'],'HM3D_strict_definition':contract['strict_HM3D']}
    write_json(OUT/'metrics/summary.json',summary)
    write_json(OUT/'metrics/per_target.json',records)
    with (OUT/'metrics/summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['task','dataset_or_category','sets','targets',*METRICS,'fid']);writer.writeheader()
        for task,groups in results.items():
            for group,value in groups.items(): writer.writerow({'task':task,'dataset_or_category':group,**value})
    write_json(OUT/'_SUCCESS',{'complete':True,'time':now(),'sets':len(rows),'generated_images':6*len(rows)})

if __name__=='__main__':
    main()
