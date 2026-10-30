"""Per-target historical metrics and 2048-D Clean-FID features, at output size."""
import argparse
import importlib.metadata
import math
from eval_shared import *
sys.path[:0]=[str(TOOLS),str(FID_PACKAGES)]
import numpy as np
import torch
from PIL import Image
import lpips
from dreamsim import dreamsim
from cleanfid import fid
from cleanfid.inception_torchscript import InceptionV3W
from score_predictions import load_rgb,scalar_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.device_count()==1
    assert 0<=args.rank<NUM_SHARDS
    data=manifest();rows=smoke_rows(data['sets']) if args.smoke else [r for r in data['sets'] if r['index']%NUM_SHARDS==args.rank]
    suffix='smoke' if args.smoke else f'shard_{args.rank:02d}'
    destination=OUT/f'metrics/{suffix}.json';feature_path=OUT/f'metrics/features_{suffix}.npz'
    contract=json.loads((OUT/'provenance.json').read_text())
    assert code_hashes()==contract['code_sha256']
    if destination.exists() and feature_path.exists():
        old=json.loads(destination.read_text())
        assert old['manifest_sha256']==contract['manifest_sha256'] and old['checkpoint_sha256']==contract['checkpoint_sha256']
        assert len(old['records'])==len(rows)*6
        return
    lpips_model=lpips.LPIPS(net='vgg').eval().to('cuda')
    dream_model,preprocess=dreamsim(pretrained=True,device='cuda',cache_dir=str(DREAM_CACHE),dreamsim_type='ensemble')
    dream_model.eval()
    inception=InceptionV3W(str(CACHE/'fid'),download=False,resize_inside=False).eval().to('cuda')
    versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','dreamsim','lpips','scikit-image','numpy','timm','peft','transformers','clean-fid')}
    def perceptual(pred_paths,gt_paths,size):
        arrays=[];dp=[];dg=[]
        for pred,gt in zip(pred_paths,gt_paths):
            arrays.append((load_rgb(pred,size,resize=False),load_rgb(gt,size,resize=False)))
            with Image.open(pred) as im: dp.append(preprocess(im.convert('RGB')))
            with Image.open(gt) as im: dg.append(preprocess(im.convert('RGB')))
        with torch.inference_mode():
            pred_tensor=torch.stack([torch.from_numpy(a.transpose(2,0,1)) for a,b in arrays]).to('cuda')
            gt_tensor=torch.stack([torch.from_numpy(b.transpose(2,0,1)) for a,b in arrays]).to('cuda')
            lp=lpips_model(gt_tensor,pred_tensor,normalize=True).flatten().cpu().tolist()
            dr=dream_model(torch.cat(dp).to('cuda'),torch.cat(dg).to('cuda')).flatten().cpu().tolist()
        assert len(lp)==len(dr)==len(arrays)
        return [{**scalar_metrics(a,b),'lpips':float(l),'dreamsim':float(d)} for (a,b),l,d in zip(arrays,lp,dr)]
    def features(paths):
        with torch.inference_mode():
            value=fid.get_files_features([str(p) for p in paths],inception,num_workers=0,batch_size=3,
                device=torch.device('cuda'),mode='clean',verbose=False)
        assert value.shape==(len(paths),2048) and np.isfinite(value).all()
        return value.astype(np.float32)
    if args.smoke:
        # Regression against the exact existing metric protocol on cached priors.
        row=rows[0]
        reference=json.loads((PREVIOUS/'metrics/panovggt_raw.json').read_text())
        lookup={(r['set_id'],r['target_number']):r for r in reference['records']}
        raw=perceptual([Path(t['noisy']) for t in row['targets']],
                       [Path(t['metric_gt']) for t in row['targets']],(512,256))
        errors={m:max(abs(v[m]-lookup[row['source_id'],j][m]) for j,v in enumerate(raw)) for m in METRICS}
        assert max(errors.values())<1e-4,errors
        identity=float(fid.frechet_distance(np.zeros(8),np.eye(8),np.zeros(8),np.eye(8)))
        assert abs(identity)<1e-8
        write_json(OUT/'runtime/metric_regression.json',{'passed':True,'time':now(),'errors':errors,
            'fid_identity':identity,'versions':versions,'fid_weights_sha256':sha256(CACHE/'fid/inception-2015-12-05.pt')})
    records=[];keys=[];gt_features=[];pred_features={t:[] for t in TASKS}
    for i,row in enumerate(rows,1):
        size=(row['width'],row['height'])
        gt_paths=[Path(t['metric_gt']) for t in row['targets']]
        gt_features.append(features(gt_paths))
        keys.extend([f'{row["id"]}:{j}' for j in range(3)])
        paired=[]
        for task in TASKS:
            assert completed(row,task),(row['id'],task)
            record=json.loads(completion(row,task).read_text())
            assert record['checkpoint_sha256']==contract['checkpoint_sha256']
            assert record['query_conditions']==(['all_black']*3 if task=='nvs' else [t['noisy'] for t in row['targets']])
            paired.append(record['noise_sha256'])
            paths=[prediction(row,task,j) for j in range(3)]
            assert [sha256(path) for path in paths]==record['prediction_sha256']
            values=perceptual(paths,gt_paths,size)
            pred_features[task].append(features(paths))
            for j,(target,value) in enumerate(zip(row['targets'],values)):
                assert all(math.isfinite(value[m]) for m in METRICS)
                records.append({'set_id':row['id'],'dataset':row['dataset'],'projection':row['projection'],
                    'task':task,'target_number':j,'prediction':str(paths[j]),'target':str(gt_paths[j]),
                    'reference_overlap':target['reference_overlap'],**{m:value[m] for m in METRICS}})
        assert len(set(paired))==1
        if i%10==0 or i==len(rows):
            print(json.dumps({'time':now(),'rank':args.rank,'scored_sets':i,'total':len(rows)}),flush=True)
    feature_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=feature_path.with_name(f'.{feature_path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,keys=np.asarray(keys),gt=np.concatenate(gt_features),
                            nvs=np.concatenate(pred_features['nvs']),cleanup=np.concatenate(pred_features['cleanup']))
    temporary.replace(feature_path)
    write_json(destination,{'records':records,'rank':args.rank,'num_shards':NUM_SHARDS,'sets':len(rows),
        'manifest_sha256':contract['manifest_sha256'],'checkpoint_sha256':contract['checkpoint_sha256'],
        'versions':versions,'feature_sha256':sha256(feature_path),'fid_mode':'clean','fid_dim':2048,
        'fid_weights_sha256':sha256(CACHE/'fid/inception-2015-12-05.pt')})

if __name__=='__main__':
    main()
