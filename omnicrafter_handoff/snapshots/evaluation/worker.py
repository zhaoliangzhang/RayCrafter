"""GT-blind, paired-seed 2ref3 inference for either supported projection."""
import argparse
import fcntl
import gc
import time
from eval_shared import *
imports()
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from model import FrameCrafter
from mixed_camera import camera_features
from mixed_model import install_batch_head
from diffsynth.core.vram.layers import AutoTorchModule

class MixedCrafter(FrameCrafter):
    @torch.no_grad()
    def generate_render_queries(self,references,queries,raymap,height,width,seed):
        assert len(references)==2 and len(queries)==3
        assert (width,height) in ((512,256),(512,288))
        assert self.input_dim==420 and tuple(raymap.shape)==(5,384,height//8,width//8)
        assert torch.isfinite(raymap).all()
        assert all(im.mode=='RGB' and im.size==(width,height) for im in references+queries)
        video=self.pipe(prompt='',negative_prompt='',input_image=references,conditioning_images=queries,
            raymap=raymap.to(self.device,dtype=torch.bfloat16),height=height,width=width,
            num_frames=5,num_latent_frames=5,cfg_scale=1.0,num_inference_steps=20,
            sigma_shift=5.0,seed=seed,tiled=True)
        assert len(video)==5
        return video[-3:]

def load_inputs(row,task):
    size=(row['width'],row['height'])
    def load(path):
        with Image.open(path) as im:
            return im.convert('RGB').resize(size,Image.Resampling.BILINEAR)
    refs=[load(r['image']) for r in row['references']]
    if task=='nvs':
        queries=[Image.new('RGB',size) for _ in range(3)]
        assert all(im.getbbox() is None for im in queries)
    else:
        assert task=='cleanup'
        queries=[load(t['noisy']) for t in row['targets']]
    return refs,queries

def verify_checkpoint(model,path):
    state=load_file(str(path),device='cpu');covered=set()
    for module in model.pipe.dit.modules():
        if getattr(module,'lora_A_weights',[]):
            assert len(module.lora_A_weights)==len(module.lora_B_weights)==1
            for letter,weights in [('A',module.lora_A_weights),('B',module.lora_B_weights)]:
                key=f'{module.name}.lora_{letter}.default.weight'
                assert torch.equal(weights[0].detach().cpu(),state[key].to(weights[0].dtype)),key
                covered.add(key)
    assert len(covered)==800
    patch=getattr(model.pipe.dit.patch_embedding,'module',model.pipe.dit.patch_embedding)
    for name in ('weight','bias'):
        key='patch_embedding.'+name;value=getattr(patch,name)
        assert torch.equal(value.detach().cpu(),state[key].to(value.dtype)),key
        covered.add(key)
    assert covered==set(state) and len(covered)==802
    del state

FORBIDDEN_GT=set()
def guard_gt(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)) and FORBIDDEN_GT:
        path=os.fsdecode(args[0])
        if path in FORBIDDEN_GT:
            raise RuntimeError(f'Inference attempted to read target RGB: {path}')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    assert 0<=args.rank<NUM_SHARDS
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    contract=json.loads((OUT/'provenance.json').read_text())
    assert code_hashes()==contract['code_sha256']
    assert sha256(CHECKPOINT)==sha256(CACHE/CHECKPOINT.name)==contract['checkpoint_sha256']
    rows=manifest()['sets']
    selected=smoke_rows(rows) if args.smoke else [r for r in rows if r['index']%NUM_SHARDS==args.rank]
    if all(completed(r,t) for r in selected for t in TASKS):
        print('All assigned inference already complete',flush=True)
        return
    with (OUT/'runtime/model_init.lock').open('a') as lock:
        print(f'{now()} rank={args.rank} waiting for CPU model initialization',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        model=MixedCrafter(str(CACHE/CHECKPOINT.name),base_model_dir=str(WAN),vram_limit=32,input_dim=420)
        install_batch_head(model.pipe.dit)
        verify_checkpoint(model,CACHE/CHECKPOINT.name)
        for name in ('dit','vae','text_encoder','image_encoder'):
            getattr(model.pipe,name).eval().requires_grad_(False)
        pinned=0
        for module in model.pipe.dit.modules():
            if isinstance(module,AutoTorchModule):
                module.offload_device=module.onload_device=module.preparing_device=module.computation_device=model.device
                pinned+=1
        assert pinned>0
        write_json(OUT/f'runtime/loaded_rank{args.rank}.json',{'time':now(),'checkpoint_tensors_verified':802,
            'checkpoint_sha256':contract['checkpoint_sha256'],'pinned_dit_modules':pinned})
        gc.collect()
    sys.addaudithook(guard_gt)
    noises=[];calls=[]
    original_noise,original_fn=model.pipe.generate_noise,model.pipe.model_fn
    def noise(*a,**kw):
        value=original_noise(*a,**kw)
        noises.append(hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        return value
    def forward(*a,**kw):
        calls.append(1)
        return original_fn(*a,**kw)
    model.pipe.generate_noise,model.pipe.model_fn=noise,forward
    for row in selected:
        if all(completed(row,t) for t in TASKS):
            continue
        rays,scale=camera_features(row,'concat')
        for task in TASKS:
            if completed(row,task):
                continue
            refs_allowed={r['image'] for r in row['references']}
            FORBIDDEN_GT.update({t[k] for t in row['targets'] for k in ('image','metric_gt')} - refs_allowed)
            try:
                references,queries=load_inputs(row,task)
                noises.clear();calls.clear()
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                generated=model.generate_render_queries(references,queries,rays['raymap'],row['height'],row['width'],row['seed'])
                torch.cuda.synchronize();elapsed=time.monotonic()-start
            finally:
                FORBIDDEN_GT.clear()
            assert len(generated)==3 and len(calls)==20 and len(noises)==1
            assert all(im.size==(row['width'],row['height']) and im.mode=='RGB' for im in generated)
            hashes=[]
            for j,im in enumerate(generated):
                path=prediction(row,task,j);atomic_png(im,path);hashes.append(sha256(path))
            record={'time':now(),'set_id':row['id'],'source_sample':row['source_sample'],'dataset':row['dataset'],
                'task':task,'projection':row['projection'],'height':row['height'],'width':row['width'],
                'seed':row['seed'],'diffusion_steps':20,'dit_calls':len(calls),'cfg_scale':1,'sigma_shift':5,
                'noise_sha256':noises[0],'checkpoint_sha256':contract['checkpoint_sha256'],
                'query_conditions':['all_black']*3 if task=='nvs' else [t['noisy'] for t in row['targets']],
                'query_pixel_sums':[int(np.asarray(im,dtype=np.uint64).sum()) for im in queries],
                'target_rgb_accessed_during_inference':False,'camera_shape':list(rays['raymap'].shape),
                'pose_scale':scale,'seconds':elapsed,'target_fps':3/elapsed,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'prediction_sha256':hashes}
            write_json(completion(row,task),record)
            print(json.dumps(record),flush=True)
        first=json.loads(completion(row,'nvs').read_text());second=json.loads(completion(row,'cleanup').read_text())
        assert first['noise_sha256']==second['noise_sha256'],'NVS/cleanup initial noise differs'
    write_json(OUT/f'runtime/{"inference_smoke" if args.smoke else "inference_done"}_rank{args.rank}.json',
        {'time':now(),'passed':True,'sets':len(selected),'tasks':TASKS})

if __name__=='__main__':
    main()
