"""GT-blind scheduled-camera inference; exact saved gate and third-query gauge."""
import argparse
import fcntl
import functools
import gc
import time
from eval_shared import *
imports()
import torch
import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from model import FrameCrafter
from camera import ordered_cameras,prepare_camera_context
from contract import check_contract,verify_vendor
from wan_camera import install_camera_attention,model_fn
from diffsynth.core.vram.layers import AutoTorchModule
from memory_policy import retain_injected_camera_devices

class MixedCrafter(FrameCrafter):
    def __init__(self,path,**kwargs):
        with safe_open(path,framework='pt',device='cpu') as f: check_contract(f.metadata())
        verify_vendor()
        super().__init__(path,input_dim=420,**kwargs)
        install_camera_attention(self.pipe.dit);self.pipe.model_fn=model_fn
        state=load_file(path,device='cpu')
        for i,block in enumerate(self.pipe.dit.blocks):
            for component in COMPONENTS:
                prefix=f'blocks.{i}.self_attn.{component}.'
                values={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
                module=getattr(block.self_attn,component)
                module.load_state_dict(values,strict=True)
                module.to(device=self.device,dtype=torch.bfloat16)
        with safe_open(path,framework='pt',device='cpu') as f:
            saved=f.metadata()
        assert int(saved['step'])==EXPECTED_STEP
        schedule=json.loads(saved['injection_schedule'])
        assert schedule['weight']==1. and schedule['completed_updates']==EXPECTED_STEP-50000
        self.pipe.dit.injection_strength=float(schedule['weight'])

    @torch.no_grad()
    def generate_render_queries(self,references,queries,cameras,height,width,seed):
        assert len(references)==2 and len(queries)==3
        assert (width,height) in ((512,256),(512,288))
        assert all(im.mode=='RGB' and im.size==(width,height) for im in references+queries)
        original=self.pipe.model_fn
        context=prepare_camera_context(cameras,self.device)
        self.pipe.model_fn=functools.partial(original,camera_context=context)
        try:
            video=self.pipe(prompt='',negative_prompt='',input_image=references,conditioning_images=queries,
                raymap=None,height=height,width=width,num_frames=5,num_latent_frames=5,
                cfg_scale=1.,num_inference_steps=20,sigma_shift=5.,seed=seed,tiled=True)
        finally: self.pipe.model_fn=original
        assert len(video)==5
        return video[-3:]

def load_inputs(row,task):
    size=(row['width'],row['height'])
    def load(path):
        with Image.open(path) as im: return im.convert('RGB').resize(size,Image.Resampling.BILINEAR)
    refs=[load(r['image']) for r in row['references']]
    if task=='nvs': queries=[Image.new('RGB',size) for _ in range(3)]
    else:
        assert task=='cleanup';queries=[load(t['noisy']) for t in row['targets']]
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
    for i,block in enumerate(model.pipe.dit.blocks):
        for component in COMPONENTS:
            for name,value in getattr(block.self_attn,component).state_dict().items():
                key=f'blocks.{i}.self_attn.{component}.{name}'
                assert torch.equal(value.detach().cpu(),state[key].to(value.dtype)),key
                covered.add(key)
    assert covered==set(state) and len(covered)==EXPECTED_KEYS

FORBIDDEN_GT=set()
def guard_gt(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        if os.fsdecode(args[0]) in FORBIDDEN_GT:
            raise RuntimeError(f'Inference attempted to read target RGB: {args[0]}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    assert 0<=args.rank<NUM_SHARDS
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    contract=json.loads((OUT/'provenance.json').read_text())
    assert code_hashes()==contract['code_sha256']
    assert sha256(CHECKPOINT)==sha256(CACHE/CHECKPOINT.name)==contract['checkpoint_sha256']
    rows=manifest()['sets']
    selected=smoke_rows(rows) if args.smoke else [r for r in rows if r['index']%NUM_SHARDS==args.rank]
    if all(completed(r,t) for r in selected for t in TASKS): return
    with (CACHE_ROOT/'model_init.lock').open('a') as lock:
        print(f'{now()} rank={args.rank} waiting for CPU initialization',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        while True:
            available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                               if line.startswith('MemAvailable:')))*1024
            if available>=75*1024**3: break
            print(f'{now()} rank={args.rank} waiting for 75 GiB CPU initialization headroom',flush=True)
            time.sleep(30)
        # Use the four initially idle 32-GiB cards. The threshold controls total
        # device weight caching, not a hard per-process activation-memory cap.
        # Retain serial CPU construction and the 75-GiB available-RAM guard.
        model=MixedCrafter(str(CACHE/CHECKPOINT.name),base_model_dir=str(WAN),vram_limit=VRAM_LIMIT_GIB)
        verify_checkpoint(model,CACHE/CHECKPOINT.name)
        for name in ('dit','vae','text_encoder','image_encoder'): getattr(model.pipe,name).eval().requires_grad_(False)
        # Keep the bounded inference offload policy. The new camera modules and
        # monkey-patched output head need explicit placement after parent onload.
        head=model.pipe.dit.head
        for module in head.modules():
            if isinstance(module,AutoTorchModule):
                module.offload_device=module.onload_device=module.preparing_device=module.computation_device=model.device
        head.to(model.device)
        retain_injected_camera_devices(model.pipe,model.device)
        write_json(OUT/f'runtime/loaded_rank{args.rank}.json',{'time':now(),'checkpoint_tensors_verified':EXPECTED_KEYS,
            'checkpoint_sha256':contract['checkpoint_sha256'],'global_vram_cache_threshold_gib':VRAM_LIMIT_GIB,'dit_channels':420})
        gc.collect()
    sys.addaudithook(guard_gt)
    noises=[];calls=[];original_noise,original_fn=model.pipe.generate_noise,model.pipe.model_fn
    def noise(*a,**kw):
        value=original_noise(*a,**kw)
        noises.append(hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest());return value
    def forward(*a,**kw): calls.append(1);return original_fn(*a,**kw)
    model.pipe.generate_noise,model.pipe.model_fn=noise,forward
    for row in selected:
        if all(completed(row,t) for t in TASKS): continue
        ordered,cameras=ordered_cameras(dict(row,near_depth=near_depth_for(row)))
        assert ordered['references']==row['references']
        for task in TASKS:
            if completed(row,task): continue
            refs_allowed={r['image'] for r in row['references']}
            FORBIDDEN_GT.update({t[k] for t in row['targets'] for k in ('image','metric_gt')}-refs_allowed)
            try:
                references,queries=load_inputs(ordered,task)
                noises.clear();calls.clear()
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                images=model.generate_render_queries(references,queries,cameras,row['height'],row['width'],row['seed'])
                torch.cuda.synchronize();elapsed=time.monotonic()-start
            finally: FORBIDDEN_GT.clear()
            assert len(images)==3 and len(calls)==20 and len(noises)==1
            hashes=[]
            for j,im in enumerate(images):
                assert im.size==(row['width'],row['height']) and im.mode=='RGB'
                path=prediction(row,task,j);atomic_png(im,path);hashes.append(sha256(path))
            for j,ref in enumerate(ordered['references']): link(ref['image'],bundle(row,task)/'predictions/references_model_order'/f'ref_{j}.png')
            record={'time':now(),'set_id':row['id'],'source_sample':row['source_sample'],'dataset':row['dataset'],
                'task':task,'projection':row['projection'],'height':row['height'],'width':row['width'],'seed':row['seed'],
                'diffusion_steps':20,'dit_calls':len(calls),'cfg_scale':1,'sigma_shift':5,'noise_sha256':noises[0],
                'checkpoint_sha256':contract['checkpoint_sha256'],'model':MODEL,'target_rgb_accessed_during_inference':False,
                'injection_weight':model.pipe.dit.injection_strength,
                'query_conditions':['all_black']*3 if task=='nvs' else [t['noisy'] for t in row['targets']],
                'query_pixel_sums':[int(np.asarray(im,dtype=np.uint64).sum()) for im in queries],
                'reference_model_order':[r['image'] for r in ordered['references']],
                'pose_normalization':contract['pose_normalization'],'near_depth':near_depth_for(row),
                'seconds':elapsed,'target_fps':3/elapsed,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'prediction_sha256':hashes}
            historical=json.loads((bundle_at(MATCHED,row,task)/'_SUCCESS').read_text())
            assert historical['seed']==row['seed'] and historical['noise_sha256']==record['noise_sha256']
            write_json(completion(row,task),record);print(json.dumps(record),flush=True)
        a=json.loads(completion(row,'nvs').read_text());b=json.loads(completion(row,'cleanup').read_text())
        assert a['noise_sha256']==b['noise_sha256']
    write_json(OUT/f'runtime/{"inference_smoke" if args.smoke else "inference_done"}_rank{args.rank}.json',
        {'time':now(),'passed':True,'sets':len(selected),'tasks':TASKS})

if __name__=='__main__': main()
