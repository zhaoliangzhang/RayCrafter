"""CPU correctness/regression tests; no GPU allocation and no training pilot."""
import copy
import math
import types
import unittest
import torch
import bootstrap
from config import VARIANT, injection_weight
from camera import ordered_cameras,prepare_camera_context
from concat_camera import camera_features
from contract import verify_vendor
from wan_camera import install_camera_attention,model_fn,batched_head_forward,optimizer_groups
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video import model_fn_wan_video

def row(projection='panorama',near_depth=2.):
    views=[]
    for i in range(5):
        a=.13*(i+1)
        c=torch.eye(4,dtype=torch.float64)
        c[:3,:3]=torch.tensor([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
        c[:3,3]=torch.tensor([.3*i,.1*(i%2),.2*i])
        views.append({'w2c':torch.linalg.inv(c).tolist(),'image':str(i),'camera':
            {'K':[[45.,0.,32.],[0.,45.,16.],[0.,0.,1.]],'width':64,'height':32,
             'distortion':[0.,0.,0.,0.]}})
    return dict(projection=projection,height=32,width=64,near_depth=near_depth,
                references=views[:2],targets=views[2:])

class ScheduledTests(unittest.TestCase):
    def test_vendor(self):
        verify_vendor()

    def test_schedule(self):
        self.assertEqual([injection_weight(s) for s in [50000,50001,55000,60000,350000]],
                         [0.,.0001,.5,1.,1.])

    def test_camera_center_order_scale(self):
        for projection in ('panorama','perspective'):
            source=row(projection)
            record,cameras=ordered_cameras(source)
            self.assertEqual(record['references'],source['references'])
            self.assertTrue(torch.equal(cameras['raymap'],camera_features(source,'concat')[0]['raymap']))
            context=prepare_camera_context(cameras,'cpu')
            torch.testing.assert_close(context['normalized_c2w'][0,-1],torch.eye(4),atol=2e-6,rtol=0)
            if VARIANT=='scope':
                torch.testing.assert_close(context['normalized_c2w'][0,:,:3,3],cameras['c2w'][:,:3,3]/2)
                rays=context['plucker'].reshape(1,5,-1,6)
                self.assertLess(rays[0,-1,:,3:6].abs().max().item(),1e-6)
            else:
                centers=context['normalized_c2w'][0,:,:3,3]
                self.assertAlmostEqual((centers[0]-centers[1]).norm().item(),1.,places=5)

    def test_endpoints_and_gradients(self):
        torch.manual_seed(42)
        base=WanModel(dim=128,in_dim=420,ffn_dim=192,out_dim=16,text_dim=16,freq_dim=16,
            eps=1e-6,patch_size=(1,2,2),num_heads=1,num_layers=2,has_image_input=False,require_clip_embedding=False)
        base.head.forward=types.MethodType(batched_head_forward,base.head)
        adapted=copy.deepcopy(base)
        install_camera_attention(adapted)
        # Nonzero new tensors prove gate-zero parity is a real bypass, not
        # merely a consequence of zero initialization.
        if VARIANT=='scope':
            with torch.no_grad():
                for b in adapted.blocks:
                    b.self_attn.scope_pe.eq[2].weight.normal_(0,.01)
                    b.self_attn.scope_pe.ek[2].weight.normal_(0,.01)
        for projection in ('panorama','perspective'):
            cameras=prepare_camera_context(ordered_cameras(row(projection))[1],'cpu')
            inputs=dict(latents=torch.randn(1,16,5,4,8),y=torch.randn(1,20,5,4,8),
                        context=torch.randn(1,6,16),timestep=torch.ones(1))
            expected=model_fn_wan_video(base,**inputs,raymap=cameras['raymap'])
            adapted.injection_strength=0.
            rng=torch.random.get_rng_state().clone()
            actual=model_fn(adapted,**inputs,camera_context=cameras)
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            for strength in (.0001,.2,1.):
                adapted.zero_grad(set_to_none=True)
                adapted.injection_strength=strength
                loss=model_fn(adapted,**inputs,camera_context=cameras,use_gradient_checkpointing=True).square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                new=[p for n,p in adapted.named_parameters() if '.scope_pe.' in n or '.rayrope_depth.' in n]
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in new))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in new))
        if VARIANT=='rayrope':
            self.assertEqual(optimizer_groups(adapted)[1]['lr'],4e-4)
        else:
            self.assertEqual(adapted.blocks[0].self_attn.scope_pe.eq[0].in_features,7)

    def test_rayrope_gauge_is_q3(self):
        if VARIANT!='rayrope':
            self.skipTest('RayRoPE only')
        # Moving the world origin from ref1 to q3 must leave every camera-to-camera
        # transform, and so RayRoPE's per-camera coordinates, unchanged.
        from vendor.raycrafter_camera.rayrope import prepare_geometry
        for projection in ('panorama','perspective'):
            cameras=ordered_cameras(row(projection))[1]
            moved=prepare_camera_context(cameras,'cpu')['geom']
            plain=prepare_geometry(cameras['c2w'][None],cameras['fxfycxcy'][None],cameras['image_hw'],
                                   cameras['latent_hw'],camera_model=cameras['camera_model'])
            pairs=lambda g,a,b: g[a][:,:,None]@g[b][:,None]
            self.assertFalse(torch.allclose(moved['c2w'],plain['c2w']))
            torch.testing.assert_close(pairs(moved,'w2c','c2w'),pairs(plain,'w2c','c2w'),atol=1e-5,rtol=0)
            torch.testing.assert_close(pairs(moved,'projection','projection_inv'),
                                       pairs(plain,'projection','projection_inv'),atol=1e-5,rtol=0)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
