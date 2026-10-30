"""Protocol tests, plus a CUDA loss regression when GPUs are visible."""
import copy
import sys
import unittest
import numpy as np
import torch
from common import REPO, balanced_quotas
sys.path.insert(0,str(REPO))
from mixed_camera import camera_features, perspective_directions, scaled_intrinsics, distort_normalized, normalize_near_depth_last_query
from diffsynth.core.data.dataset import normalize_w2c_make_cam_last_origin, get_equirectangular_plucker_rays, _camera_to_raymap, _equirectangular_camera_to_raymap, _raymap_to_plucker
from diffsynth.models.wan_video_dit import WanModel
import diffsynth.models.wan_video_dit as dit_code
from diffsynth.pipelines.wan_video import model_fn_wan_video
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from scope_model import install_scope, model_fn_scope
from mixed_model import install_batch_head, target_loss


def synthetic(projection='panorama'):
    poses = torch.eye(4).repeat(5,1,1)
    poses[:,0,3] = torch.arange(5)
    camera = {'width':960,'height':540,'K':[[500.,0.,480.],[0.,510.,270.],[0.,0.,1.]],'dist':[0.,0.,0.,0.]}
    views = [{'w2c':pose.tolist(),'camera':copy.deepcopy(camera)} for pose in poses]
    return {'projection':projection,'height':256 if projection=='panorama' else 288,
            'width':512,'references':views[:2],'targets':views[2:],'near_depth':2.}


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        dit_code.FLASH_ATTN_3_AVAILABLE = dit_code.FLASH_ATTN_2_AVAILABLE = dit_code.SAGE_ATTN_AVAILABLE = False

    def test_balanced_sampling_includes_every_scene(self):
        capacities = {'a':1,'b':100,'c':100,'d':100}
        q = balanced_quotas(capacities,32)
        self.assertEqual(sum(q.values()),32)
        self.assertEqual(q['a'],1)
        self.assertTrue(all(q[s] >= 1 and q[s] <= capacities[s] for s in q))
        self.assertLessEqual(max(q[s] for s in ('b','c','d'))-min(q[s] for s in ('b','c','d')),1)

    def test_panorama_concat_is_unchanged(self):
        record = synthetic()
        poses = torch.tensor([v['w2c'] for v in record['references']+record['targets']])
        c2w,scale = normalize_near_depth_last_query(poses,record['near_depth'])
        actual,s = camera_features(record,'concat')
        expected = get_equirectangular_plucker_rays(c2w,256,512)
        torch.testing.assert_close(actual['raymap'],expected,rtol=0,atol=0)
        self.assertEqual(s,float(scale))

    def test_panorama_scope_is_unchanged(self):
        record = synthetic()
        poses = torch.tensor([v['w2c'] for v in record['references']+record['targets']])
        c2w,_ = normalize_near_depth_last_query(poses,record['near_depth'])
        expected = _raymap_to_plucker(_equirectangular_camera_to_raymap(c2w,16,32)).reshape(-1,6)
        actual,_ = camera_features(record,'scope')
        torch.testing.assert_close(actual['scope_rays'],expected,rtol=0,atol=0)

    def test_perspective_zero_distortion_matches_framecrafter(self):
        record = synthetic('perspective')
        camera = record['references'][0]['camera']
        actual = perspective_directions(camera,288,512)
        K = torch.from_numpy(scaled_intrinsics(camera,288,512)).float().unsqueeze(0)
        expected = _camera_to_raymap(K,torch.eye(4).unsqueeze(0),288,512)[0,...,3:]
        torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-5)

    def test_intrinsics_follow_anisotropic_resize(self):
        camera = synthetic('perspective')['references'][0]['camera']
        K = scaled_intrinsics(camera,288,512)
        np.testing.assert_allclose(K[0],np.array(camera['K'])[0]*512/960)
        np.testing.assert_allclose(K[1],np.array(camera['K'])[1]*288/540)

    def test_distortion_forward_inverse_round_trip(self):
        camera = synthetic('perspective')['references'][0]['camera']
        camera['dist'] = [.037903,-.035025,.000679,.000131]
        directions = perspective_directions(camera,288,512,16).numpy()
        xy = directions[...,:2]/directions[...,2:]
        observed = distort_normalized(xy,camera['dist'])
        K = scaled_intrinsics(camera,288,512)
        y,x = np.meshgrid((np.arange(18)+.5)*16,(np.arange(32)+.5)*16,indexing='ij')
        expected = (np.stack([x,y],-1)-K[:2,2])/np.diag(K)[:2]
        np.testing.assert_allclose(observed,expected,atol=2e-7)

    def test_projection_switch_shapes_and_ray_validity(self):
        for projection,tokens,height in [('panorama',2560,32),('perspective',2880,36)]:
            record = synthetic(projection)
            scope,_ = camera_features(record,'scope')
            rays = scope['scope_rays']
            self.assertEqual(tuple(rays.shape),(tokens,6))
            torch.testing.assert_close(rays[:,:3].norm(dim=-1),torch.ones(tokens))
            torch.testing.assert_close((rays[:,:3]*rays[:,3:]).sum(-1),torch.zeros(tokens),atol=1e-6,rtol=0)
            self.assertTrue((rays[-tokens//5:,3:] == 0).all())
            concat,_ = camera_features(record,'concat')
            self.assertEqual(tuple(concat['raymap'].shape),(5,384,height,64))

    def test_common_world_transform_invariance(self):
        for projection in ('panorama','perspective'):
            a = synthetic(projection)
            b = copy.deepcopy(a)
            transform = torch.tensor([[0.,0.,1.,8.],[0.,1.,0.,-2.],[-1.,0.,0.,3.],[0.,0.,0.,1.]])
            for v in b['references']+b['targets']:
                v['w2c'] = torch.linalg.inv(transform@torch.linalg.inv(torch.tensor(v['w2c']))).tolist()
            ra,_ = camera_features(a,'scope');rb,_ = camera_features(b,'scope')
            torch.testing.assert_close(ra['scope_rays'],rb['scope_rays'],atol=1e-6,rtol=1e-5)

    def test_invalid_projection_rejected(self):
        record = synthetic();record['projection'] = 'unknown'
        with self.assertRaises(ValueError):
            camera_features(record,'scope')

    def test_scene_depth_scales_moments_only(self):
        for projection in ('panorama','perspective'):
            a = synthetic(projection)
            b = copy.deepcopy(a);b['near_depth'] *= 2
            ra,_ = camera_features(a,'scope');rb,scale = camera_features(b,'scope')
            torch.testing.assert_close(ra['scope_rays'][:,:3],rb['scope_rays'][:,:3],rtol=0,atol=0)
            torch.testing.assert_close(ra['scope_rays'][:,3:]/2,rb['scope_rays'][:,3:],rtol=0,atol=0)
            self.assertEqual(scale,4.)
            # This explicitly catches accidentally reapplying baseline normalization.
            self.assertGreater(ra['scope_rays'][:,3:].abs().max(),rb['scope_rays'][:,3:].abs().max())

    def test_pose_and_depth_joint_scale_invariance(self):
        a = synthetic('perspective');b=copy.deepcopy(a)
        for view in b['references']+b['targets']:
            c2w = torch.linalg.inv(torch.tensor(view['w2c']))
            c2w[:3,3] *= 7
            view['w2c'] = torch.linalg.inv(c2w).tolist()
        b['near_depth'] *= 7
        ra,_ = camera_features(a,'scope');rb,_ = camera_features(b,'scope')
        torch.testing.assert_close(ra['scope_rays'],rb['scope_rays'])

    def test_last_query_is_coordinate_frame_not_only_origin(self):
        transform = torch.tensor([[0.,0.,1.,8.],[0.,1.,0.,-2.],[-1.,0.,0.,3.],[0.,0.,0.,1.]])
        poses = torch.eye(4).repeat(5,1,1)
        poses[:,:3,3] = torch.tensor([[1.,2.,3.],[4.,2.,0.],[1.,0.,2.],[0.,0.,1.],[0.,0.,0.]])
        world_poses = transform @ poses
        actual,scale = normalize_near_depth_last_query(torch.linalg.inv(world_poses),2.)
        torch.testing.assert_close(actual[-1],torch.eye(4))
        torch.testing.assert_close(actual[:,:3,3],poses[:,:3,3]/2)
        self.assertEqual(scale,2.)

    def test_pure_rotation_and_invalid_depth(self):
        poses=torch.eye(4).repeat(5,1,1)
        actual,_=normalize_near_depth_last_query(poses,1.2)
        torch.testing.assert_close(actual,poses)
        for bad in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                normalize_near_depth_last_query(poses,bad)
        record=synthetic();del record['near_depth']
        with self.assertRaisesRegex(ValueError,'Missing per-scene near depth'):
            camera_features(record,'scope')

    def test_reference_frames_not_supervised(self):
        p,t = torch.zeros(1,16,5,4,8),torch.zeros(1,16,5,4,8)
        p[:,:,:2] = 10
        self.assertEqual(target_loss(p,t,torch.ones(1)).item(),0.)

    def test_target_loss_preserves_per_sample_weighting(self):
        prediction = torch.ones(2,16,5,4,8,requires_grad=True)
        target = torch.zeros_like(prediction)
        weight = torch.tensor([1.,3.],dtype=torch.float64)
        loss = target_loss(prediction,target,weight)
        self.assertEqual(loss.dtype,torch.float32)
        self.assertEqual(loss.item(),2.)
        loss.backward()
        self.assertEqual(prediction.grad[:,:,:2].count_nonzero().item(),0)
        expected = weight.float()[:,None,None,None,None].expand(2,16,3,4,8)/(16*3*4*8)
        torch.testing.assert_close(prediction.grad[:,:,2:],expected)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA loss regression requires visible GPUs')
    def test_cpu_scheduler_weight_with_cuda_target_loss(self):
        scheduler = FlowMatchScheduler('Wan')
        scheduler.set_timesteps(1000,training=True)
        for device_index in range(torch.cuda.device_count()):
            device = torch.device('cuda',device_index)
            for height in (32,36):
                with self.subTest(device=str(device),latent_height=height):
                    timestep = scheduler.timesteps[250:251].to(device=device,dtype=torch.bfloat16)
                    weight = scheduler.training_weight(timestep)
                    self.assertEqual(weight.device.type,'cpu')
                    prediction = torch.zeros(1,16,5,height,64,device=device,
                                             dtype=torch.bfloat16,requires_grad=True)
                    target = torch.full_like(prediction,2.)
                    loss = target_loss(prediction,target,weight)
                    self.assertEqual(loss.device,device)
                    self.assertEqual(loss.dtype,torch.float32)
                    torch.testing.assert_close(loss.detach().cpu(),4*weight)
                    loss.backward()
                    self.assertTrue(torch.isfinite(prediction.grad).all().item())
                    self.assertEqual(prediction.grad[:,:,:2].count_nonzero().item(),0)
                    expected = (-4*weight/(16*3*height*64)).to(device=device,dtype=torch.bfloat16)
                    torch.testing.assert_close(prediction.grad[:,:,2:],
                                               torch.full_like(prediction.grad[:,:,2:],expected.item()))
                    torch.cuda.synchronize(device)
                    print(f'CUDA loss/backward passed: {device}, latent height {height}',flush=True)

    def test_both_injection_paths_accept_alternating_grids(self):
        # Small spatial grids exercise both projection sizes without a 14B load.
        for architecture in ('scope','concat'):
            dit = WanModel(dim=48,in_dim=36 if architecture=='scope' else 420,ffn_dim=64,
                out_dim=16,text_dim=16,freq_dim=16,eps=1e-6,patch_size=(1,2,2),num_heads=2,
                num_layers=2,has_image_input=False,require_clip_embedding=False)
            install_batch_head(dit)
            if architecture == 'scope':
                install_scope(dit)
            function = model_fn_scope if architecture=='scope' else model_fn_wan_video
            for h in (4,6,4):
                inputs = {'latents':torch.randn(1,16,5,h,8),'y':torch.randn(1,20,5,h,8),
                    'context':torch.randn(1,6,16),'timestep':torch.ones(1)}
                if architecture=='scope':
                    inputs['scope_rays'] = torch.randn(1,5*(h//2)*4,6)
                else:
                    inputs['raymap'] = torch.randn(1,384,5,h,8)
                output = function(dit=dit,**inputs,use_gradient_checkpointing=True)
                self.assertEqual(output.shape,inputs['latents'].shape)
                output.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in dit.parameters() if p.grad is not None))
                dit.zero_grad()


if __name__ == '__main__':
    unittest.main(verbosity=2)
