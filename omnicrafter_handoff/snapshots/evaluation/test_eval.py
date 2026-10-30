"""Small checks for variable-resolution generation and GT-blind conditioning."""
import unittest
import importlib.util
from unittest.mock import patch
from eval_shared import *
imports()
import torch
from PIL import Image
from worker import MixedCrafter,load_inputs
from mixed_camera import camera_features
from test_protocol import synthetic
from diffsynth.core.data.dataset import normalize_w2c_make_cam_last_origin,get_equirectangular_plucker_rays

class Pipe:
    def __call__(self,**kw):
        self.kw=kw
        return [Image.new('RGB',(kw['width'],kw['height']),(i,i,i)) for i in range(5)]

class Tests(unittest.TestCase):
    def test_seven_shards_cover_every_original_set_once(self):
        self.assertEqual(NUM_SHARDS,7)
        data=json.loads((MATCHED/'benchmark/manifest.json').read_text())
        rows=data['sets']
        shards=[[r['id'] for r in rows if r['index']%NUM_SHARDS==rank] for rank in range(NUM_SHARDS)]
        flattened=[item for shard in shards for item in shard]
        self.assertEqual(len(flattened),3791)
        self.assertEqual(len(set(flattened)),3791)
        self.assertEqual(set(flattened),{r['id'] for r in rows})

    def test_gpu_check_ignores_gpu7_but_refuses_busy_selected_gpu(self):
        spec=importlib.util.spec_from_file_location('matched_eval_controller',CODE/'run.py')
        controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
        other='GPU-b75ff6d9-e49d-a797-cd09-ee59de580966'
        inventory='\n'.join(f'{uuid}, 0, 0' for uuid in GPUS)+f'\n{other}, 32309, 100\n'
        with patch.object(controller.subprocess,'check_output',side_effect=[inventory,f'123, {other}\n']):
            controller.gpu_check()
        with patch.object(controller.subprocess,'check_output',side_effect=[inventory,f'456, {GPUS[0]}\n']):
            with self.assertRaises(AssertionError):
                controller.gpu_check()

    def test_both_resolutions_and_last_three_outputs(self):
        for height in (256,288):
            model=MixedCrafter.__new__(MixedCrafter);model.pipe=Pipe();model.device='cpu';model.input_dim=420
            refs=[Image.new('RGB',(512,height)) for _ in range(2)]
            queries=[Image.new('RGB',(512,height)) for _ in range(3)]
            result=model.generate_render_queries(refs,queries,torch.zeros(5,384,height//8,64),height,512,123)
            self.assertEqual([im.getpixel((0,0))[0] for im in result],[2,3,4])
            kw=model.pipe.kw
            self.assertEqual(kw['num_inference_steps'],20);self.assertEqual(kw['sigma_shift'],5)
            self.assertEqual(kw['num_latent_frames'],5);self.assertEqual(kw['cfg_scale'],1)
            self.assertEqual(len(kw['input_image']),2);self.assertEqual(len(kw['conditioning_images']),3)

    def test_gt_never_opened_for_conditions(self):
        row={'width':512,'height':288,'references':[{'image':'ref0'},{'image':'ref1'}],
             'targets':[{'image':f'gt{j}','noisy':f'prior{j}'} for j in range(3)]}
        opened=[]
        def fake(path):
            self.assertFalse(str(path).startswith('gt'))
            opened.append(str(path));return Image.new('RGB',(960,540),(16,32,48))
        with patch('worker.Image.open',side_effect=fake):
            refs,queries=load_inputs(row,'nvs')
            self.assertEqual(opened,['ref0','ref1'])
            self.assertTrue(all(im.getbbox() is None for im in queries))
            opened.clear();refs,queries=load_inputs(row,'cleanup')
            self.assertEqual(opened,['ref0','ref1','prior0','prior1','prior2'])
            self.assertTrue(all(im.size==(512,288) for im in refs+queries))

    def test_panorama_rays_exactly_match_previous_evaluation(self):
        row=synthetic();poses=torch.tensor([v['w2c'] for v in row['references']+row['targets']])
        _,c2w,_=normalize_w2c_make_cam_last_origin(poses)
        expected=get_equirectangular_plucker_rays(c2w,256,512)
        actual,_=camera_features(row,'concat')
        torch.testing.assert_close(actual['raymap'],expected,atol=0,rtol=0)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
