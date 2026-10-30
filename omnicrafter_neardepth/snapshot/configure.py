"""Record explicit model/initialization choices before enabling the queued run."""
import argparse
import json
from safetensors import safe_open
from common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',choices=['scope','concat'],required=True)
    parser.add_argument('--initialization',required=True,help='published or absolute checkpoint path')
    args = parser.parse_args()
    path = PUBLISHED if args.initialization=='published' else Path(args.initialization).resolve(strict=True)
    with safe_open(str(path),framework='pt',device='cpu') as checkpoint:
        keys = set(checkpoint.keys())
        channels = checkpoint.get_slice('patch_embedding.weight').get_shape()[1]
        if args.initialization=='published':
            assert len(keys)==802 and channels==420
            assert sha256(path)==PUBLISHED_SHA
        else:
            expected = 1202 if args.architecture=='scope' else 802
            assert len(keys)==expected
            assert channels==(36 if args.architecture=='scope' else 420)
    request = {'camera_injection':args.architecture,'initialization_kind':'published' if args.initialization=='published' else 'checkpoint',
        'initialization_path':str(path),'initialization_sha256':sha256(path),
        'nvs_probability':.5,'task_sampling':'per sample: all three queries black together',
        'steps_in_this_run':FINAL_STEP,'batch_size_per_gpu':BATCH,'save_steps':SAVE_STEPS,
        'pose_normalization':POSE_NORMALIZATION,'world_size':WORLD_SIZE,
        'global_batch_size':GLOBAL_BATCH,'frozen_encoder_policy':FROZEN_ENCODER_POLICY}
    destination = OUT/'request.json'
    if destination.exists():
        assert json.loads(destination.read_text())==request,'Existing requested configuration differs'
    else:
        write_json(destination,request)
    print(json.dumps(request,indent=2),flush=True)


if __name__ == '__main__':
    main()
