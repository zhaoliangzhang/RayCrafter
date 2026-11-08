"""Execute immutable original inference and scoring with recorded runtime overrides."""
import argparse
import runpy
from config_resume import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('script', choices=['worker.py', 'score.py', 'aggregate.py', 'test_eval.py'])
    args, extra = parser.parse_known_args()
    s = configure()
    contract = json.loads((RUNTIME / 'launch_contract.json').read_text())
    assert adapter_hashes() == contract['adapter_hashes']
    assert s.code_hashes() == json.loads((OUT / 'provenance.json').read_text())['code_sha256']
    base_write = s.write_json

    def write(path, value, immutable=False):
        path = Path(path)
        if path.name == '_SUCCESS' and path.is_relative_to(OUT / 'bundles'):
            value = dict(value, execution_host='io95', logical_shards=NUM_SHARDS,
                         inference_cache_threshold_gib=VRAM_LIMIT_GIB,
                         resume_contract=str(RUNTIME / 'launch_contract.json'))
        if path.parent == OUT / 'runtime':
            path = RUNTIME / path.name
        base_write(path, value, immutable)

    s.write_json = write
    sys.argv = [str(ORIGINAL / args.script), *extra]
    runpy.run_path(str(ORIGINAL / args.script), run_name='__main__')


if __name__ == '__main__':
    main()
