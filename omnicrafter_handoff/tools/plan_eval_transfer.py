"""Build a deduplicated rsync file list. Does not copy data or contact a host."""
import argparse
from handoff_common import *


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'metadata/eval_manifest.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assets = set()
    for row in read_json(args.manifest)['sets']:
        validate_row(row)
        for obj, key in image_fields(row):
            assets.add(str(safe_absolute(obj[key]).relative_to('/')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        for name in sorted(assets):
            f.write(name.encode() + b'\0')
    print(f'{len(assets)} unique assets; source root is /; use rsync -aL --from0 --files-from=LIST')
    print('Links must be dereferenced (-L); never add --delete. No transfer was executed.')


if __name__ == '__main__':
    main()
