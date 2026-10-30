"""Record/check archive content integrity without following external symlinks."""
import argparse
from handoff_common import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true', help='Generate manifest after intentional package edits')
    args = p.parse_args()
    destination = ROOT/'PACKAGE_FILES.json'
    files = {}
    for path in sorted(ROOT.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Package must not contain symlinks: {path}')
        if path.is_file() and path != destination and '__pycache__' not in path.parts:
            files[str(path.relative_to(ROOT))] = {'bytes':path.stat().st_size,'sha256':sha256(path)}
    if args.write:
        write_json(destination, files)
    elif files != read_json(destination):
        raise ValueError('Package content differs from PACKAGE_FILES.json')
    print(f'{len(files)} files, {sum(v["bytes"] for v in files.values())} bytes; integrity OK')


if __name__ == '__main__':
    main()
