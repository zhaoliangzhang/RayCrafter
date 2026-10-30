#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" "${PACKAGE_DIR}/render_priors.py" "$@"
