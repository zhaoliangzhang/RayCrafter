#!/usr/bin/env bash
set -euo pipefail
export OC_REQUIRE_RESUME_STEP=28000
export OC_RESERVE_HEADROOM_GIB=16
exec bash /cis/home/xliu253/Documents/PanoLagerNVS/mixed_projection_2ref3_20261023/run_io104.sh
