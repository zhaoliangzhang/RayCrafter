Resume SCoPE-FF step-100000 evaluation from io107 on io95 GPUs 0-7.

Run on io95 in tmux:
  bash /cis/home/xliu253/Documents/PanoLagerNVS/scheduled_scopeff100k_resume_io95_20261106/run_io95.sh

The controller waits for all eight GPUs to be idle, without a queue timeout.
It does not terminate, suspend, or modify other jobs, including UniRoPE metrics.
CPU-only validation and local weight caching can run while those jobs finish.

This is a runtime adapter. The original evaluation code, provenance, checkpoint,
camera implementation, manifest, seeds, resolutions, and 20-step sampler remain
unchanged. Completed set/task markers and PNGs are hash-verified and skipped.
Only unfinished set/tasks are rendered. All eight shards then compute metrics,
and the original aggregator produces final PSNR, SSIM, LPIPS, DreamSim and FID.
Per-sample reference, ground-truth and noisy-render links remain in place.

All outputs stay in the original evaluation folder:
/cis/net/io81/data/xliu253/omniraycrafter/mixed_concat_scheduled_scope_ff_q3center_50kinit_dl3dv100k_hm3d60k_jhu40k_b1_plus300k_io104/evaluations/step100000_hm3d_replica_jhu_dl3dv_q3_20261105

New logs have prefix logs/io95_resume_. New command records, status, validation,
launch contract and preservation audits are under runtime/resume_io95_20261106.
The original four-shard runtime records and original provenance are retained.
The common controller.lock prevents concurrent io107/io95 controllers.

io95 uses eight independent inference workers with serialized CPU construction
and at least 75 GiB host-RAM headroom. Its inference device weight-cache threshold
is 32 GiB (io107 used 23 GiB). This changes device placement, not weights, BF16
precision, sampling, input data or camera normalization. This is not training.

If the controller fails, inspect its stage log and rerun the same script after
resolving the cause. Completed rendering is skipped; partial metric shards are
recomputed. Never run the old four-shard controller at the same time.
