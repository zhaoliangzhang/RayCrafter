# Skipjack / ARCH Slurm notes

Checked against the official ARCH documentation on2026-10-28. These are template
instructions, not proof that a particular allocation/account/storage path is
available to you. No login, file transfer or job submission was performed.

## VS Code is for development; use sbatch for long experiments

The linked VS Code helper creates an interactive Slurm allocation. Its documented
behavior is to release that allocation when the VS Code connection closes, and
the allocation also ends at its time limit. **Do not start a long training run
inside that VS Code allocation and assume tmux will keep it alive after the
allocation is released.** Use a separate `sbatch` job for training/evaluation.
VS Code can then be used to edit code and inspect the batch job's shared logs.
[Source: ARCH VS Code guide](https://docs.arch.jhu.edu/en/latest/1_Clusters/Skipjack/vscode.html).

## Allocation and GPU choice

The example Slurm templates request one node, one scheduler task and four full
GPUs; Accelerate launches the four training ranks itself. `interactive_gpu` is
documented as20GB A100 MIG slices, not the four full GPUs needed for this recipe.
The partitions page lists four H200 NVL141GB GPUs per `h200` node; `h100` contains
both80GB SXM and94GB NVL nodes and supports `--constraint=h100_nvl`. An H200 run is
a sensible first porting target **if your allocation allows it**, but fit, cost
and numerical parity still need measurement. Confirm current resources using
`sinfo` / `scontrol`; do not assume a partition name guarantees access.
[Source: ARCH partitions](https://docs.arch.jhu.edu/en/latest/1_Clusters/Skipjack/2_Slurm/Partitions.html).

## Storage and transfer

ARCH lists100GB home directories for small code/configs, PI-shared scratch at
`/scratch/jhu/<PI>` (default1TB), and requested Project Storage at `/projects/<PI>`.
Our training RGB cache alone is747.2GB decimal, so check **free group quota** before
adding WAN, validation assets, checkpoint history and outputs. Do not put it in
home. Recommended layout example: `/scratch/jhu/<PI>/<USER>/omnicrafter/`, or an
approved `/projects/<PI>/...` directory. ARCH recommends Globus for large moves;
many-small-file transfers may benefit from staging archives. Project Storage is
not a backup, so retain another copy of valuable checkpoints and manifests.
[Source: ARCH storage](https://docs.arch.jhu.edu/en/latest/1_Clusters/Skipjack/1_Resources/Storage.html).

The generic `/scratch/YOUR_USER/...` examples elsewhere in this package are
placeholders. Change them to the **actual PI/project path** above. Do not assume
any `/cis/net` or old `/export/io104a` mount exists on Skipjack. Use the manifest
mapping tools after transfer; do not recreate the old absolute path tree by
modifying system directories.

## Submission and monitoring

Edit `configs/skipjack_train.sbatch` or `skipjack_eval.sbatch`: account, partition,
CPU/RAM request, walltime, environment Python and configuration. They are not
ready to submit unchanged. GPU batch partitions generally have a72-hour maximum;
long experiments need checkpoint-based continuation across allocations. A walltime
expiration does not create a new adapter checkpoint between our save intervals.

Submit from an authorized login session using `sbatch`; inspect with `squeue --me`
and `sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS`. Do not run model
loading, training or full image audits on the login node. ARCH may ask for cost
confirmation on submission. This package deliberately does **not** add
`--comment=accept_cost` or `ACCEPT_COST=1`: review and approve resource costs
yourself. [Source: ARCH job submission](https://docs.arch.jhu.edu/en/latest/1_Clusters/Skipjack/2_Slurm/Job_Submission.html).

The templates inherit Slurm's GPU visibility and do not poll for idle cards.
After successful `sbatch` submission the batch job is independent of your laptop
or editor connection, subject to scheduler limits/cancellation and system health.
Do not set a production run's `--gres` to20GB MIG or add GPU memory parking.

The same training template supports the portable SCoPE variant: point
`TRAIN_CONFIG` to your edited `configs/train_scope.local.json` with
`"architecture": "scope"`. No alternative GPU allocation or old io105 launcher
is needed. See [SCoPE training](MIGRATION.md#scope-training) for initialization
conversion, continuation and smoke checks. The evaluation template remains
concat-only; this training update does not make SCoPE evaluation portable.

## Information still needed

- ARCH username, Slurm account/project and permitted GPU partition.
- PI/project storage directory and available quota, including long-term checkpoint placement.
- Chosen environment/container and tested CUDA/driver compatibility.
- Approved walltime/RAM request and whether the first job starts fresh or continues a specified checkpoint.

Never send passwords, OTPs, SSH private keys or billing credentials with this package.
