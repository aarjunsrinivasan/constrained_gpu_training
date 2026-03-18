# constrained_gpu_training

Training experiments for `Qwen3` on constrained hardware using my `torchtitan` fork as a submodule.

The current target is `Qwen3-1.7B` on `2x RTX 3090 24 GB` with `FSDP2 + torch.compile + activation checkpointing`.

The full notes and measurements are in [my_experiment_flow.md](./my_experiment_flow.md).

## Setup

Initialize the submodule and use the exported conda environment:

```bash
git submodule update --init --recursive
conda env create -f environment.yml
conda activate ptdist
```

The exported environment pins:

- `torch==2.12.0.dev20260307+cu128`
- `torchvision==0.26.0.dev20260307+cu128`
- `triton==3.6.0+git9844da95`

If you do not create the env from [environment.yml](./environment.yml), just make sure you install the torch nightly version.

## Running

Run experiments from [torchtitan](./torchtitan).

The commands used for the experiments are collected in [experiment_commands.sh](./experiment_commands.sh).

Examples:

```bash
./experiment_commands.sh download_assets
./experiment_commands.sh fsdp2_compile
./experiment_commands.sh full_ac
```

## Main Results

For `Qwen3-1.7B` at `seq_len=2048` on `2x RTX 3090`:

| Config | Max Batch | Peak tps | Peak MFU |
| --- | ---: | ---: | ---: |
| FSDP2 | 2 | 1,392 | 19.3% |
| FSDP2 + compile | 4 | 2,585 | 35.9% |
| FSDP2 + compile + Selective AC | 4 | 2,579 | 35.8% |
| FSDP2 + compile + Full AC | 9 | 3,598 | 50.0% |

Current takeaway: full activation checkpointing is the best result so far. On this setup it pushes the local batch ceiling from `4` to `9` and reaches about `50% MFU`.

For `Qwen3-0.6B` on a single RTX 3090, `torch.compile` raised the batch ceiling from `3` to `6`.
