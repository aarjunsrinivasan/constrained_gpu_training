#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_DIR="${ROOT_DIR}/torchtitan"

usage() {
  cat <<'EOF'
Usage: ./experiment_commands.sh <target>

Targets:
  download_assets
  qwen3_0_6b_base
  qwen3_0_6b_compile
  qwen3_1_7b_single_gpu_oom
  fsdp2_base
  fsdp2_compile
  selective_ac_layer2
  selective_ac_op
  full_ac
  profile_trace
  profile_memory
  accum128
  seq4096
EOF
}

run_in_torchtitan() {
  cd "${TT_DIR}"
  "$@"
}

target="${1:-}"

case "${target}" in
  download_assets)
    run_in_torchtitan python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --assets tokenizer
    run_in_torchtitan python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B --assets tokenizer
    ;;

  qwen3_0_6b_base)
    run_in_torchtitan env NGPU=1 MODULE=qwen3 CONFIG=qwen3_0_6b ./run_train.sh \
      --training.steps 50 --training.local_batch_size 3 --training.seq_len 2048 \
      --metrics.log_freq 5 --metrics.enable_tensorboard \
      --metrics.save_tb_folder outputs/tb/0_6b_1gpu_base
    ;;

  qwen3_0_6b_compile)
    run_in_torchtitan env NGPU=1 MODULE=qwen3 CONFIG=qwen3_0_6b ./run_train.sh \
      --training.steps 50 --training.local_batch_size 6 --training.seq_len 2048 \
      --metrics.log_freq 5 --metrics.enable_tensorboard \
      --metrics.save_tb_folder outputs/tb/0_6b_1gpu_compile \
      --compile.enable
    ;;

  qwen3_1_7b_single_gpu_oom)
    run_in_torchtitan env NGPU=1 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 10 --training.local_batch_size 1 --training.seq_len 256 \
      --compile.enable
    ;;

  fsdp2_base)
    run_in_torchtitan env NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 50 --training.local_batch_size 2 --training.seq_len 2048 \
      --metrics.log_freq 5 --metrics.enable_tensorboard \
      --metrics.save_tb_folder outputs/tb/1_7b_fsdp2_base
    ;;

  fsdp2_compile)
    run_in_torchtitan env NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 50 --training.local_batch_size 4 --training.seq_len 2048 \
      --compile.enable \
      --metrics.log_freq 5 --metrics.enable_tensorboard \
      --metrics.save_tb_folder outputs/tb/1_7b_fsdp2_compile
    ;;

  selective_ac_layer2)
    run_in_torchtitan env NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 50 --training.local_batch_size 3 --training.seq_len 2048 \
      --compile.enable \
      --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option 2 \
      --metrics.log_freq 5 --metrics.enable_tensorboard \
      --metrics.save_tb_folder outputs/tb/1_7b_selective_ac_layer2
    ;;

  selective_ac_op)
    run_in_torchtitan env NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 50 --training.local_batch_size 4 --training.seq_len 2048 \
      --compile.enable \
      --activation_checkpoint.mode selective --activation_checkpoint.selective_ac_option op \
      --metrics.log_freq 5 --metrics.enable_tensorboard \
      --metrics.save_tb_folder outputs/tb/1_7b_selective_ac_op
    ;;

  full_ac)
    run_in_torchtitan env NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 50 --training.local_batch_size 9 --training.seq_len 2048 \
      --compile.enable \
      --activation_checkpoint.mode full \
      --metrics.log_freq 5 --metrics.enable_tensorboard \
      --metrics.save_tb_folder outputs/tb/1_7b_full_ac_b9
    ;;

  profile_trace)
    run_in_torchtitan env NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 20 --training.local_batch_size 4 --training.seq_len 2048 \
      --compile.enable --activation_checkpoint.mode full \
      --profiling.enable_profiling \
      --profiling.profile_freq 10 \
      --profiling.save_traces_folder outputs/traces/1_7b_full_ac_b4
    ;;

  profile_memory)
    run_in_torchtitan env NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 20 --training.local_batch_size 4 --training.seq_len 2048 \
      --compile.enable --activation_checkpoint.mode full \
      --profiling.enable_memory_snapshot \
      --profiling.save_memory_snapshot_folder outputs/mem_snapshots/1_7b_full_ac_b4
    ;;

  accum128)
    run_in_torchtitan env NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 100 --training.local_batch_size 4 --training.seq_len 2048 \
      --training.global_batch_size 128 \
      --compile.enable --activation_checkpoint.mode full \
      --metrics.log_freq 5 --metrics.enable_tensorboard \
      --metrics.save_tb_folder outputs/tb/1_7b_full_ac_accum128
    ;;

  seq4096)
    run_in_torchtitan env NGPU=2 MODULE=qwen3 CONFIG=qwen3_1_7b ./run_train.sh \
      --training.steps 50 --training.local_batch_size 4 --training.seq_len 4096 \
      --compile.enable --activation_checkpoint.mode full \
      --metrics.log_freq 5 --metrics.enable_tensorboard \
      --metrics.save_tb_folder outputs/tb/1_7b_full_ac_seq4096_b4
    ;;

  *)
    usage
    exit 1
    ;;
esac
