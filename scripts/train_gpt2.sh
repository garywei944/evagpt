#!/bin/bash

set -ex

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

cd "$PROJECT_DIR"

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3"
export NCCL_IB_GID_INDEX=5
export NCCL_SOCKET_IFNAME="=eth0,eth1,eth2,eth3"
export NCCL_SOCKET_FAMILY=AF_INET6

# unique version as timestamp
run_name="$(date +%Y%m%d_%H%M%S)"
GLOG_log_dir="$PROJECT_DIR/logs/gpt2/$run_name"

mkdir -p "$GLOG_log_dir"

BATCH_SIZE=64
MODEL_NAME=gpt2


python -m experiments.train_gpt2 \
    --model.model_name="$MODEL_NAME" \
    --data.dataset_path="wikitext" \
    --data.dataset_name="wikitext-103-v1" \
    --run_ver="$run_name" \
    --wandb_project="gpt2-wiki103" \
    --epochs=20 \
    --accumulate_grad_batches=1 \
    --data.train_batch_size_per_device="$BATCH_SIZE" \
    --data.eval_batch_size_per_device="$BATCH_SIZE" \
    --strategy=deepspeed_stage_2 \
    --precision=bf16-mixed \
    "$@" |&
    (
        trap '' SIGINT
        tee "$GLOG_log_dir"/run.log
    )
