#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")

cd "$PROJECT_ROOT"

# 8x B200 GPUs, NCCL env
# export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export CUDA_VISIBLE_DEVICES=6,7
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME='=eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9'
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA='=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7'
export NCCL_NET_GDR_LEVEL=2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_BUFFSIZE=8388608

# Jax and XLA
export JAX_DEFAULT_MATMUL_PRECISION=bfloat16
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python -m experiments.train_gpt2_owt "$@"
