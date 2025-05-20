#!/bin/bash

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

eval "$(conda shell.bash hook)"
conda activate JaxGCRL

method=exploration-ppo
env=sparse_hopper

#for seed in 1 2 3 4 5 ; do
  #XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
  #JAX_DEBUG_NANS=True
  JAX_ENABLE_X64=True python run.py "$method" --log_wandb --env "$env" --exploration-bonus-type "rnk" #--normalize-bonus
# done

echo "All runs have finished."
