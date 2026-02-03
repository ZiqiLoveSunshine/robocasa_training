#!/bin/bash

# Activate conda environment and run training
source ~/anaconda3/etc/profile.d/conda.sh
conda activate robocasa
python rl_scripts/train_ppo.py --task PnPCounterToCab --max_timesteps 1000000 --n_envs 4
