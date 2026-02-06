#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate robocasa

# Run the training script
# Use PYTHONPATH to ensure local env module is accessible
# but robosuite package is imported from site-packages
cd /home/ziqi/robocasa_training
export PYTHONPATH="/home/ziqi/robocasa_training:$PYTHONPATH"

# Temporarily rename local robosuite to prevent import conflicts
if [ -d "robosuite" ]; then
    mv robosuite _robosuite_local
    trap "mv _robosuite_local robosuite" EXIT
fi

python rl_scripts/train_ppo.py --task PnPCounterToCab --max_timesteps 3000000 --n_envs 8 --headless
