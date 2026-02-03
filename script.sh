tensorboard --logdir /home/philippe/Documents/robocasa_training/runs/robocasa/PnPCounterToCab_visual/


python train.py --env_name PnPCounterToCab --max_timesteps 500000 --headless --num_envs 2

python rl_scripts/train_ppo.py --task PnPCounterToCab --max_timesteps 1000000  --n_envs 2