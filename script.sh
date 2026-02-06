python rl_scripts/train_ppo.py --task PnPCounterToCab --max_timesteps 3000000  --n_envs 8 --headless

python rl_scripts/train_ppo.py --task TurnOnMicrowave --max_timesteps 3000000  --n_envs 1 --headless

python rl_scripts/eval_ppo.py --task PnPCounterToCab --model_path models/PnPCounterToCab_ppo_20260203_111846/final_model --save_video