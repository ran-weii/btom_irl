#! /bin/bash
python train_rail.py \
--filename "hopper-expert-v2.p" \
--cp_path "none" \
--dynamics_path "../../exp/mujoco/dynamics/hopper-medium-expert-v2/03-10-2023 17-31-51" \
--num_samples 100000 \
--norm_obs False \
--norm_rwd False \
--ensemble_dim 7 \
--topk 5 \
--hidden_dim 200 \
--num_hidden 2 \
--activation relu \
--gamma 0.99 \
--beta 0.2 \
--polyak 0.995 \
--tune_beta False \
--clip_lv True \
--residual False \
--state_only False \
--rwd_clip_max 10. \
--adv_clip_max 6. \
--obs_penalty 10. \
--adv_penalty 1. \
--rwd_rollout_steps 100 \
--adv_rollout_steps 5 \
--norm_advantage True \
--update_critic_adv True \
--buffer_size 1000000 \
--d_batch_size 256 \
--a_batch_size 256 \
--rollout_batch_size 10000 \
--rollout_min_steps 5 \
--rollout_max_steps 5 \
--rollout_min_epoch 50 \
--rollout_max_epoch 200 \
--model_retain_epochs 1 \
--real_ratio 0.5 \
--eval_ratio 0.2 \
--m_steps 250 \
--d_steps 100 \
--a_steps 1 \
--lr_d 3e-4 \
--lr_a 3e-4 \
--lr_c 3e-4 \
--lr_m 3e-4 \
--decay "0.000025, 0.00005, 0.000075, 0.0001" \
--grad_clip 600. \
--grad_penalty 1. \
--grad_target 0. \
--env_name "Hopper-v4" \
--pretrain_steps 0 \
--epochs 20 \
--steps_per_epoch 1000 \
--sample_model_every 250 \
--update_model_every 250 \
--update_reward_every 250 \
--update_policy_every 1 \
--cp_every 10 \
--eval_steps 1000 \
--num_eval_eps 5 \
--eval_deterministic True \
--verbose 50 \
--render False \
--save False \
--seed 0