#! /bin/bash
num_traj=100
gamma=0.7
rollout_steps=30
obs_penalty=10.
lr=0.05
epochs=1000
exact=True
save=True

# benchmark
python train_gridworld.py \
--init_type one_state \
--goal_type one_goal \
--p_goal 0.95 \
--epsilon 0. \
--num_traj $num_traj \
--gamma $gamma \
--alpha 1. \
--horizon 0 \
--algo btom \
--fit_transition False \
--fit_reward True \
--mle_transition True \
--pess_penalty 0. \
--rollout_steps $rollout_steps \
--exact $exact \
--obs_penalty 0. \
--lr $lr \
--decay 0. \
--epochs $epochs \
--save $save

# btom agents
# python train_gridworld.py \
# --init_type one_state \
# --goal_type one_goal \
# --p_goal 0.95 \
# --epsilon 0. \
# --num_traj $num_traj \
# --gamma $gamma \
# --alpha 1. \
# --horizon 0 \
# --algo btom \
# --fit_transition True \
# --fit_reward True \
# --mle_transition False \
# --pess_penalty 0. \
# --rollout_steps $rollout_steps \
# --exact $exact \
# --obs_penalty 0.001 \
# --lr $lr \
# --decay 0. \
# --epochs $epochs \
# --save $save

# python train_gridworld.py \
# --init_type one_state \
# --goal_type one_goal \
# --p_goal 0.95 \
# --epsilon 0. \
# --num_traj $num_traj \
# --gamma $gamma \
# --alpha 1. \
# --horizon 0 \
# --algo btom \
# --fit_transition True \
# --fit_reward True \
# --mle_transition False \
# --pess_penalty 0. \
# --rollout_steps $rollout_steps \
# --exact $exact \
# --obs_penalty 0.5 \
# --lr $lr \
# --decay 0. \
# --epochs $epochs \
# --save $save

# python train_gridworld.py \
# --init_type one_state \
# --goal_type one_goal \
# --p_goal 0.95 \
# --epsilon 0. \
# --num_traj $num_traj \
# --gamma $gamma \
# --alpha 1. \
# --horizon 0 \
# --algo btom \
# --fit_transition True \
# --fit_reward True \
# --mle_transition False \
# --pess_penalty 0. \
# --rollout_steps $rollout_steps \
# --exact $exact \
# --obs_penalty 10. \
# --lr $lr \
# --decay 0. \
# --epochs $epochs \
# --save $save

# compare all algos

# btom settings
# algo=btom
# fit_transition=True
# fit_reward=True
# mle_transition=False
# pess_penalty=0.
# lr=0.05

# irl settings
# algo=irl
# fit_transition=False
# fit_reward=True
# mle_transition=True
# pess_penalty=0.
# lr=0.1

# pil settings
# algo=pil
# fit_transition=False
# fit_reward=True
# mle_transition=True
# pess_penalty=1.
# lr=0.1

# python train_gridworld.py \
# --init_type one_state \
# --goal_type one_goal \
# --p_goal 0.95 \
# --epsilon 0. \
# --num_traj $num_traj \
# --gamma $gamma \
# --alpha 1. \
# --horizon 0 \
# --algo $algo \
# --fit_transition $fit_transition \
# --fit_reward $fit_reward \
# --mle_transition $mle_transition \
# --pess_penalty $pess_penalty \
# --rollout_steps $rollout_steps \
# --exact $exact \
# --obs_penalty $obs_penalty \
# --lr $lr \
# --decay 0. \
# --epochs $epochs \
# --save $save

# python train_gridworld.py \
# --init_type one_state \
# --goal_type three_goals \
# --p_goal 0.95 \
# --epsilon 0. \
# --num_traj $num_traj \
# --gamma $gamma \
# --alpha 1. \
# --horizon 0 \
# --algo $algo \
# --fit_transition $fit_transition \
# --fit_reward $fit_reward \
# --mle_transition $mle_transition \
# --pess_penalty $pess_penalty \
# --rollout_steps $rollout_steps \
# --exact $exact \
# --obs_penalty $obs_penalty \
# --lr $lr \
# --decay 0. \
# --epochs $epochs \
# --save $save

# python train_gridworld.py \
# --init_type uniform \
# --goal_type three_goals \
# --p_goal 0.95 \
# --epsilon 0. \
# --num_traj $num_traj \
# --gamma $gamma \
# --alpha 1. \
# --horizon 0 \
# --algo $algo \
# --fit_transition $fit_transition \
# --fit_reward $fit_reward \
# --mle_transition $mle_transition \
# --pess_penalty $pess_penalty \
# --rollout_steps $rollout_steps \
# --exact $exact \
# --obs_penalty $obs_penalty \
# --lr $lr \
# --decay 0. \
# --epochs $epochs \
# --save $save