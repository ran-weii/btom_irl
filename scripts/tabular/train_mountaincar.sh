#! /bin/bash
num_traj=50
gamma=0.99
alpha=10.
rollout_steps=50
obs_penalty=10.
epochs=1000
exact=True
save=True

# btom settings
algo=btom
fit_transition=True
fit_reward=True
mle_transition=False
pess_penalty=0.
lr=0.05

# irl settings
# algo=irl
# rollout_steps=200
# fit_transition=False
# fit_reward=True
# mle_transition=True
# pess_penalty=0.
# lr=0.05

# pil settings
# algo=pil
# rollout_steps=200
# fit_transition=False
# fit_reward=True
# mle_transition=True
# pess_penalty=2.
# lr=0.05

python train_mountaincar.py \
--x_bins 20 \
--v_bins 20 \
--num_traj $num_traj \
--gamma $gamma \
--alpha $alpha \
--horizon 0 \
--algo $algo \
--fit_transition $fit_transition \
--fit_reward $fit_reward \
--mle_transition $mle_transition \
--pess_penalty $pess_penalty \
--rollout_steps $rollout_steps \
--exact $exact \
--obs_penalty $obs_penalty \
--lr $lr \
--decay 0. \
--epochs $epochs \
--save $save