#! /bin/bash
python train_dynamics_offline.py \
--filename "hopper-medium-expert-v2.p" \
--ensemble_dim 7 \
--hidden_dim 128 \
--num_hidden 2 \
--activation relu \
--clip_lv False \
--num_samples 100000 \
--eval_ratio 0.2 \
--batch_size 200 \
--lr 1e-3 \
--decay "0.000025, 0.00005, 0.000075, 0.0001" \
--grad_clip 100. \
--epochs 200 \
--cp_every 10 \
--verbose True \
--save False \
--seed 0