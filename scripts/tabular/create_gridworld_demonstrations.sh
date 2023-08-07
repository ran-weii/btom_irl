#! /bin/bash
python create_gridworld_demonstrations.py \
--num_grids 5 \
--init_type one_state \
--goal_type one_goal \
--p_goal 0.3333333333 \
--epsilon 0. \
--gamma 0.7 \
--alpha 1. \
--horizon 0 \
--num_eps 300 \
--max_steps 50 \
--num_workers 2 \
--seed 0 \
--save True

# python create_gridworld_demonstrations.py \
# --num_grids 5 \
# --init_type one_state \
# --goal_type three_goals \
# --p_goal 0.95 \
# --epsilon 0. \
# --gamma 0.9 \
# --alpha 1. \
# --horizon 0 \
# --num_eps 300 \
# --max_steps 50 \
# --num_workers 2 \
# --seed 0 \
# --save True

# python create_gridworld_demonstrations.py \
# --num_grids 5 \
# --init_type uniform \
# --goal_type three_goals \
# --p_goal 0.3333333333 \
# --epsilon 0. \
# --gamma 0.9 \
# --alpha 1. \
# --horizon 0 \
# --num_eps 300 \
# --max_steps 50 \
# --num_workers 2 \
# --seed 0 \
# --save True