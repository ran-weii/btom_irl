#! /bin/bash
python create_mountaincar_demonstrations.py \
--x_bins 20 \
--v_bins 20 \
--gamma 0.99 \
--alpha 10. \
--horizon 0 \
--num_eps 10 \
--max_steps 200 \
--seed 0 \
--save True
