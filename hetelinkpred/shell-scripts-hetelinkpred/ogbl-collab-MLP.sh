#!/usr/bin/env bash
set -x

python graphsage_wo_exclude_train_target.py \
    --model_name MLPDecoder --dataset="ogbl-collab" \
    --num_layers 2 \
    --batch_size 65536 --n_epochs 640 --runs 3 --log_steps 8 "$@"
