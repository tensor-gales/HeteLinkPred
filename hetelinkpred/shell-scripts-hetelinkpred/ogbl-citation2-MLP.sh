#!/usr/bin/env bash
set -x
{
    python graphsage_wo_exclude_train_target.py \
        --model_name MLPDecoder --dataset="ogbl-citation2" \
        --num_layers 2 \
        --batch_size 32768 --n_epochs 160 --runs 3 --log_steps 8 \
        --checkpoint_folder output/ogbl-citation2/
}
