#!/usr/bin/env bash
set -x
{
    for model in GCN GCN_DistMultS GCN_DOT; do
        python graphsage_wo_exclude_train_target.py \
            --model_name $model --dataset="ogbl-citation2" \
            --num_layers 2 \
            --batch_size 32768 --n_epochs 160 --runs 3 --log_steps 8 \
            --checkpoint_folder output/ogbl-citation2/
    done
}
