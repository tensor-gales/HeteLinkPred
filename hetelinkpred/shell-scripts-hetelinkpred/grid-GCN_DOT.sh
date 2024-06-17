#!/usr/bin/env bash
set -x
{
    for model in GCN_DOT; do
        for i in 0 9 1 8 2 7 3 6 4 5; do
            for deg_threshold in -0.5 0; do
                python graphsage_wo_exclude_train_target.py \
                    --model_name $model --dataset=grid-dense --grid_idx $i \
                    --num_layers 2 \
                    --batch_size 65536 --n_epochs 1200 --runs 3 \
                    --exclude_target_degree $deg_threshold
            done
        done
    done
}
