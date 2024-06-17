#!/usr/bin/env bash
set -x
{
    for i in 0 9 1 8 2 7 3 6 4 5; do
        python graphsage_wo_exclude_train_target.py \
            --model_name SAGE_DistMultS --dataset=grid-dense --grid_idx $i \
            --num_layers 2 \
            --batch_size 65536 --n_epochs 500 --runs 3
    done
}