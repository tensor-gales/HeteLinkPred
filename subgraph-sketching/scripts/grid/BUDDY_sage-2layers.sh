#!/usr/bin/env bash

set -x
{
    for i in 0 9 1 8 2 7 3 6 4 5; do
        python -m src.runners.run --dataset_name grid --graph_idx $i \
            --model BUDDY --cache_subgraph_features --num_negs 5 \
            --hidden_channels 256 --sign_k 2 --sign_norm sage \
            --reps 3 --batch_size 1024 --epochs 75 --eval_steps 5 \
            --eval_batch_size 524288 \
            --checkpoint_folder output/subgraph-sketching/
    done
}