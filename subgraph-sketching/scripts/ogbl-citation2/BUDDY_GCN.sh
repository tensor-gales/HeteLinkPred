#!/usr/bin/env bash

set -x
{
    python -m src.runners.run --dataset ogbl-citation2 \
    --hidden_channels 128 --num_negs 5 --lr 0.0005 --sign_dropout 0.2 \
    --feature_dropout 0.7 --label_dropout 0.8 --batch_size 261424 \
    --reps 3 --eval_batch_size 522848 --cache_subgraph_features \
    --model BUDDY --epochs 75 --eval_steps 5 "$@"
}
