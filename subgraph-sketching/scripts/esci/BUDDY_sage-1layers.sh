#!/usr/bin/env bash

set -x
{
    python -m src.runners.run  --dataset esci \
        --hidden_channels 256 --num_negs 5 --lr 0.0005 --sign_dropout 0.2 \
        --feature_dropout 0.7 --label_dropout 0.8 \
        --sign_k 1 --sign_norm sage \
        --reps 3 --batch_size 261424 --epochs 200 --eval_steps 20 \
        --eval_batch_size 261424 --cache_subgraph_features \
        --model BUDDY "$@"
}