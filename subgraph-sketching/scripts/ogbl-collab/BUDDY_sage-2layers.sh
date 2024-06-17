#!/usr/bin/env bash

#python runners/run.py --dataset_name ogbl-collab --K 50 \
    # --lr 0.02 --feature_dropout 0.05 --add_normed_features 1 \
    # --cache_subgraph_features --label_dropout 0.1 --year 2007 --model BUDDY

set -x
{
    python -m src.runners.run --dataset_name ogbl-collab --K 50 \
    --lr 0.02 --feature_dropout 0.05 \
    --hidden_channels 256 --sign_k 2 --sign_norm sage \
    --cache_subgraph_features --label_dropout 0.1 \
    --year 2007 --model BUDDY \
    --reps 3 --checkpoint_folder gnn-output/ogbl-collab/subgraph-sketching/
}
