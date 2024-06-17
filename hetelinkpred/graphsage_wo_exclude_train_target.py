import torch
from utils import Logger, to_bidirected_with_reverse_mapping, load_usair_dataset, load_esci_dataset, \
    remove_collab_dissimilar_edges, load_grid_dataset, load_dense_grid_dataset, debug_attach
import torch.nn.functional as F
import dgl
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler, as_edge_prediction_sampler, \
    negative_sampler
import tqdm
from args import get_args, get_output_name
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from models import SAGE, SAGE_DOT, SAGE_DistMultS, GCN, GATv2, GIN, \
    GCN_DOT, GCN_DistMultS, MLPDecoder, DistMultSDecoder
import os
import numpy as np
from sklearn import metrics
import pickle
from low_degree_edge_sampler import EdgePredictionSamplerwithDegree
from random_edge_sampler import EdgePredictionSamplerwithRandom
from bipartite_edge_sampler import BipartiteEdgePredictionSampler
import traceback
import random
import pandas as pd
import gzip
import concurrent.futures
import gc
from pathlib import Path
import signal
import warnings

args = get_args()

def set_random_seed(seed):
    print(f"Setting random seed to {seed}...")
    random.seed(int(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


class DetailEvaluator(Evaluator):
    def __init__(self, name):
        super().__init__(name)
        self.detailed_hits = None

    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):
        '''
            compute Hits@K
            For each positive target node, the negative target nodes are the same.

            y_pred_neg is an array.
            rank y_pred_pos[i] against y_pred_neg for each i
        '''

        if len(y_pred_neg) < self.K:
            return {'hits@{}'.format(self.K): 1.}

        if type_info == 'torch':
            kth_score_in_negative_edges = torch.topk(y_pred_neg, self.K)[0][-1]
            _comp = y_pred_pos > kth_score_in_negative_edges
            hitsK = float(torch.sum(_comp).cpu()) / len(y_pred_pos)
            self.detailed_hits = _comp.cpu()

        # type_info is numpy
        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            self.detailed_hits = y_pred_pos > kth_score_in_negative_edges
            hitsK = float(np.sum(self.detailed_hits)) / len(y_pred_pos)
            self.detailed_hits = torch.as_tensor(self.detailed_hits)

        return {'hits@{}'.format(self.K): hitsK}

def compute_auc(model, node_emb, pos_edge, neg_edge, device, batch_size=500):
    """Compute AUC in batches.
    """
    pos_preds = []
    for start in tqdm.trange(0, pos_edge.size(0), batch_size, desc='Evaluate'):
        end = min(start + batch_size, pos_edge.size(0))
        edge = pos_edge[start:end].t()
        h_src = node_emb[edge[0]].to(device)
        h_dst = node_emb[edge[1]].to(device)
        pos_preds += [model.predictor(h_src * h_dst).squeeze(-1)]
    pos_scores = np.asarray(torch.cat(pos_preds, dim=0).cpu())

    neg_preds = []
    for start in tqdm.trange(0, neg_edge.size(0), batch_size, desc='Evaluate'):
        end = min(start + batch_size, neg_edge.size(0))
        edge = neg_edge[start:end].t()
        h_src = node_emb[edge[0]].to(device)
        h_dst = node_emb[edge[1]].to(device)
        neg_preds += [model.predictor(h_src * h_dst).squeeze(-1)]
    neg_scores = np.asarray(torch.cat(neg_preds, dim=0).cpu())

    # print("AUC negative score: ")
    # print(neg_scores[: 10])
    # print("AUC positive score: ")
    # print(pos_scores[: 10])

    scores = np.concatenate([pos_scores, neg_scores])
    # scores = 1/(1 + np.exp(-scores))
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)

    auc = metrics.auc(fpr, tpr)
    return torch.tensor(auc)


def compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device, 
                batch_size=500, detailed_results=False):
    """Compute Mean Reciprocal Rank (MRR) in batches."""
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = model.predictor(h_src * h_dst).squeeze(-1)
        input_dict = {'y_pred_pos': pred[:, 0], 'y_pred_neg': pred[:, 1:]}
        rr[start:end] = evaluator.eval(input_dict)['mrr_list']
    if detailed_results:
        results_df = pd.DataFrame({
            "src": src.cpu().numpy(),
            "dst": dst.cpu().numpy(),
            "MRR": rr.cpu().numpy()
        })    
        return rr.mean(), results_df
    else:
        return rr.mean()


# def compute_mrr_esci(model, node_emb, src, dst, neg_dst, device, batch_size=500):
#     """Compute Mean Reciprocal Rank (MRR) in batches in esci dataset."""
#     rr = torch.zeros(src.shape[0])
#     for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):
#         end = min(start + batch_size, src.shape[0])
#         all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
#         h_src = node_emb[src[start:end]][:, None, :].to(device)
#         h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
#         pred = model.predictor(h_src * h_dst).squeeze(-1)
#         y_pred_pos = pred[:, 0]
#         y_pred_neg = pred[:, 1:]
#         y_pred_pos = y_pred_pos.view(-1, 1)
#         # optimistic rank: "how many negatives have at least the positive score?"
#         # ~> the positive is ranked first among those with equal score
#         optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
#         # pessimistic rank: "how many negatives have a larger score than the positive?"
#         # ~> the positive is ranked last among those with equal score
#         pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
#         ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
#         mrr_list = 1. / ranking_list.to(torch.float)
#         rr[start:end] = mrr_list
#     return rr.mean()


def compute_mrr_esci(model, node_emb, src, dst, neg_dst, device, 
                     batch_size=500, preload_node_emb=True, detailed_results=False):
    """Compute Mean Reciprocal Rank (MRR) in batches in esci dataset."""

    # gpu may be out of memory for large datasets
    if preload_node_emb:
        node_emb = node_emb.to(device)

    rr = torch.zeros(src.shape[0])
    hits_at_10 = torch.zeros(src.shape[0])
    hits_at_1 = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = model.predictor(h_src * h_dst).squeeze(-1)
        y_pred_pos = pred[:, 0]
        y_pred_neg = pred[:, 1:]
        y_pred_pos = y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hits_at_10[start:end] = ranking_list <= 10
        hits_at_1[start:end] = ranking_list <= 1
        mrr_list = 1. / ranking_list.to(torch.float)
        rr[start:end] = mrr_list
    MRR = rr.mean()
    Hits_10 = hits_at_10.sum() / src.shape[0]
    Hits_1 = hits_at_1.sum() / src.shape[0]

    if not detailed_results:
        return MRR, Hits_10, Hits_1
    else:
        results_df = pd.DataFrame({
            "src": src.cpu().numpy(),
            "dst": dst.cpu().numpy(),
            "MRR": rr.cpu().numpy(),
            "Hits@10": hits_at_10.cpu().numpy(),
            "Hits@1": hits_at_1.cpu().numpy()
        })    
        return MRR, Hits_10, Hits_1, results_df


def compute_hits_esci(model, node_emb, pos_edge, neg_edge, device, batch_size=500):
    """Compute hits@50 in batches in esci."""
    pos_preds = []
    for start in tqdm.trange(0, pos_edge.size(0), batch_size, desc='Evaluate'):
        end = min(start + batch_size, pos_edge.size(0))
        edge = pos_edge[start:end].t()
        h_src = node_emb[edge[0]].to(device)
        h_dst = node_emb[edge[1]].to(device)
        pos_preds += [model.predictor(h_src * h_dst).squeeze(-1)]
    pos_pred = torch.cat(pos_preds, dim=0)

    neg_preds = []
    for start in tqdm.trange(0, neg_edge.size(0), batch_size, desc='Evaluate'):
        end = min(start + batch_size, neg_edge.size(0))
        edge = neg_edge[start:end].t()
        h_src = node_emb[edge[0]].to(device)
        h_dst = node_emb[edge[1]].to(device)
        neg_preds += [model.predictor(h_src * h_dst).squeeze(-1)]
    neg_pred = torch.cat(neg_preds, dim=0)

    y_pred_pos = pos_pred
    y_pred_neg = neg_pred

    # print("Hits negative score: ")
    # print(y_pred_neg[: 10])
    # print("Hits positive score: ")
    # print(y_pred_pos[: 10])

    K = 50
    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    # print(kth_score_in_negative_edges)
    hits_50 = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    return torch.tensor(hits_50)


def compute_hits(model, evaluator, node_emb, pos_edge, neg_edge, device, batch_size=500):
    """
    compute hits@50 in batches.
    """
    pos_preds = []
    for start in tqdm.trange(0, pos_edge.size(0), batch_size, desc='Evaluate'):
        end = min(start + batch_size, pos_edge.size(0))
        edge = pos_edge[start:end].t()
        h_src = node_emb[edge[0]].to(device)
        h_dst = node_emb[edge[1]].to(device)
        pos_preds += [model.predictor(h_src * h_dst).squeeze(-1)]
    pos_pred = torch.cat(pos_preds, dim=0)

    neg_preds = []
    for start in tqdm.trange(0, neg_edge.size(0), batch_size, desc='Evaluate'):
        end = min(start + batch_size, neg_edge.size(0))
        edge = neg_edge[start:end].t()
        h_src = node_emb[edge[0]].to(device)
        h_dst = node_emb[edge[1]].to(device)
        neg_preds += [model.predictor(h_src * h_dst).squeeze(-1)]
    neg_pred = torch.cat(neg_preds, dim=0)

    # score = torch.cat([pos_pred, neg_pred])
    # pos_label = torch.ones_like(pos_pred)
    # neg_label = torch.zeros_like(neg_pred)
    # labels = torch.cat([pos_label, neg_label])
    # loss = F.binary_cross_entropy_with_logits(score, labels)
    # print(loss)

    evaluator.K = 50
    hits_50 = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })[f'hits@{50}']
    return torch.tensor(hits_50)

def zero_grad(params):
    for param in params:
        param.grad = None

def write_gzip_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'w') as f:
        df.to_csv(f, index=False)

def handler(signum, frame):
    try:
        debug_attach()
    except:
        debug_attach(5679)

def train(args, device, g, reverse_eids, seed_edges, model, edge_split, logger, run, eval_batch_size=256):
    # create sampler & dataloader
    total_it = np.ceil(seed_edges.shape[0] / args.batch_size)
    max_it = min(total_it, 250)
    # total_minit = 
    # total_it = 2358104 / args.batch_size
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    checkpoint_path = args.checkpoint_folder + get_output_name(args, run) + "_best.pth"
    if args.full_neighbor:
        print("We use the full neighbor of the target node to train the models. ")
        sampler = MultiLayerFullNeighborSampler(num_layers=args.num_layers, prefetch_node_feats=['feat'])
    else:
        print("We sample the neighbor node of the target node to train the models. ")
        if args.num_layers == 3:
            sampler = NeighborSampler([15, 10, 5], prefetch_node_feats=['feat'])
        else:
            sampler = NeighborSampler([15] * args.num_layers, prefetch_node_feats=['feat'])

    if args.dataset == "esci":
        if args.exclude_target_degree >= 0:
            print("We exclude training target with degree constraints")
            sampler = BipartiteEdgePredictionSampler(sampler,
                                                     exclude="reverse_id",
                                                     reverse_eids=reverse_eids,
                                                     negative_sampler=negative_sampler.Uniform(1),
                                                     degree_threshold=args.exclude_target_degree)
        elif args.exclude_target_degree == -1:
            print("We exclude the training target. ")
            max_degree = torch.max(g.in_degrees()).item()
            print("The maximum degree of the graph is: " + str(max_degree))
            sampler = BipartiteEdgePredictionSampler(sampler,
                                                     exclude="reverse_id",
                                                     reverse_eids=reverse_eids,
                                                     negative_sampler=negative_sampler.Uniform(1),
                                                     degree_threshold=max_degree + 1)
    else:
        if 0 > args.exclude_target_degree > -1:
            ratio = torch.as_tensor(abs(args.exclude_target_degree))
            degrees = g.out_degrees().type(torch.float32)
            threshold = torch.quantile(degrees, ratio)
            print(f"We only exclude training target for nodes with bottom {ratio:.2%} degree (threshold: {threshold})")
            sampler = EdgePredictionSamplerwithDegree(sampler, exclude="reverse_id", reverse_eids=reverse_eids,
                                                      negative_sampler=negative_sampler.Uniform(1),
                                                      degree_threshold=threshold)
        else:
            args.exclude_target_degree = int(args.exclude_target_degree)
            if args.exclude_target_degree == -1:
                print("We exclude the training target. ")
                sampler = as_edge_prediction_sampler(
                    sampler, exclude="reverse_id", reverse_eids=reverse_eids, negative_sampler=negative_sampler.Uniform(1))
            elif args.exclude_target_degree > 0:
                print("We only exclude training target with degree constraints")
                sampler = EdgePredictionSamplerwithDegree(sampler, exclude="reverse_id", reverse_eids=reverse_eids,
                                                        negative_sampler=negative_sampler.Uniform(1),
                                                        degree_threshold=args.exclude_target_degree)
            elif args.exclude_target_degree == 0:
                print("We do NOT exclude the training target. ")
                sampler = as_edge_prediction_sampler(
                    sampler, reverse_eids=reverse_eids, negative_sampler=negative_sampler.Uniform(1))
            elif args.exclude_target_degree < -1:
                print("We only randomly exclude training target with certain probability")
                sampler = EdgePredictionSamplerwithRandom(sampler, exclude="reverse_id", reverse_eids=reverse_eids,
                                                        negative_sampler=negative_sampler.Uniform(1),
                                                        degree_threshold=abs(args.exclude_target_degree))
            else:
                raise ValueError(f"Exclude degree number '{args.exclude_target_degree}' is not supported")

    use_uva = (args.mode == 'mixed')
    dataloader = DataLoader(
        g, seed_edges, sampler,
        device=device, batch_size=args.batch_size, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva)
    if args.alt_training_steps > 0:
        opt_list = [
            torch.optim.Adam(model.predictor.parameters(), lr=args.lr),
            torch.optim.Adam(model.layers.parameters(), lr=args.lr)
        ]
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)
    epoch_progress = tqdm.tqdm(range(args.n_epochs), desc="Epochs", dynamic_ncols=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = None
        dataloader_iter = enumerate(iter(dataloader))
        for epoch in epoch_progress:
            if args.alt_training_steps > 0:
                opt_selector = (epoch / args.alt_training_steps) % 1
                if opt_selector >= args.alt_training_ratio:
                    opt_ind = 1
                else:
                    opt_ind = 0
                print(f"Optimizing {opt_ind}th part of the model")
                opt = opt_list[opt_ind]
                opt.zero_grad()
            else:
                opt_ind = None
            model.train()
            total_loss = 0
            # batch accumulation parameter
            accum_iter = args.accum_iter_number

            print('Training...')
            at_least_one_batch = False
            it_progress = tqdm.tqdm(dataloader_iter, total=max_it, desc="Iter", 
                                    dynamic_ncols=True, postfix=dict(total_it=total_it))
            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in it_progress:
                at_least_one_batch = True
                # pair_graph: all positive edge pairs in this batch, stored  as a graph
                # neg_pair_graph: all negative edge pairs in this batch, stored as a graph
                # blocks: each block is the aggregated graph as input for each layer
                x = blocks[0].srcdata['feat'].float()
                pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
                score = torch.cat([pos_score, neg_score])
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                labels = torch.cat([pos_label, neg_label])
                loss = F.binary_cross_entropy_with_logits(score, labels)
                (loss / accum_iter).backward()
                if ((it + 1) % accum_iter == 0) or (it + 1 == len(dataloader)) or (it + 1 == max_it):
                    # if opt_ind == 1:
                    #     zero_grad(model.predictor.parameters())
                    # elif opt_ind == 0:
                    #     zero_grad(model.layers.parameters())

                    # Update Optimizer
                    opt.step()
                    opt.zero_grad()
                total_loss += loss.item()
                it_progress.set_postfix(it=it, total_it=total_it)
                if (it + 1) == total_it:
                    dataloader_iter = enumerate(iter(dataloader))
                    break
                elif (it + 1) % max_it == 0:
                    break
            it_progress.close()
            if not at_least_one_batch:
                warnings.warn("No batch is processed in this epoch.")
                dataloader_iter = enumerate(iter(dataloader))
                continue

            print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))
            if (epoch + 1) % args.log_steps == 0:
                model.eval()
                print('Validation/Testing...')
                with torch.no_grad():
                    node_emb = model.inference(g, device, eval_batch_size)
                    results = []

                    if args.dataset == "ogbl-citation2":
                        # We randomly pick some training samples that we want to evaluate on:
                        # torch.manual_seed(12345)
                        num_sampled_nodes = edge_split['valid']['target_node_neg'].size(dim=0)
                        idx = torch.randperm(edge_split['train']['source_node'].numel())[:num_sampled_nodes]
                        edge_split['eval_train'] = {
                            'source_node': edge_split['train']['source_node'][idx],
                            'target_node': edge_split['train']['target_node'][idx],
                            'target_node_neg': edge_split['valid']['target_node_neg'],
                        }

                        src = edge_split['eval_train']['source_node'].to(node_emb.device)
                        dst = edge_split['eval_train']['target_node'].to(node_emb.device)
                        neg_dst = edge_split['eval_train']['target_node_neg'].to(node_emb.device)
                        evaluator = Evaluator(name=args.dataset)
                        print('Train MRR {:.4f} '.format(
                            compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device).item()))
                    elif args.dataset == "ogbl-collab":
                        pos_edge = edge_split['train']['edge'].to(node_emb.device)
                        neg_edge = edge_split['valid']['edge_neg'].to(node_emb.device)
                        print(pos_edge.size())
                        evaluator = DetailEvaluator(name=args.dataset)
                        print('Train Hits@50 {:.4f} '.format(
                            compute_hits(model, evaluator, node_emb, pos_edge, neg_edge, device).item()))
                    elif args.dataset == "USAir":
                        pos_edge = edge_split['train']['edge'].to(node_emb.device)
                        neg_edge = edge_split['valid']['edge_neg'].to(node_emb.device)
                        print('Train AUC {:.4f} '.format(
                            compute_auc(model, node_emb, pos_edge, neg_edge, device).item()))
                    elif args.dataset == "esci":
                        # pos_edge = edge_split['train']['edge'].to(node_emb.device)
                        # neg_edge = edge_split['valid']['edge_neg'].to(node_emb.device)
                        # print('Train Hits@50 {:.4f} '.format(
                        #     compute_hits_esci(model, node_emb, pos_edge, neg_edge, device).item()))
                        torch.manual_seed(12345)
                        num_sampled_nodes = edge_split['valid']['target_node_neg'].size(dim=0)
                        idx = torch.randperm(edge_split['train']['source_node'].numel())[:num_sampled_nodes]
                        edge_split['eval_train'] = {
                            'source_node': edge_split['train']['source_node'][idx],
                            'target_node': edge_split['train']['target_node'][idx],
                            'target_node_neg': edge_split['valid']['target_node_neg'],
                        }

                        src = edge_split['eval_train']['source_node'].to(node_emb.device)
                        dst = edge_split['eval_train']['target_node'].to(node_emb.device)
                        neg_dst = edge_split['eval_train']['target_node_neg'].to(node_emb.device)

                        # head_degree = g.in_degrees(src)
                        # tail_degree = g.in_degrees(dst)
                        # print("Average Train degree")
                        # print(head_degree.float().mean())
                        # print(tail_degree.float().mean())
                        MRR, H_10, H_1 = compute_mrr_esci(model, node_emb, src, dst, neg_dst, device)
                        print('Train MRR {:.4f} '.format(MRR.item()))
                    
                    results_df_dict = dict()
                    for split in ['valid', 'test']:
                        if args.dataset == "ogbl-citation2":
                            src = edge_split[split]['source_node'].to(node_emb.device)
                            dst = edge_split[split]['target_node'].to(node_emb.device)
                            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
                            mrr, results_df = compute_mrr(model, evaluator, node_emb, 
                                                          src, dst, neg_dst, device, detailed_results=True)
                            results.append(mrr)
                            results_df_dict[split] = results_df
                        elif args.dataset == "ogbl-collab":
                            pos_edge = edge_split[split]['edge'].to(node_emb.device)
                            neg_edge = edge_split[split]['edge_neg'].to(node_emb.device)
                            results.append(compute_hits(model, evaluator, node_emb, pos_edge, neg_edge, device))
                            rr = evaluator.detailed_hits
                            src, dst = pos_edge.t()
                            results_df_dict[split] = pd.DataFrame({
                                "src": src.cpu().numpy(),
                                "dst": dst.cpu().numpy(),
                                "MRR": rr.cpu().numpy()
                            })
                        elif args.dataset == "USAir":
                            pos_edge = edge_split[split]['edge'].to(node_emb.device)
                            neg_edge = edge_split[split]['edge_neg'].to(node_emb.device)
                            results.append(compute_auc(model, node_emb, pos_edge, neg_edge, device))
                        elif args.dataset in ["esci" , "grid", "grid-dense"]:
                            evaluator = Evaluator(name="ogbl-collab")
                            src = edge_split[split]['source_node'].to(node_emb.device)
                            dst = edge_split[split]['target_node'].to(node_emb.device)
                            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)

                            # print("Average Test/Valid degree")
                            # head_degree = g.in_degrees(src)
                            # tail_degree = g.in_degrees(dst)
                            # print(head_degree.float().mean())
                            # print(tail_degree.float().mean())

                            # results.append(compute_hits_esci(model, node_emb, pos_edge, neg_edge, device))
                            MRR, H_10, H_1, results_df = compute_mrr_esci(
                                model, node_emb, src, dst, neg_dst, device, detailed_results=True)
                            results.append(MRR) # Recording MRR here
                            results_df_dict[split] = results_df
                            print(split + " Hits@10: ")
                            print(H_10.item() * 100)
                            print(split + " Hits@1: ")
                            print(H_1.item() * 100)
                        else:
                            raise ValueError(f"Dataset '{args.dataset}' is not supported")

                # save best checkpoint
                valid_result, test_result = results[0].item(), results[1].item()
                if logger.results[run]:
                    previous_best_valid_result = torch.as_tensor(logger.results[run])[:, 0].max().item()
                    if valid_result > previous_best_valid_result:
                        print("Saving checkpoint. ")
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                        }, checkpoint_path)
                        epoch_progress.set_postfix(best_valid=valid_result, best_test=test_result, pid=os.getpid())
                        if len(results_df_dict) > 0: 
                            results_df = pd.concat(
                                [results_df_dict['valid'], results_df_dict['test']], 
                                keys=['valid', 'test'], names=['split'])
                            # with gzip.open(checkpoint_path.replace(".pth", "_edge_results.csv.gz"), 'w') as f:
                            #     results_df.to_csv(f, index=False)
                            if future:
                                future.result()
                            executor.submit(write_gzip_csv, results_df, checkpoint_path.replace(".pth", "_edge_results.csv.gz"))
                        # write_gzip_csv(results_df, checkpoint_path.replace(".pth", "_edge_results.csv.gz"))

                logger.add_result(run, [valid_result, test_result])
                
                if args.dataset == "ogbl-citation2" or args.dataset == "esci" or args.dataset == "grid" or args.dataset == "grid-dense":
                    logger.metric_name = "MRR"
                    print('Validation MRR {:.4f}, Test MRR {:.4f}'.format(valid_result, test_result))
                elif args.dataset == "ogbl-collab":
                    logger.metric_name = "Hits@50"
                    print('Validation Hits@50 {:.4f}, Test Hits@50 {:.4f}'.format(valid_result, test_result))
                elif args.dataset == "USAir":
                    logger.metric_name = "AUC"
                    print('Validation AUC {:.4f}, Test AUC {:.4f}'.format(valid_result, test_result))
                else:
                    raise ValueError(f"Dataset '{args.dataset}' is not supported")
                gc.collect()
                torch.cuda.empty_cache()
    logger.print_statistics(run)
    epoch_progress.close()

if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, handler)
    if args.debug:
        debug_attach()
    
    # load and preprocess dataset
    print('Loading data')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    if args.dataset == 'USAir':
        g, reverse_eids, seed_edges, edge_split = load_usair_dataset(device,
                                                                     dataset_root="./seal-dataset/")
    elif args.dataset == "grid":
        g, reverse_eids, seed_edges, edge_split = load_grid_dataset(device, dataset_root="./dataset/shared_data_new", graph_idx=10)
    elif args.dataset == "grid-dense":
        g, reverse_eids, seed_edges, edge_split = load_dense_grid_dataset(device, 
            dataset_root="./dataset/dense_datasets", graph_idx=args.grid_idx)
    elif args.dataset == 'esci':
        g, reverse_eids, seed_edges, edge_split = load_esci_dataset(device, dataset_root="./dataset/esci")
    elif args.dataset == 'ogbl-collab_low_degree':
        dataset = DglLinkPropPredDataset('ogbl-collab')
        print(dataset[0].num_edges())
        g, edge_split = remove_collab_dissimilar_edges(dataset)
        g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
        g, reverse_eids = to_bidirected_with_reverse_mapping(g)
        reverse_eids = reverse_eids.to(device)
        seed_edges = torch.arange(g.num_edges()).to(device)
        print(g.num_edges())
        # print(edge_split['test']['edge'])
        print("Average degree")
        print(torch.mean(g.in_degrees().float()).item())
        args.dataset = "ogbl-collab"
    else:
        dataset = DglLinkPropPredDataset(args.dataset)
        g = dataset[0]
        g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
        g, reverse_eids = to_bidirected_with_reverse_mapping(g)
        reverse_eids = reverse_eids.to(device)
        seed_edges = torch.arange(g.num_edges()).to(device)
        edge_split = dataset.get_edge_split()

    # create GNN model
    in_size = g.ndata['feat'].shape[1]
    logger = Logger(args.runs)
    rnd_seed_generator = np.random.RandomState(args.seed)
    rnd_seed_list = rnd_seed_generator.randint(np.iinfo(np.int32).max, size=args.runs)
    print(f"Random seeds for runs: {rnd_seed_list}")
    for run in range(args.runs):
        # Results are not deterministic even with seed set
        set_random_seed(rnd_seed_list[run])
        if args.model_name == "SAGE":
            model = SAGE(in_size, args.hidden_dim, args.num_layers).to(device)
        elif args.model_name == "SAGE_DOT":
            model = SAGE_DOT(in_size, args.hidden_dim, args.num_layers).to(device)
        elif args.model_name == "SAGE_DistMultS":
            model = SAGE_DistMultS(in_size, args.hidden_dim, args.num_layers).to(device)
        elif args.model_name in ("GCN", "GCN_DOT", "GCN_DistMultS"):
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            print("Add self loop")
            if args.model_name == "GCN":
                model = GCN(in_size, args.hidden_dim, args.num_layers).to(device)
            elif args.model_name == "GCN_DistMultS":
                model = GCN_DistMultS(in_size, args.hidden_dim, args.num_layers).to(device)
            else:
                model = GCN_DOT(in_size, args.hidden_dim, args.num_layers).to(device)
        elif args.model_name == "MLPDecoder":
            model = MLPDecoder(in_size, args.hidden_dim, args.num_layers).to(device)
        elif args.model_name == "DistMultSDecoder":
            model = DistMultSDecoder(in_size).to(device)
        elif args.model_name == "GAT":
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            if args.dataset == "ogbl-citation2":
                num_heads = 8
            else:
                num_heads = 8
            num_out_heads = 1
            heads = ([num_heads] * (args.num_layers - 1)) + [num_out_heads]
            activation = F.elu
            feat_drop = 0
            attn_drop = 0
            negative_slope = 0.2
            residual = True
            model = GATv2(in_size, args.hidden_dim, args.num_layers, heads, activation, feat_drop, attn_drop,
                          negative_slope, residual).to(device)
        elif args.model_name == "GIN":
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            print("Add self loop")
            model = GIN(in_size, args.hidden_dim, args.num_layers).to(device)
        else:
            raise ValueError(f"Model '{args.model_name}' is not supported")
        # model training
        print('Training...')
        # print(edge_split['test'].keys())
        train(args, device, g, reverse_eids, seed_edges, model, edge_split, logger, run)
        # # validate/test the model
        # print('Validation/Testing...')
        # valid_mrr, test_mrr = evaluate(device, g, edge_split, model, batch_size=1000)
        # print('Validation MRR {:.4f}, Test MRR {:.4f}'.format(valid_mrr.item(),test_mrr.item()))
        results_df = logger.results_df
        csv_path = args.checkpoint_folder + get_output_name(args) + f"_{logger.metric_name}_results.csv"
        results_df.to_csv(csv_path)
    valid_stats, test_stats = logger.print_statistics()

    results_df = logger.results_df
    results_df.loc[("final", "mean"), :] = pd.Series(
        {"valid": valid_stats[0], "test": test_stats[0], "metric": logger.metric_name})
    results_df.loc[("final", "std"), :] = pd.Series(
        {"valid": valid_stats[1], "test": test_stats[1], "metric": logger.metric_name})
    csv_path = args.checkpoint_folder + get_output_name(args) + f"_{logger.metric_name}_results.csv"
    results_df.to_csv(csv_path)
    print(f"Results are saved to {csv_path}")