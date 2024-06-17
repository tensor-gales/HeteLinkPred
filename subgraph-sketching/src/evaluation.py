"""
hitrate@k, mean reciprocal rank (MRR) and Area under the receiver operator characteristic curve (AUC) evaluation metrics
"""
from sklearn.metrics import roc_auc_score
import torch
from ogb.linkproppred import Evaluator
import numpy as np


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

def evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,
                  Ks=[20, 50, 100], use_val_negs_for_train=True):
    """
    Evaluate the hit rate at K
    :param evaluator: an ogb Evaluator object
    :param pos_val_pred: Tensor[val edges]
    :param neg_val_pred: Tensor[neg val edges]
    :param pos_test_pred: Tensor[test edges]
    :param neg_test_pred: Tensor[neg test edges]
    :param Ks: top ks to evaluatate for
    :return: dic[ks]
    """
    results = {}
    # As the training performance is used to assess overfitting it can help to use the same set of negs for
    # train and val comparisons.
    if use_val_negs_for_train:
        neg_train = neg_val_pred
    else:
        neg_train = neg_train_pred
    for K in Ks:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_train,
        })[f'hits@{K}']
        rr_train = evaluator.detailed_hits

        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        rr_valid = evaluator.detailed_hits

        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        rr_test = evaluator.detailed_hits
        
        results["RR"] = (rr_train, rr_valid, rr_test)
        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

def compute_mrr_esci(pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    """Compute Mean Reciprocal Rank (MRR) in batches in esci dataset."""

    results = {}
    pos_train_pred = pos_train_pred.view(pos_train_pred.shape[0], -1)
    pos_val_pred = pos_val_pred.view(pos_val_pred.shape[0], -1)
    pos_test_pred = pos_test_pred.view(pos_test_pred.shape[0], -1)
    
    neg_train_pred = neg_train_pred.view(pos_train_pred.shape[0], -1)
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)

    print("COMPUTING TRAINING MRR ESCI")
    print("PREDICTION SHAPE: ", pos_train_pred.shape)
    print("NEG PREDICTION SHAPE: ", neg_train_pred.shape)
    rr = torch.zeros(pos_train_pred.shape[0])
    # hits_at_10 = torch.zeros(pos_train_pred.shape[0])
    # hits_at_1 = torch.zeros(pos_train_pred.shape[0])

    # TRAIN

    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (neg_train_pred >= pos_train_pred).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (neg_train_pred > pos_train_pred).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    # hits_at_50 = ranking_list <= 50
    hits_at_10 = ranking_list <= 10
    hits_at_1 = ranking_list <= 1
    mrr_list = 1. / ranking_list.to(torch.float)
    rr = mrr_list

    rr_train = rr
    MRR_train = rr.mean()
    # Hits_50_train = hits_at_50.sum() / pos_train_pred.shape[0]
    Hits_10_train = hits_at_10.sum() / pos_train_pred.shape[0]
    Hits_1_train = hits_at_1.sum() / pos_train_pred.shape[0]


    # VALID

    print("COMPUTING VALID MRR ESCI")
    print("PREDICTION SHAPE: ", pos_val_pred.shape)
    print("PREDICTION SHAPE: ", neg_val_pred.shape)

    rr = torch.zeros(pos_val_pred.shape[0])
    # hits_at_10 = torch.zeros(pos_val_pred.shape[0])
    # hits_at_1 = torch.zeros(pos_val_pred.shape[0])

    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (neg_val_pred >= pos_val_pred).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (neg_val_pred > pos_val_pred).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    # hits_at_50 = ranking_list <= 50
    hits_at_10 = ranking_list <= 10
    hits_at_1 = ranking_list <= 1
    mrr_list = 1. / ranking_list.to(torch.float)
    rr = mrr_list

    rr_valid = rr
    MRR_valid = rr.mean()
    # Hits_50_valid = hits_at_50.sum() / pos_val_pred.shape[0]
    Hits_10_valid = hits_at_10.sum() / pos_val_pred.shape[0]
    Hits_1_valid = hits_at_1.sum() / pos_val_pred.shape[0]

    # TEST

    print("COMPUTING TEST MRR ESCI")
    print("PREDICTION SHAPE: ", pos_test_pred.shape)
    print("PREDICTION SHAPE: ", neg_test_pred.shape)

    rr = torch.zeros(pos_test_pred.shape[0])
    # hits_at_10 = torch.zeros(pos_test_pred.shape[0])
    # hits_at_1 = torch.zeros(pos_test_pred.shape[0])

    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (neg_test_pred >= pos_test_pred).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (neg_test_pred > pos_test_pred).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    # hits_at_50 = ranking_list <= 50
    hits_at_10 = ranking_list <= 10
    hits_at_1 = ranking_list <= 1
    mrr_list = 1. / ranking_list.to(torch.float)
    rr = mrr_list

    rr_test = rr
    MRR_test = rr.mean()
    # Hits_50_test = hits_at_50.sum() / pos_test_pred.shape[0]
    Hits_10_test = hits_at_10.sum() / pos_test_pred.shape[0]
    Hits_1_test = hits_at_1.sum() / pos_test_pred.shape[0]

    results["RR"] = (rr_train, rr_valid, rr_test)
    results['MRR'] = (MRR_train, MRR_valid, MRR_test)
    # results['Hits@50'] = (Hits_50_train, Hits_50_valid, Hits_50_test)
    results['Hits@10'] = (Hits_10_train, Hits_10_valid, Hits_10_test)
    results['Hits@1'] = (Hits_1_train, Hits_1_valid, Hits_1_test)

    return results

def evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, 
                 pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    """
    Evaluate the mean reciprocal rank at K
    :param evaluator: an ogb Evaluator object
    :param pos_val_pred: Tensor[val edges]
    :param neg_val_pred: Tensor[neg val edges]
    :param pos_test_pred: Tensor[test edges]
    :param neg_test_pred: Tensor[neg test edges]
    :param Ks: top ks to evaluatate for
    :return: dic with single key 'MRR'
    """
    neg_train_pred = neg_train_pred.view(pos_train_pred.shape[0], -1)
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}

    train_mrr_list = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        # for mrr negs all have the same src, so can't use the val negs, but as long as the same  number of negs / pos are
        # used the results will be comparable.
        'y_pred_neg': neg_train_pred,
    })['mrr_list']
    train_mrr = train_mrr_list.mean().item()

    valid_mrr_list = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list']
    valid_mrr = valid_mrr_list.mean().item()

    test_mrr_list = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list']
    test_mrr = test_mrr_list.mean().item()

    results["RR"] = (train_mrr_list, valid_mrr_list, test_mrr_list)
    results['MRR'] = (train_mrr, valid_mrr, test_mrr)

    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    """
    the ROC AUC
    :param val_pred: Tensor[val edges] predictions
    :param val_true: Tensor[val edges] labels
    :param test_pred: Tensor[test edges] predictions
    :param test_true: Tensor[test edges] labels
    :return:
    """
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results
