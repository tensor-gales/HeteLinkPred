"""
run the heuristic baselines resource allocation, common neighbours, personalised pagerank and adamic adar
"""

"""
runner for heuristic link prediction methods. Currently Adamic Adar, Common Neighbours and Personalised PageRank
"""
import time

import sklearn # Fix frozen ogb import
from ogb.linkproppred import Evaluator
import argparse
from argparse import Namespace
import scipy.sparse as ssp
import torch
import numpy as np

from src.evaluation import evaluate_auc, evaluate_mrr, evaluate_hits, compute_mrr_esci
from src.data import get_data, get_esci_data
from src.heuristics import AA, CN, PPR, RA
from src.utils import DEFAULT_DIC, get_pos_neg_edges
from src.data import get_data, get_loaders, get_grid_data, get_grid_loaders_BUDDY
from .utils import Logger, get_output_name
import pandas as pd


def vscode_debug():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    breakpoint()

def get_output_name_heuristics(args, heuristics):
  dataset_name = args.dataset_name
  if dataset_name.startswith('grid'):
      dataset_name += f'-{args.graph_idx}'
  
  name = heuristics + "_" + dataset_name 
  return name


def run(args):
  args = Namespace(**{**DEFAULT_DIC, **vars(args)})
  # set the correct metric for ogb
  k = 100
  if args.dataset_name == 'ogbl-collab' or args.dataset_name == 'grid':
    k = 50
  elif args.dataset_name == 'ogbl-ppi':
    k = 20

  # for heuristic in [RA, CN, AA, PPR]:
  for heuristic in [PPR]:
    logger = Logger(args.reps)

    heuristic_name = heuristic.__name__
    
    results_list = []
    for rep in range(args.reps):
      t0 = time.time()
      if args.dataset_name == 'grid':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset, splits, directed, eval_metric = get_grid_data(args, device, "./src/datasets/grid/dense_datasets")
        train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
        num_nodes = dataset.num_nodes
      elif args.dataset_name == 'esci':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset, splits, directed, eval_metric = get_esci_data(args, device, "./dataset/esci")
        train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
        num_nodes = dataset.num_nodes
      else:
        dataset, splits, directed, eval_metric = get_data(args)
        train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
        num_nodes = dataset.data.num_nodes
      if 'edge_weight' in train_data:
        train_weight = train_data.edge_weight.view(-1)
        test_weight = test_data.edge_weight.view(-1)
      else:
        train_weight = torch.ones(train_data.edge_index.size(1), dtype=int)
        test_weight = torch.ones(test_data.edge_index.size(1), dtype=int)
      train_edges, val_edges, test_edges = train_data['edge_index'], val_data['edge_index'], test_data['edge_index']
      assert torch.equal(val_edges, train_edges)
      A_train = ssp.csr_matrix((train_weight, (train_edges[0], train_edges[1])),
                               shape=(num_nodes, num_nodes))
      A_test = ssp.csr_matrix((test_weight, (test_edges[0], test_edges[1])),
                              shape=(num_nodes, num_nodes))

      # this function returns transposed edge list of shape [?,2]
      pos_train_edge, neg_train_edge = get_pos_neg_edges(splits['train'])
      pos_val_edge, neg_val_edge = get_pos_neg_edges(splits['valid'])
      pos_test_edge, neg_test_edge = get_pos_neg_edges(splits['test'])

      print(f'results for {heuristic.__name__} (val, test)')
      pos_train_pred, pos_train_edge = heuristic(A_train, pos_train_edge)
      neg_train_pred, neg_train_edge = heuristic(A_train, neg_train_edge)
      pos_val_pred, pos_val_edge = heuristic(A_train, pos_val_edge)
      neg_val_pred, neg_val_edge = heuristic(A_train, neg_val_edge)
      pos_test_pred, pos_test_edge = heuristic(A_test, pos_test_edge)
      neg_test_pred, neg_test_edge = heuristic(A_test, neg_test_edge)

      
      if args.dataset_name == 'grid' or args.dataset_name == 'ogbl-citation2' or args.dataset_name == 'ogbl-collab' or args.dataset_name == 'esci':
        results = compute_mrr_esci(pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                               neg_test_pred)

        result = results['MRR'] 
        train_res, val_res, test_res = result
        logger.add_result(rep, [val_res, test_res])
        logger.metric_name = "MRR"
        print('Validation MRR {:.4f}, Test MRR {:.4f}'.format(val_res, test_res))

        results_df = logger.results_df
        csv_path = args.checkpoint_folder + get_output_name_heuristics(args, heuristic_name) + f"_{logger.metric_name}_results.csv"
        results_df.to_csv(csv_path)

      else:
        evaluator = Evaluator(name='ogbl-ppa')
        hit_results = evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred,
                                    pos_test_pred,
                                    neg_test_pred, Ks=[k])
        key = f'Hits@{k}'
        train_res, val_res, test_res = hit_results[key]
        res_dic = {f'rep{rep}_Train' + key: 100 * train_res, f'rep{rep}_Val' + key: 100 * val_res,
                   f'rep{rep}_Test' + key: 100 * test_res}
        results_list.append(hit_results[key])
        print(hit_results)

    if args.reps > 1:
      train_acc_mean, val_acc_mean, test_acc_mean = np.mean(results_list, axis=0) * 100
      test_acc_std = np.sqrt(np.var(results_list, axis=0)[-1]) * 100
    print(f'{heuristic.__name__} ran in {time.time() - t0:.1f} s for {args.reps} reps')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', type=str, default='Cora',
                      choices=['Cora', 'producer', 'Citeseer', 'Pubmed', 'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
                               'ogbl-citation2', 'grid', 'esci'])
  parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
  parser.add_argument("--checkpoint_folder", type=str, default='output/', help="Folder to save the checkpoint")
  parser.add_argument("--graph_idx", type=int, default = 0)
  parser.add_argument('--reps', type=int, default=1, help='the number of repetition of the experiment to run')
  parser.add_argument('--sample_size', type=int, default=None,
                      help='the number of training edges to sample. Currently only implemented for producer data')
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()
  if args.debug:
        vscode_debug()
  print(args)
  run(args)
