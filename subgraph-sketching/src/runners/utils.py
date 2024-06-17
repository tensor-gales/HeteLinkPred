import torch
import numpy as np
import scipy.sparse as ssp
import torch.nn.functional as F
import dgl
import os
from tqdm import tqdm
import pandas as pd

class Logger(object):
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]
        self.metric_name = None

    def add_result(self, run, result):
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Valid: {result[:, 0].max():.2%}')
            print(f'   Final Test: {result[argmax, 1]:.2%}')
        else:
            result = torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            valid_stats = (r.mean().item(), r.std().item())
            print(f'Highest Valid: {valid_stats[0]:.2%} ± {valid_stats[1]:.2%}')
            r = best_result[:, 1]
            test_stats = (r.mean().item(), r.std().item())
            print(f'   Final Test: {test_stats[0]:.2%} ± {test_stats[1]:.2%}')
            return valid_stats, test_stats

    @property
    def results_df(self):
        results_arr = np.array(self.results)
        run_df_list = []
        for i in range(results_arr.shape[0]):
            run_arr = np.array(self.results[i])
            if len(run_arr) == 0:
                continue
            run_df = pd.DataFrame(run_arr, columns=['valid', 'test'])
            run_df.index.name = 'epoch'
            run_df_list.append(run_df)
        df = pd.concat(run_df_list, keys=range(results_arr.shape[0]), names=["run"])
        if self.metric_name is not None:
            df["metric"] = self.metric_name
        return df


def get_output_name(args, run=None):
    dataset_name = args.dataset_name
    if dataset_name.startswith('grid'):
        dataset_name += f'-{args.graph_idx}'
    name_list = []
    name_list.append(args.model)
    name_list.append(dataset_name)
    if args.use_feature == False:
        name_list.append('nofeat')
    elif args.sign_norm != 'gcn':
        name_list.append(args.sign_norm)
    name_list.append(f'sign_k_{args.sign_k}')
    name_list.append(f'hidden_dim_{args.hidden_channels}')
    name_list.append(f'num_negs_{args.num_negs}')
    name_list.append(f'batch_size_{args.batch_size}')
    name_list.append(f'lr_{args.lr}')
    name_list.append(f'n_epochs_{args.epochs}')
    if run is not None:
        name_list.append(f'run_{run}')
    return '_'.join(name_list)

    # name = args.model + "_" + dataset_name + (
    #     f"_{args.sign_norm}" if args.sign_norm != "gcn" else "") \
    #     + "_sign_k_" + str(args.sign_k) \
    #     + "_hidden_dim_" + str(args.hidden_channels) \
    #     + "_num_negs_" + str(args.num_negs) + "_batch_size_" + str(
    #     args.batch_size) + "_lr_" + str(
    #     args.lr) + "_n_epochs_" + str(args.epochs) 
    # if run is not None:
    #     name += "_run_" + str(run)
    # return name