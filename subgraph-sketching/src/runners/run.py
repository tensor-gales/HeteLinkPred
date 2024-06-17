"""
main module
"""
import sklearn # Fix frozen ogb import
from ogb.linkproppred import Evaluator
import argparse
import time
import warnings
from math import inf
import sys
import random
from .utils import Logger, get_output_name
import pandas as pd
import gzip
from pathlib import Path
import tqdm
import gc
from src.evaluation import DetailEvaluator

# sys.path.insert(0, '..')

import numpy as np
import torch


torch.set_printoptions(precision=4)
# when generating subgraphs the supervision edge is deleted, which triggers a SparseEfficiencyWarning, but this is
# not a performance bottleneck, so suppress for now
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from src.data import get_data, get_esci_data, get_esci_loaders_BUDDY, get_loaders, get_grid_data, get_grid_loaders_BUDDY
from src.models.elph import ELPH, BUDDY
from src.models.seal import SEALDGCNN, SEALGCN, SEALGIN, SEALSAGE
from src.utils import ROOT_DIR, print_model_params, select_embedding, str2bool
from src.runners.train import get_train_func
from src.runners.inference import test

def print_results_list(results_list):
    for idx, res in enumerate(results_list):
        print(f'repetition {idx}: test {res[0]:.2f}, val {res[1]:.2f}, train {res[2]:.2f}')

def set_seed(seed):
    """
    setting a random seed for reproducibility and in accordance with OGB rules
    @param seed: an integer seed
    @return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def vscode_debug():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    breakpoint()

def write_gzip_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'w') as f:
        df.to_csv(f, index=False)

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"executing on {device}")
    results_list = []
    train_func = get_train_func(args)
    logger = Logger(args.reps)
    
    for rep in range(args.reps):
        set_seed(rep)
        print("Loading Data")
        if args.dataset_name =="grid":
            dataset, splits, directed, eval_metric = get_grid_data(args, device, "./src/datasets/grid/dense_datasets")
            train_loader, train_eval_loader, val_loader, test_loader = get_grid_loaders_BUDDY(args, dataset, device, splits, "./src/datasets/grid/dense_datasets")
        elif args.dataset_name == "esci":
            dataset, splits, directed, eval_metric = get_esci_data(args, device, "./dataset/esci")
            train_loader, train_eval_loader, val_loader, test_loader = get_esci_loaders_BUDDY(args, dataset, device, splits, "./src/datasets/ESCI")
        else:
            dataset, splits, directed, eval_metric = get_data(args)
            train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)

        if args.dataset_name.startswith('ogbl'):  # then this is one of the ogb link prediction datasets
            evaluator = DetailEvaluator(name=args.dataset_name)
        elif args.dataset_name == "grid" or args.dataset_name == "esci":
            evaluator = DetailEvaluator(name="ogbl-collab")
        else:
            evaluator = DetailEvaluator(name='ogbl-ppa')  # this sets HR@100 as the metric
        emb = select_embedding(args, dataset.num_nodes, device)
        model, optimizer = select_model(args, dataset, emb, device)
        val_res = test_res = best_epoch = 0
        print(f'running repetition {rep}')
        csv_path = args.checkpoint_folder + get_output_name(args, run=rep) + "_edge_results.csv.gz"
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if rep == 0:
            print_model_params(model)
        for epoch in tqdm.tqdm(range(args.epochs), desc='Epochs', dynamic_ncols=True):
            t0 = time.time()
            loss = train_func(model, optimizer, train_loader, args, device, logger)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, epoch time: {time.time() - t0:.1f}')
            if (epoch + 1) % args.eval_steps == 0:
                results = test(model, evaluator, train_eval_loader, val_loader, test_loader, args, device,
                               eval_metric=eval_metric)
                one_eval_key_recorded = False
                for key, result in results.items():
                    if key == "RR":
                        write_gzip_csv(result, csv_path)
                        continue
                    elif key.lower() == eval_metric or (len(results) - int("RR" in results) == 1):
                        assert not one_eval_key_recorded
                        one_eval_key_recorded = True
                        train_res, tmp_val_res, tmp_test_res = result
                        if tmp_val_res > val_res:
                            val_res = tmp_val_res
                            test_res = tmp_test_res
                            best_epoch = epoch
                            print("Saving checkpoint. ")
                            torch.save({
                                'epoch': epoch + 1,
                                'loss': loss,
                                'model_state_dict': model.state_dict(),
                            }, csv_path.replace("_edge_results.csv.gz", "_checkpoint.pth"))
                        res_dic = {f'rep{rep}_loss': loss, f'rep{rep}_Train' + key: 100 * train_res,
                                f'rep{rep}_Val' + key: 100 * val_res, f'rep{rep}_tmp_val' + key: 100 * tmp_val_res,
                                f'rep{rep}_tmp_test' + key: 100 * tmp_test_res,
                                f'rep{rep}_Test' + key: 100 * test_res, f'rep{rep}_best_epoch': best_epoch,
                                f'rep{rep}_epoch_time': time.time() - t0, 'epoch_step': epoch}
                        to_print = f'Epoch: {epoch:02d}, Best epoch: {best_epoch}, Loss: {loss:.4f}, Train: {100 * train_res:.2f}%, Valid: ' \
                                f'{100 * tmp_val_res:.2f}%, Test: {100 * tmp_test_res:.2f}%, epoch time: {time.time() - t0:.1f}'
                        print(key)
                        print(to_print)

                    
                        logger.add_result(rep, [tmp_val_res, tmp_test_res])
                        logger.metric_name = key
                        # if args.dataset_name == "ogbl-citation2" or args.dataset_name == "esci" or args.dataset_name == "grid" or args.dataset_name == "grid-dense":
                        #     logger.metric_name = "MRR"
                        #     print('Validation MRR {:.4f}, Test MRR {:.4f}'.format(val_res, test_res))
                        # elif args.dataset_name == "ogbl-collab":
                        #     logger.metric_name = "Hits@50"
                        #     print('Validation Hits@50 {:.4f}, Test Hits@50 {:.4f}'.format(valid_result, test_result))
                        # elif args.dataset_name == "USAir":
                        #     logger.metric_name = "AUC"
                        #     print('Validation AUC {:.4f}, Test AUC {:.4f}'.format(valid_result, test_result))
                        # else:
                        #     raise ValueError(f"Dataset '{args.dataset_name}' is not supported")
                    else:
                        _train_res, tmp_val_res, tmp_test_res = result
                        to_print = f'Epoch: {epoch:02d}, Best epoch: {best_epoch}, Loss: {loss:.4f}, Train: {100 * _train_res:.2f}%, Valid: ' \
                                f'{100 * tmp_val_res:.2f}%, Test: {100 * tmp_test_res:.2f}%, epoch time: {time.time() - t0:.1f}'
                        print(key)
                        print(to_print)
                assert one_eval_key_recorded
        results_df = logger.results_df
        csv_path = args.checkpoint_folder + get_output_name(args) + f"_{logger.metric_name}_results.csv"
        results_df.to_csv(csv_path)

        logger.print_statistics(run=rep)

        if args.reps > 1:
            results_list.append([test_res, val_res, train_res])
            print_results_list(results_list)
        del dataset, splits, train_loader, train_eval_loader, val_loader, test_loader, evaluator, \
            emb, optimizer
        gc.collect()

    if args.reps > 1:
        test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results_list, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results_list, axis=0)[0]) * 100
        val_acc_std = np.sqrt(np.var(results_list, axis=0)[1]) * 100

        results_df = logger.results_df
        results_df.loc[("final", "mean"), :] = pd.Series(
            {"valid": val_acc_mean, "test": test_acc_mean, "metric": logger.metric_name})
        results_df.loc[("final", "std"), :] = pd.Series(
            {"valid": val_acc_std, "test": test_acc_std, "metric": logger.metric_name})
        csv_path = args.checkpoint_folder + get_output_name(args) + f"_{logger.metric_name}_results.csv"
        results_df.to_csv(csv_path)
        print(f"Results are saved to {csv_path}")

    if args.save_model:
        path = f'{ROOT_DIR}/saved_models/{args.dataset_name}'
        torch.save(model.state_dict(), path)


def select_model(args, dataset, emb, device):
    if args.model == 'SEALDGCNN':
        model = SEALDGCNN(args.hidden_channels, args.num_seal_layers, args.max_z, args.sortpool_k,
                          dataset, args.dynamic_train, use_feature=args.use_feature,
                          node_embedding=emb).to(device)
    elif args.model == 'SEALSAGE':
        model = SEALSAGE(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                         args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'SEALGCN':
        model = SEALGCN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout, pooling=args.seal_pooling).to(
            device)
    elif args.model == 'SEALGIN':
        model = SEALGIN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'BUDDY':
        if args.dataset_name in {"grid", "esci"}:
            model = BUDDY(args, dataset["feats"].size(1), node_embedding=emb).to(device)
        else:
            model = BUDDY(args, dataset.num_features, node_embedding=emb).to(device)
    elif args.model == 'ELPH':
        if args.dataset_name == "grid":
            model = ELPH(args, dataset.feat.size(1), node_embedding=emb).to(device)
        else:
            model = ELPH(args, dataset.num_features, node_embedding=emb).to(device)
    else:
        raise NotImplementedError
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
        print(f'SortPooling k is set to {model.k}')
    return model, optimizer


if __name__ == '__main__':
    print("Got to main")
    # Data settings
    parser = argparse.ArgumentParser(description='Efficient Link Prediction with Hashes (ELPH)')
    parser.add_argument('--dataset_name', type=str, default='Cora',
                        choices=['Cora', 'Citeseer', 'Pubmed', 'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
                                 'ogbl-citation2', 'grid', "esci"])
    parser.add_argument("--graph_idx", type=int, default = 0)
    parser.add_argument("--checkpoint_folder", type=str, default='output/', help="Folder to save the checkpoint")
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help='the percentage of supervision edges to be used for validation. These edges will not appear'
                             ' in the training set and will only be used as message passing edges in the test set')
    parser.add_argument('--test_pct', type=float, default=0.2,
                        help='the percentage of supervision edges to be used for test. These edges will not appear'
                             ' in the training or validation sets for either supervision or message passing')
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    parser.add_argument('--val_samples', type=float, default=inf, help='the number of val edges or % if < 1')
    parser.add_argument('--test_samples', type=float, default=inf, help='the number of test edges or % if < 1')
    parser.add_argument('--preprocessing', type=str, default=None)
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--sign_norm', type=str, default="gcn")
    parser.add_argument('--load_features', action='store_true', help='load node features from disk')
    parser.add_argument('--load_hashes', action='store_true', help='load hashes from disk')
    parser.add_argument('--cache_subgraph_features', action='store_true',
                        help='write / read subgraph features from disk')
    parser.add_argument('--train_cache_size', type=int, default=inf, help='the number of training edges to cache')
    parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
    # GNN settings
    parser.add_argument('--model', type=str, default='BUDDY')
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=1000000,
                        help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--feature_dropout', type=float, default=0.5)
    parser.add_argument('--sign_dropout', type=float, default=0.5)
    parser.add_argument('--save_model', action='store_true', help='save the model to use later for inference')
    parser.add_argument('--feature_prop', type=str, default='gcn',
                        help='how to propagate ELPH node features. Values are gcn, residual (resGCN) or cat (jumping knowledge networks)')
    # SEAL settings
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_seal_layers', type=int, default=3)
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--label_pooling', type=str, default='add', help='add or mean')
    parser.add_argument('--seal_pooling', type=str, default='edge', help='how SEAL pools features in the subgraph')
    # Subgraph settings
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--node_label', type=str, default='drnl')
    parser.add_argument('--max_dist', type=int, default=4)
    parser.add_argument('--max_z', type=int, default=1000,
                        help='the size of the label embedding table. ie. the maximum number of labels possible')
    parser.add_argument('--use_feature', type=str2bool, default=True,
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_struct_feature', type=str2bool, default=True,
                        help="whether to use structural graph features as GNN input")
    parser.add_argument('--use_edge_weight', action='store_true',
                        help="whether to consider edge weight in GNN")
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimization')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--propagate_embeddings', action='store_true',
                        help='propagate the node embeddings using the GCN diffusion operator')
    parser.add_argument('--loss', default='bce', type=str, help='bce or auc')
    parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
                        help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    parser.add_argument('--use_RA', type=str2bool, default=False, help='whether to add resource allocation features')
    # SEAL specific args
    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    # Testing settings
    parser.add_argument('--reps', type=int, default=1, help='the number of repetition of the experiment to run')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_metric', type=str, default='hits',
                        choices=('hits', 'mrr', 'auc'))
    parser.add_argument('--K', type=int, default=100, help='the hit rate @K')
    # hash settings
    parser.add_argument('--use_zero_one', type=str2bool, default=0,
                        help="whether to use the counts of (0,1) and (1,0) neighbors")
    parser.add_argument('--floor_sf', type=str2bool, default=0,
                        help='the subgraph features represent counts, so should not be negative. If --floor_sf the min is set to 0')
    parser.add_argument('--hll_p', type=int, default=8, help='the hyperloglog p parameter')
    parser.add_argument('--minhash_num_perm', type=int, default=128, help='the number of minhash perms')
    parser.add_argument('--max_hash_hops', type=int, default=2, help='the maximum number of hops to hash')
    parser.add_argument('--subgraph_feature_batch_size', type=int, default=4194304,
                        help='the number of edges to use in each batch when calculating subgraph features. '
                             'Reduce or this or increase system RAM if seeing killed messages for large graphs')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        vscode_debug()
    if (args.max_hash_hops == 1) and (not args.use_zero_one):
        print("WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features")
    if args.dataset_name == 'ogbl-ddi':
        args.use_feature = 0  # dataset has no features
        assert args.sign_k > 0, '--sign_k must be set to > 0 i.e. 1,2 or 3 for ogbl-ddi'
    print(args)
    run(args)
