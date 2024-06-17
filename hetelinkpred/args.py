import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument("--model_name", default='SAGE', 
                        choices=['SAGE', 'SAGE_DOT', 'SAGE_DistMultS', 'GCN', 'GCN_DOT', 'GCN_DistMultS',
                                 'GAT', 'GIN', 'MLPDecoder', 'DistMultSDecoder'],
                        help="the model to run the link prediction")
    parser.add_argument("--checkpoint_folder", default='output/', help="Folder to save the checkpoint")
    parser.add_argument("--load_checkpoint_folder", default=None, help="Folder to load the poretrained checkpoint")
    parser.add_argument("--dataset", default='ogbl-citation2',
                        choices=['ogbl-citation2', 'ogbl-collab', 'USAir', 'esci', 'ogbl-collab_low_degree', 'grid', 'grid-dense'],
                        help="Dataset to do the experiment")
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--runs", type=int, default=1, help='Number of runs to repeat the experiment')
    parser.add_argument("--log_steps", type=int, default=1, help='Number of epochs to output the evaluation result.')
    parser.add_argument("--accum_iter_number", type=int, default=1, help='Number of gradient accumulation.')

    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of model layers.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--exclude_target_degree', type=float, default=0,
                        help='Whether to exclude the training target according to degree constraints. '
                             '-1 represents excluding all training edges and 0 represents excluding not edges ')
    parser.add_argument('--full_neighbor', type=bool, default=False,
                        help='Whether to include all the neighbor inside the computation. ')
    parser.add_argument('--inference_mode', default='train', choices=['train', 'valid', 'test'],
                        help='Different type of inference mode for part 2. When train is set, only train edges are present\
                        in the graph. When validation is set, train+valid edges will be in the graph. When test is set, train\
                        +valid+test edges will be in the graph.')
    parser.add_argument('--seed', type=int, default=1429362699, help='Random seed.')
    parser.add_argument('--grid_idx', type=int, default=0, help='The index of the grid dataset.')
    parser.add_argument('--alt_training_steps', type=int, default=0, help='Whether to use the alternative training method.')
    parser.add_argument('--alt_training_ratio', type=float, default=0.5, help='The ratio of the alternative training steps.')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    return args


def get_output_name(args, run=None):
    dataset_name = args.dataset
    if dataset_name.startswith('grid'):
        dataset_name += f'-{args.grid_idx}'
    
    name = args.model_name + "_" + dataset_name + "_" + "batch_size_" + str(
        args.batch_size) + "_n_layers_" + str(args.num_layers) + "_hidden_dim_" + str(args.hidden_dim) + "_lr_" + str(
        args.lr) + "_exclude_degree_" + str(args.exclude_target_degree) + "_full_neighbor_" + str(
        args.full_neighbor) + "_accu_num_" + str(args.accum_iter_number) + \
            f"_alt_train_{args.alt_training_steps}_{args.alt_training_ratio}" + "_n_epochs_" + str(args.n_epochs)
    if run is not None:
        name += "_run_" + str(run)
    return name