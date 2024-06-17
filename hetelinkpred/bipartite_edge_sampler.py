from dgl.dataloading.base import EdgePredictionSampler, _find_exclude_eids
from collections.abc import Mapping
from dgl.base import EID, NID
from dgl.utils import recursive_apply, context_of
from dgl.convert import heterograph
from dgl import backend as F
import dgl
import torch
from low_degree_edge_sampler import find_exclude_eids


class BipartiteEdgePredictionSampler(EdgePredictionSampler):
    """Sampler class that builds upon EdgePredictionSampler
    The exlucde train target is only done on edges with a degree < 10
    TODO: Change the fixed degree to user-defined args

    ------------------------------
    Need to call this directly in the code instead of calling as_edge_prediction_sampler
    """

    def __init__(self, sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None, negative_sampler=None, prefetch_labels=None, degree_threshold=10):
        super().__init__(sampler, exclude, reverse_eids, reverse_etypes, negative_sampler, prefetch_labels)
        self.degree_threshold = degree_threshold

    def _build_neg_graph(self, g, seed_edges):
        device = seed_edges.device
        us, vs = g.find_edges(seed_edges)
        queries = us < 33804
        bz_size = seed_edges.shape[0]
        num_asins = g.number_of_nodes() - 33804
        asin_idxs = torch.randperm(num_asins)[:bz_size]
        asins_to_sample = torch.arange(33804, g.number_of_nodes())[asin_idxs].to(device)
        query_idxs = torch.randperm(33804)[:bz_size]
        queries_to_sample = torch.arange(0, 33804)[query_idxs].to(device)
        neg_vs = torch.where(queries, asins_to_sample, queries_to_sample)
        neg_srcdst = (us, neg_vs)
        # TODO: double check negative edge does not exist in graph
        # neg_srcdst = self.negative_sampler(g, seed_edges)
        if not isinstance(neg_srcdst, Mapping):
            assert len(g.canonical_etypes) == 1, \
                'graph has multiple or no edge types; ' \
                'please return a dict in negative sampler.'
            neg_srcdst = {g.canonical_etypes[0]: neg_srcdst}

        dtype = F.dtype(list(neg_srcdst.values())[0][0])
        ctx = context_of(seed_edges) if seed_edges is not None else g.device
        neg_edges = {
            etype: neg_srcdst.get(etype,
                                  (F.copy_to(F.tensor([], dtype), ctx=ctx),
                                   F.copy_to(F.tensor([], dtype), ctx=ctx)))
            for etype in g.canonical_etypes}
        neg_pair_graph = heterograph(
            neg_edges, {ntype: g.num_nodes(ntype) for ntype in g.ntypes})
        return neg_pair_graph

    def sample(self, g, seed_edges):  # pylint: disable=arguments-differ
        """Samples a list of blocks, as well as a subgraph containing the sampled
        edges from the original graph.
        If :attr:`negative_sampler` is given, also returns another graph containing the
        negative pairs as edges.
        """
        if isinstance(seed_edges, Mapping):
            seed_edges = {g.to_canonical_etype(k): v for k, v in seed_edges.items()}
        exclude = self.exclude
        pair_graph = g.edge_subgraph(
            seed_edges, relabel_nodes=False, output_device=self.output_device)
        eids = pair_graph.edata[EID]

        if self.negative_sampler is not None:
            neg_graph = self._build_neg_graph(g, seed_edges)
            pair_graph, neg_graph = dgl.compact_graphs([pair_graph, neg_graph])
        else:
            pair_graph = dgl.compact_graphs(pair_graph)

        pair_graph.edata[EID] = eids
        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = find_exclude_eids(
            g, seed_edges, exclude, self.reverse_eids, self.reverse_etypes,
            self.output_device, self.degree_threshold)

        input_nodes, _, blocks = self.sampler.sample(g, seed_nodes, exclude_eids)

        if self.negative_sampler is None:
            return self.assign_lazy_features((input_nodes, pair_graph, blocks))
        else:
            return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))
