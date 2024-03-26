import torch as t
from torch_geometric.utils.subgraph import k_hop_subgraph
import numpy as np
import hashlib


def extract_n_hop_subgraphs(geo_data, scores=None, num_hops=1, ids=None):

    sg_list = []
    if ids is None:
        ids = list(range(geo_data.x.shape[0]))
    for node_idx in ids:
        node_subset, trimmed_edge_index, _, edge_mask = k_hop_subgraph(
            node_idx=node_idx, num_hops=num_hops, edge_index=geo_data.edge_index
        )
        node_subset = node_subset.tolist()
        node_dict = {'id': node_idx,
                      'node_list': node_subset}
        if scores is not None:
            node_dict['score'] = scores[node_idx]
        sg_list.append(node_dict)
    return sg_list


def extract_top_k_n_hop_subgraphs(geo_data, scores, num_hops=1, k=10):

    score_indices = np.argsort(scores).tolist()
    score_indices.reverse()
    score_indices = score_indices[:k]

    return extract_n_hop_subgraphs(geo_data, scores=scores, num_hops=num_hops, ids=score_indices)


def extract_quantile_subgraphs(geo_data, scores, num_hops=1, q=0.75):
    score_indices = np.where(scores >= np.quantile(scores, q))[0].tolist()
    score_indices.sort(reverse=True, key=lambda x: scores[x])

    return extract_n_hop_subgraphs(geo_data, scores=scores, num_hops=num_hops, ids=score_indices)


def extract_sdev_subgraphs(geo_data, scores, num_hops=1, m=0):
    mean, sdev = np.mean(scores), np.std(scores)
    threshold = mean + (m * sdev)
    score_indices = np.where(scores >= threshold)[0].tolist()
    score_indices.sort(reverse=True, key=lambda x: scores[x])

    return extract_n_hop_subgraphs(geo_data, scores=scores, num_hops=num_hops, ids=score_indices)


def generate_tg_feat_hash(x):
    feat_list = x.reshape(-1).cpu().tolist()
    node_feats_as_str = "".join([str(f) for f in feat_list])
    return hashlib.sha512(node_feats_as_str.encode("UTF-8")).hexdigest()