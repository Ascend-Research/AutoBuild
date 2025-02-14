from sdm.graph_util import generate_nx_digraph_skeleton, QUANT_METHOD_LUT, BIT_PRECISION_LUT
from copy import deepcopy
from itertools import product
import torch as t


def _trim_cartesian_list(c_list):
    new_c_list = []
    for sublist in c_list:
        new_sublist = [entry for entry in sublist if entry.split("-")[0] in QUANT_METHOD_LUT.keys() and int(entry.split("-")[1]) in BIT_PRECISION_LUT.keys()]
        new_c_list.append(new_sublist)
    return new_c_list


def _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, hops):

    nx_digraph = generate_nx_digraph_skeleton(relevant_nodes, adj_list)
    relevant_nodes_dict = {k: node_dict[k] for k in relevant_nodes}
    for rn in relevant_nodes:
        del node_dict[rn]

    variant_list = []
    cartesian_list = [list(v.keys()) for v in relevant_nodes_dict.values()]
    cartesian_list = _trim_cartesian_list(cartesian_list)

    for comb in product(*cartesian_list):
        digraph_copy = deepcopy(nx_digraph)
        digraph_copy.hops = hops
        node_list = list(digraph_copy.nodes(data=True))
        for index, key in enumerate(comb):
            node_name, node_feat_dict = node_list[index][0], node_list[index][1]
            node_feat_dict['config'] = key
            node_feat_dict['all'] = relevant_nodes_dict[node_name][key]
        variant_list.append(digraph_copy)
    return variant_list
