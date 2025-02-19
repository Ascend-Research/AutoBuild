import numpy as np
from sdm.graph_util import AVAILABLE_BP, AVAILABLE_QM
from collections import OrderedDict
from sdm.constants import QM_BIT_ORDER


QUANT_METHODS = ["-".join([m, str(b)]) for m in AVAILABLE_QM for b in AVAILABLE_BP]

# Passing in a list of subgraphs
# each subgraph has a .score attribute.
# Template for others like min, 
def all_score(nx_graph_list):
    return [sg.score for sg in nx_graph_list]

def max_score(nx_graph_list):
    scores = [sg.score for sg in nx_graph_list]
    return max(scores)


def min_score(nx_graph_list):
    scores = [sg.score for sg in nx_graph_list]
    return min(scores)


def mean_score(nx_graph_list):
    scores = [sg.score for sg in nx_graph_list]
    return np.mean(scores)


def std_score(nx_graph_list):
    scores = [sg.score for sg in nx_graph_list]
    return np.std(scores)


def best_quant_methods(nx_graph_list):
    nx_graph_list.sort(reverse=True, key=lambda x:x.score)
    best_graph = nx_graph_list[0]
    return [g[1]['config'] for g in best_graph.nodes(data=True) if g[1]['config'] != 'combine-16']


def best_quant_methods_subgraphs(nx_graph_list):
    nx_graph_list.sort(reverse=True, key=lambda x:x.score)
    best_graph = nx_graph_list[0]
    qm_dict = {}
    for node in best_graph.nodes(data=True):
        if node[1]['config'] != "combine-16":
            qm_dict[node[0]] = node[1]['config']
    return qm_dict


def worst_quant_methods(nx_graph_list):
    nx_graph_list.sort(reverse=False, key=lambda x:x.score)
    best_graph = nx_graph_list[0]
    return [g[1]['config'] for g in best_graph.nodes(data=True) if g[1]['config'] != 'combine-16']


def node_qm_hist(sublist):
    hist = {}
    for aqm in QM_BIT_ORDER.keys():
        hist[aqm] = 0
    for qm in sublist:
        hist[qm] += 1
    return hist

def quant_method_hist(node_qm_list, nx_graph):
    assert all([len(sublist) == len(nx_graph.nodes) for sublist in node_qm_list])
    node_dict = {}
    for idx, node in enumerate(nx_graph.nodes()):
        node_dict[node] = node_qm_hist(node_qm_list[idx])
    return node_dict


def print_qm_hist(qm_hist):
    for k, v in qm_hist.items():
        print(f"     {k}: {v}")


def convert_fqms_to_dist(qms):
    total = 0
    for k, v in qms.items():
        total += v
    new_qms = {}
    for k in qms.keys():
        new_qms[k] = qms[k] / total
    return new_qms