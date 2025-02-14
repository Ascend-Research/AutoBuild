import argparse
from ge_utils.misc_utils import get_train_config_from_chkpt_folder, save_units_folder
from ge_utils.model import make_predictor
import torch as t
import torch_geometric as tg
from copy import deepcopy
from joblib import delayed, Parallel, cpu_count
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import os
import subprocess


def _apply_feat_func(graph, func_handle):
    graph_intermittent = deepcopy(graph)
    for node in graph_intermittent.nodes(data=True):
        node[1]['all'] = func_handle(node[0], node[1]['all'])
    return graph_intermittent


def run_predictor_on_graph(predictor, g_list):
    score_list = []
    for i in range(len(g_list)):
        graph, graph_tg = g_list[i][0], g_list[i][1]
        sg_embed = predictor.get_gnn_node_embeds(graph_tg)[graph.hops][-1, :]
        biased_score = t.linalg.norm(sg_embed, ord=train_config['norm'], keepdim=False)
        score = predictor.standardize_embedding(biased_score, graph.hops) * predictor.hop_srcc[graph.hops]
        score_list.append(score.detach().item())
    return score_list


if __name__ == "__main__":
    params = argparse.ArgumentParser(description="")
    params.add_argument("-chkpt", type=str, required=True)
    params.add_argument("-node", default=False, action="store_true")

    params = params.parse_args()

    train_config = get_train_config_from_chkpt_folder(params.chkpt)
    assert train_config['format'] == "sdm"
    chkpts = os.listdir(params.chkpt)
    predictors = [make_predictor(gnn_chkpt=os.sep.join([params.chkpt, c]), **train_config) for c in chkpts]
    for p in predictors:
        p.eval()

    family = train_config['families']
    if "dit" in family: 
        from sdm.modules_dit import iterate_dit as iterate_sdm
        from sdm.graph_dit import convert_dit_layer_to_vector as feat_convert_func
    elif "sdv15" in family: 
        from sdm.modules_sdv15 import iterate_sdv15 as iterate_sdm
        from sdm.graph_sdv15 import convert_sdv15_layer_to_vector as feat_convert_func
    elif "sdxl" in family: 
        from sdm.modules_sdxl import iterate_sdxl as iterate_sdm
        from sdm.graph_sdxl import convert_sdxl_layer_to_vector as feat_convert_func
    elif "alpha" in family or "sigma" in family:
        from sdm.modules_pixart import iterate_pixart as iterate_sdm
        from sdm.graph_pixart import convert_dit_layer_to_vector as feat_convert_func
    elif "hunyuan" in family:
        from sdm.modules_hunyuan import iterate_hunyuan as iterate_sdm
        from sdm.graph_hunyuan import convert_dit_layer_to_vector as feat_convert_func
    else: raise NotImplementedError

    module_dict = iterate_sdm(nodes=params.node)
    for key, graphs in module_dict.items():
        print("Profiling", key)
        graph_list = [[graph, tg.utils.convert.from_networkx(_apply_feat_func(graph, feat_convert_func), group_node_attrs = ['all'])] for graph in graphs]
        score_list = Parallel(n_jobs=5)(delayed(run_predictor_on_graph)(predictor, graph_list) for predictor in predictors)
        for i, graph in enumerate(graphs):
            graph.score = sum([score_list[p][i] for p in range(len(score_list))])
        graphs.sort(reverse=True, key=lambda x: x.score)
    
    if params.chkpt.endswith(os.sep):
        params.chkpt = params.chkpt[:-1]
    add_str = "_nodes" if params.node else "_sgs"
    params.chkpt += add_str
    params.chkpt += os.sep

    if params.node:
        params.chkpt = params.chkpt.replace("predictor", "nodes_predictor")
    save_f = save_units_folder(module_dict, params.chkpt)

    cmd_str = f"python convert_sdm_unit_to_scheme.py -unit {save_f}"
    subprocess.run(cmd_str.split(" "))