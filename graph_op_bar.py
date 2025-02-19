import pickle
from sdm.constants import *
from sdm.insight_stat import *
import numpy as np
import fnmatch
import argparse
import matplotlib.pyplot as plt
from pandas import DataFrame as df
from params import P_SEP, PLOTS_DIR


def compute_dist_and_range(pop):
    return {'mean': np.mean(pop),
            'dev': np.std(pop),
            'min': np.min(pop),
            'max': np.max(pop)}


def get_relevant_entries(score_dict, keyword):
    filtered_keys = fnmatch.filter(score_dict.keys(), keyword)
    return {k: score_dict[k] for k in filtered_keys}


if __name__ == "__main__":

    params = argparse.ArgumentParser(description="")
    params.add_argument("-unit", type=str, required=True)
    params.add_argument("-method", type=str, default="all_score")   
    params.add_argument("-title", type=str, required=True)
        
    params = params.parse_args()

    method_handle = eval(params.method)

    print("Loading scores...")
    with open(params.unit, "rb") as f:
        score_dict = pickle.load(f)

    if "dit" in params.unit:
        arch = "DiT-XL/2"
        arch_f = "dit"
        layer_types = DIT_LAYER_TYPES
        block_list = DIT_SUBGRAPHS
        layer_dict = DIT_SUBGRAPHS_TYPES
        del_list = []
        for k in score_dict.keys():
            if "emb.timestep_embedder.linear_" in k and "transformer_blocks.0" not in k:
                del_list.append(k)
        for k in del_list:
            del score_dict[k]
    elif "alpha" in params.unit or "sigma" in params.unit:
        if "alpha" in params.unit:
            arch = "PixArt-Alpha"
            arch_f = "alpha"
        else:
            arch = "PixArt-Sigma"
            arch_f = "sigma"
        layer_types = PIXART_LAYER_TYPES
        block_dict = PIXART_SUBGRAPHS
        layer_dict = PIXART_SUBGRAPHS_TYPES
    elif "hunyuan" in params.unit:
        arch = "Hunyuan"
        arch_f = "hunyuan"
        layer_types = HUNYUAN_LAYER_TYPES
        block_dict = HUNYUAN_SUBGRAPHS
        layer_dict = HUNYUAN_SUBGRAPHS_TYPES
    elif "sdv15" in params.unit:
        arch = "SDv1.5"
        arch_f = "sdv15"
        layer_types = SDV15_LAYER_TYPES
        block_list = SDV15_SUBGRAPHS
        layer_dict = SDV15_SUBGRAPH_TYPES
    elif "sdxl" in params.unit:
        arch = "SDXL"
        arch_f = "sdxl"
        layer_types = SDXL_LAYER_TYPES
        block_list = SDXL_SUBGRAPHS
        layer_dict = SDXL_SUBGRAPH_TYPES
    else:
        raise NotImplementedError

    print("========Layer-wise========")
    all_data = {}
    for k, v in QM_BIT_ORDER.items():
        all_data[v] = []
    keys = list(layer_types.keys())
    for key, layer_type in layer_dict.items():        
        sel_entries = get_relevant_entries(score_dict, layer_type)
        module_best_qms = [best_quant_methods_subgraphs(v) for v in sel_entries.values()]
        overall_qm_dict = {}
        sample_sg = list(sel_entries.values())[0][0]
        for layer_name, layer_type in layer_types.items():
            if any([layer_type.replace("*", "") in node for node in sample_sg.nodes()]):
                overall_qm_dict[layer_type] = {}
                for qm in QUANT_METHODS:
                    overall_qm_dict[layer_type][qm] = 0
        for qm_dict in module_best_qms:
            for k, v in qm_dict.items():
                for layer_type in layer_types.values():
                    if layer_type.replace("*", "") in k:
                        try:
                            overall_qm_dict[layer_type][v] += 1
                        except KeyError:
                            overall_qm_dict[layer_type] = {}
                            for qm in QUANT_METHODS:
                                overall_qm_dict[layer_type][qm] = 0
                            overall_qm_dict[layer_type][v] += 1
        print("===")
        for node_name, subdict in overall_qm_dict.items():
            print(node_name)
            print(subdict)
            qms_dist = convert_fqms_to_dist(subdict)
            print(qms_dist)
            for k, v in QM_BIT_ORDER.items():
                print(qms_dist[k])
                all_data[v].append(qms_dist[k])

    print(all_data)

    all_data = df(all_data)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    all_data.plot.bar(stacked=True, ax=axs, color=QM_COLORS)
    title = 'Box Plot' if params.title is None else params.title
    axs.set_title(title)

    axs.yaxis.grid(True)
    axs.set_xticks([y for y in range(len(keys))],
                    labels=keys)
    
    axs.legend(loc='lower center', ncol=6, fancybox=False, bbox_to_anchor=(0.5, -0.5), fontsize=8)

    plt.xticks(rotation=30)
    plt.title(f"{arch} {params.title} Quantization Setting Distributions")

    plt.tight_layout()
    plt.savefig(P_SEP.join([PLOTS_DIR, "_".join([arch_f, "graph_op_bar", params.title])]))
