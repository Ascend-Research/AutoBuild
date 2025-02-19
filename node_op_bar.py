import pickle
from sdm.constants import *
from sdm.insight_stat import *
import numpy as np
import fnmatch
import argparse
import matplotlib.pyplot as plt
import os
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
        block_list = DIT_BLOCK_TYPES
        layer_dict = DIT_LAYER_TYPES
        del_list = []
        for k in score_dict.keys():
            if "emb.timestep_embedder.linear_" in k and "transformer_blocks.0" not in k:
                #print(k)
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
        block_list = PIXART_BLOCK_TYPES
        layer_dict = PIXART_LAYER_TYPES
    elif "hunyuan" in params.unit:
        arch = "Hunyuan"
        arch_f = "hunyuan"
        block_list = HUNYUAN_BLOCK_TYPES
        layer_dict = HUNYUAN_LAYER_TYPES
    elif "sdv15" in params.unit:
        arch = "SDv1.5"
        arch_f = "sdv15"
        block_list = SDV15_BLOCK_TYPES
        layer_dict = SDV15_LAYER_TYPES
    elif "sdxl" in params.unit:
        arch = "SDXL"
        arch_f = "sdxl"
        block_list = SDXL_BLOCK_TYPES
        layer_dict = SDXL_LAYER_TYPES
    else:
        raise NotImplementedError
 
    print("========Layer-wise========")
    all_data = {}
    for k, v in QM_BIT_ORDER.items():
        all_data[v] = []
    keys = []
    for key, layer_type in layer_dict.items():
        sel_entries = get_relevant_entries(score_dict, layer_type)
        module_best_qms = [best_quant_methods(v) for v in sel_entries.values()]
        flattened_qms = [x for xs in module_best_qms for x in xs]
        qms_dist = convert_fqms_to_dist(node_qm_hist(flattened_qms))
        keys.append(key)
        if key == "FF 1":
            print(qms_dist)
        for k, v in QM_BIT_ORDER.items():
            all_data[v].append(qms_dist[k])
            
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
    plt.savefig(P_SEP.join([PLOTS_DIR, "_".join([arch_f, "node_op_bar", params.title])]))