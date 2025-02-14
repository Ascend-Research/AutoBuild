import torch as t
import networkx as nx


QUANT_METHOD_LUT = {
    "combine": 0,         # This is Add, Concat, Matmul, AdaLN - takes 2 distributions
    "mse": 1,
    "kmeans": 2,
    "kmeans_all": 3,
    #"quantile_mean": 4,   # NOTE CURRENTLY UNUSED
    #"quantile_median": 5, # NOTE CURRENTLY UNUSED
    #"combine": 6          # Add, Concat, Matmul, etc. Not quantized.
}

BIT_PRECISION_LUT = {
    16: 0,                 # Precision for combine
    4:  1,
    3:  2,
    #2:  3,                 # NOTE CURRENTLY UNUSED 
}

AVAILABLE_QM = ["mse", "kmeans", "kmeans_all"] #, "quantile_mean", "quantile_median"]
AVAILABLE_BP = [4, 3] #, 2]


def gen_combine_node():
    quant_code = QUANT_METHOD_LUT["combine"]
    bit_code = BIT_PRECISION_LUT[16]
    reduction_percentage = 0
    quantization_error = 0.0
    return t.Tensor([quant_code, bit_code, reduction_percentage, quantization_error])


# UNIVERAL MEANING THIS FUNCTION APPLIES TO DiT, SDv1.5, SDXL
# This converts the information from the MultiQuantizer. That includes:
#   quantization_method and bit precision (first element as str, separated by '-')
#   size quantized
#   original size
#   quantization error
# example: ['mse-3', 589824, 115200, 0.004047393798828125]
def convert_universal_layer_feats_to_vector(node_info_list):
    quant_method, bit_precision = node_info_list[0].split("-")
    quant_code = QUANT_METHOD_LUT[quant_method]
    bit_code = BIT_PRECISION_LUT[int(bit_precision)]
    # Quantized size / FP size
    reduction_percentage = 100 * node_info_list[1] / node_info_list[2]
    quantization_error = node_info_list[3] * 100

    universal_node_feature_vec = t.Tensor([quant_code, bit_code, reduction_percentage, quantization_error])
    return universal_node_feature_vec


def _resize_graph(dot, size_per_element=1.0, min_size=18):
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def visualize_graph(graph, filename, min_size=18):
    from graphviz import Digraph
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='14',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, filename=filename,
                  graph_attr=dict(size="15,15"))
    sg_node_list = [n for n in graph.nodes]
    for n in sg_node_list:
    #for i, n in enumerate(graph.nodes):
        if "weight_quantizer" in n:
            dot.node(n, n, fillcolor="deepskyblue", fontcolor="black")
        else:
            dot.node(n, n, fillcolor="brown2", fontcolor="white")
    for edge in graph.edges:
        dot.edge(edge[0], edge[1])
        #if undirected:
        #    node_i_list = [i for i in get_reverse_adj_dict[n.str_id] if i in sg_str_ids]
        #    for i in node_i_list:
        #        dot.edge(i, n.str_id)
    _resize_graph(dot, min_size=min_size)
    dot.render(view=False, cleanup=True, format="png")


def get_node_variant(mq_subdict, qm_choices=AVAILABLE_QM, bp_choices=AVAILABLE_BP):
    node_universal_feats = {}
    for qm in qm_choices:
        for bp in bp_choices:
            key = "-".join([qm, str(bp)])
            node_universal_feats[key] = convert_universal_layer_feats_to_vector(mq_subdict[key])
    return node_universal_feats


def get_all_dit_node_feats(universal_feat_dict):
    from sdm.graph_dit import DIT_TSX_COMBINE_LUT, DIT_TSX_LAYERS_LUT
    for mq_name, mq_dict in universal_feat_dict.items():
        if mq_name.startswith("model.transformer_blocks.") and "timestep_embedder" not in mq_name:
            mq_name = mq_name.replace("model.transformer_blocks.", "").replace(".weight_quantizer", "")
            dit_k_list = mq_name.split(".")
            blk_code = int(dit_k_list[0])
            layer_type = ".".join(dit_k_list[1:])
        elif "timestep_embedder" in mq_name and "transformer_blocks.0" not in mq_name:
            pass
        else:
            blk_code = -1
            layer_type = mq_name.replace(".weight_quantizer", "").replace("model.", "")
        if layer_type in DIT_TSX_LAYERS_LUT.keys():
            layer_code = DIT_TSX_LAYERS_LUT[layer_type]
        else:
            layer_code = DIT_TSX_COMBINE_LUT[layer_type]
        dit_specific_feats = t.Tensor([blk_code, layer_code])
        for qm_bp, universal_feat in mq_dict.items():
            mq_dict[qm_bp] = t.cat([universal_feat, dit_specific_feats.clone()])
    return universal_feat_dict


def get_all_pixart_node_feats(universal_feat_dict):
    from sdm.graph_pixart import PIXART_TSX_COMBINE_LUT, PIXART_TSX_LAYERS_LUT
    for mq_name, mq_dict in universal_feat_dict.items():
        if mq_name.startswith("model.transformer_blocks.") and "timestep_embedder" not in mq_name:
            mq_name = mq_name.replace("model.transformer_blocks.", "").replace(".weight_quantizer", "")
            dit_k_list = mq_name.split(".")
            blk_code = int(dit_k_list[0])
            layer_type = ".".join(dit_k_list[1:])
        elif "timestep_embedder" in mq_name and "transformer_blocks.0" not in mq_name:
            pass
        else:
            blk_code = -1
            layer_type = mq_name.replace(".weight_quantizer", "").replace("model.", "")
        if layer_type in PIXART_TSX_LAYERS_LUT.keys():
            layer_code = PIXART_TSX_LAYERS_LUT[layer_type]
        else:
            layer_code = PIXART_TSX_COMBINE_LUT[layer_type]
        dit_specific_feats = t.Tensor([blk_code, layer_code])
        for qm_bp, universal_feat in mq_dict.items():
            mq_dict[qm_bp] = t.cat([universal_feat, dit_specific_feats.clone()])
    return universal_feat_dict


def get_all_hunyuan_node_feats(universal_feat_dict):
    from sdm.graph_hunyuan import HUNYUAN_TSX_COMBINE_LUT, HUNYUAN_TSX_LAYERS_LUT
    for mq_name, mq_dict in universal_feat_dict.items():
        if mq_name.startswith("model.blocks.") and "timestep_embedder" not in mq_name:
            mq_name = mq_name.replace("model.blocks.", "").replace(".weight_quantizer", "").replace("_0", "")
            dit_k_list = mq_name.split(".")
            blk_code = int(dit_k_list[0])
            layer_type = ".".join(dit_k_list[1:])
        elif "timestep_embedder" in mq_name and "blocks.0" not in mq_name:
            pass
        else:
            blk_code = -1
            layer_type = mq_name.replace(".weight_quantizer", "").replace("model.", "")
        if layer_type in HUNYUAN_TSX_LAYERS_LUT.keys():
            layer_code = HUNYUAN_TSX_LAYERS_LUT[layer_type]
        else:
            layer_code = HUNYUAN_TSX_COMBINE_LUT[layer_type]
        dit_specific_feats = t.Tensor([blk_code, layer_code])
        for qm_bp, universal_feat in mq_dict.items():
            mq_dict[qm_bp] = t.cat([universal_feat, dit_specific_feats.clone()])
    return universal_feat_dict


def get_all_sdv15_node_feats(universal_feat_dict):
    from sdm.graph_sdv15 import _get_stage_code, _get_layer_code, _get_block_code
    for mq_name, mq_dict in universal_feat_dict.items():
        sc, lc, bc = _get_stage_code(mq_name), _get_layer_code(mq_name), _get_block_code(mq_name)
        specific_feats = t.Tensor([sc, bc, lc])
        for qm_bp, universal_feat in mq_dict.items():
            mq_dict[qm_bp] = t.cat([universal_feat, specific_feats.clone()])
    return universal_feat_dict


def get_all_sdxl_node_feats(universal_feat_dict):
    from sdm.graph_sdxl import _get_stage_code, _get_layer_code, _get_block_code, _get_tsx_num_code
    for mq_name, mq_dict in universal_feat_dict.items():
        sc, lc, tc, bc = _get_stage_code(mq_name), _get_layer_code(mq_name), _get_tsx_num_code(mq_name), _get_block_code(mq_name)
        specific_feats = t.Tensor([sc, bc, tc, lc])
        for qm_bp, universal_feat in mq_dict.items():
            mq_dict[qm_bp] = t.cat([universal_feat, specific_feats.clone()])
    return universal_feat_dict


def get_all_node_variants(qnn_ckpt, qm_choices=AVAILABLE_QM, bp_choices=AVAILABLE_BP, net=None):

    all_node_universal_feats = {}
    for mq_key, mq_subdict in qnn_ckpt.items():
        all_node_universal_feats[mq_key] = get_node_variant(mq_subdict, qm_choices=qm_choices, bp_choices=bp_choices)

    if net is None:
        return all_node_universal_feats
    elif net == "dit":
        return get_all_dit_node_feats(all_node_universal_feats)
    elif net == "sdv15":
        return get_all_sdv15_node_feats(all_node_universal_feats)
    elif net == "sdxl":
        return get_all_sdxl_node_feats(all_node_universal_feats)
    elif net == "pixart":
        return get_all_pixart_node_feats(all_node_universal_feats)
    elif net == "hunyuan":
        return get_all_hunyuan_node_feats(all_node_universal_feats)
    else:
        raise NotImplementedError



def generate_nx_digraph_skeleton(node_list, adj_list):
    graph = nx.DiGraph()
    for n in node_list:
        graph.add_node(n, all=None, config=None) # NOTE this is for the features which we fill-in later
    
    for [i, j] in adj_list:
        graph.add_edge(node_list[i], node_list[j])
    
    return graph


if __name__ == "__main__":

    # TODO for Wednesday, get all variations of a node.
    qnn_dict_ckpt = "sdxl_mq.pkl"
    import pickle

    with open(qnn_dict_ckpt, "rb") as f:
        qnn_dict = pickle.load(f)

    myDict = get_all_node_variants(qnn_dict, net="sdxl")
    print(myDict)