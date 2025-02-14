import torch as t
from sdm.modules_util import _generate_variant_digraphs
from sdm.graph_util import get_all_node_variants, gen_combine_node
from sdm.graph_dit import DIT_TSX_COMBINE_LUT
from collections import OrderedDict
import pickle
import networkx as nx


PKL_FILE_PATH = "cache/dit_mq.pkl"


def iterate_dit_attn(node_dict, block_idx):
    assert block_idx >= 0 and block_idx < 28

    if block_idx == 0:
        first_node = "model.pos_embed.proj.weight_quantizer"
    else:
        first_node = f"model.transformer_blocks.{block_idx - 1}.add_ff"
        node_dict[first_node] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx -1, DIT_TSX_COMBINE_LUT["add"]])])}

    relevant_nodes = [
        f"model.transformer_blocks.{block_idx}.norm1.linear.weight_quantizer",
        first_node, # 0
        f"model.transformer_blocks.{block_idx}.adaln_0", #1
        f"model.transformer_blocks.{block_idx}.attn1.to_q.weight_quantizer", #2
        f"model.transformer_blocks.{block_idx}.attn1.to_k.weight_quantizer", #3
        f"model.transformer_blocks.{block_idx}.attn1.to_v.weight_quantizer", #4
        f"model.transformer_blocks.{block_idx}.mul_qk", #5
        f"model.transformer_blocks.{block_idx}.mul_vsm", #6
        f"model.transformer_blocks.{block_idx}.attn1.to_out.0.weight_quantizer" #7
    ]

    node_dict[relevant_nodes[2]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx, DIT_TSX_COMBINE_LUT["adaln"]])])}
    node_dict[relevant_nodes[6]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx, DIT_TSX_COMBINE_LUT["add"]])])}
    node_dict[relevant_nodes[7]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx, DIT_TSX_COMBINE_LUT["add"]])])}

    adj_list = [
        [0, 2],
        [1, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [3, 6],
        [4, 6],
        [5, 7],
        [6, 7],
        [7, 8]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 4)


def iterate_dit_ff(node_dict, block_idx):
    assert block_idx >= 0 and block_idx < 28

    relevant_nodes = [
        f"model.transformer_blocks.{block_idx}.ff.net.0.proj.weight_quantizer",
        f"model.transformer_blocks.{block_idx}.ff.net.2.weight_quantizer"
    ]


    adj_list = [
        [0, 1],
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)

def iterate_dit_proj_out(node_dict):

    relevant_nodes = [
        "model.proj_out_1.weight_quantizer",
        "model.proj_out_2.weight_quantizer"
    ]

    adj_list = [
        [0, 1]
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def iterate_dit_time_embedding(node_dict):

    relevant_nodes = [
        "model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_1.weight_quantizer",
        "model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_2.weight_quantizer"
        ]
        
    adj_list = [
        [0, 1]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def iterate_dit(pkl_path=PKL_FILE_PATH, nodes=False):
    with open(pkl_path, "rb") as f:
        qnn_dict = pickle.load(f)

    node_dict = get_all_node_variants(qnn_dict, net="dit")

    if nodes:
        node_only_dict = {}
        for node_name, config_dict in node_dict.items():
            node_only_dict[node_name] = []
            for config_name, config_vec in config_dict.items():
                nx_graph = nx.DiGraph()
                nx_graph.add_node(node_name, all=config_vec, config=config_name)
                nx_graph.hops = 0
                node_only_dict[node_name].append(nx_graph)
        return node_only_dict

    module_dict = OrderedDict()
    module_dict['time_embedding'] = iterate_dit_time_embedding(node_dict)
    module_dict['proj_out'] = iterate_dit_proj_out(node_dict)

    for i in range(28):
        if i != 0:
            del node_dict[f"model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_1.weight_quantizer"]
            del node_dict[f"model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_2.weight_quantizer"]
        module_dict[f'dit_blk_{i}_attn'] = iterate_dit_attn(node_dict, i)
        module_dict[f'dit_blk_{i}_ff'] = iterate_dit_ff(node_dict, i)

    assert len(node_dict.keys()) == 0, node_dict.keys()
    return module_dict

if __name__ == "__main__":


    with open(PKL_FILE_PATH, "rb") as f:
        qnn_dict = pickle.load(f)


    node_dict = get_all_node_variants(qnn_dict, net="dit")
    for i in range(1, 28):
        del node_dict[f"model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_1.weight_quantizer"]
        del node_dict[f"model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_2.weight_quantizer"]

    running_sum = 0
    running_sum += len(iterate_dit_time_embedding(node_dict))
    running_sum += len(iterate_dit_proj_out(node_dict))

    for block_idx in range(28):
        running_sum += len(iterate_dit_attn(node_dict, block_idx))
        running_sum += len(iterate_dit_ff(node_dict, block_idx))
    print(running_sum)
    assert len(node_dict.keys()) == 0, node_dict.keys()

