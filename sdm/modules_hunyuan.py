import torch as t
from sdm.modules_util import _generate_variant_digraphs
from sdm.graph_util import get_all_node_variants, gen_combine_node
from sdm.graph_hunyuan import HUNYUAN_TSX_COMBINE_LUT
from collections import OrderedDict
import pickle
import networkx as nx


PKL_FILE_PATH = "cache/hunyuan_mq.pkl"


def iterate_hunyuan_selfattn(node_dict, block_idx):
    assert block_idx >= 0 and block_idx < 40

    relevant_nodes = [
        f"model.blocks.{block_idx}.norm1.linear.weight_quantizer", #0
        f"model.blocks.{block_idx}.attn1.to_q.weight_quantizer", #1
        f"model.blocks.{block_idx}.attn1.to_k.weight_quantizer", #2
        f"model.blocks.{block_idx}.attn1.to_v.weight_quantizer", #3
        f"model.blocks.{block_idx}.mul_qk1", #4
        f"model.blocks.{block_idx}.mul_vsm1", #5
        f"model.blocks.{block_idx}.attn1.to_out.0.weight_quantizer" #6
    ]

    node_dict[relevant_nodes[4]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx, HUNYUAN_TSX_COMBINE_LUT["add"]])])}
    node_dict[relevant_nodes[5]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx, HUNYUAN_TSX_COMBINE_LUT["add"]])])}

    adj_list = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [2, 4],
        [3, 5],
        [4, 5],
        [5, 6]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 3)


def iterate_hunyuan_crossattn1(node_dict, block_idx):
    assert block_idx >= 0 and block_idx < 40

    relevant_nodes = [
        f"model.blocks.{block_idx}.attn2.to_q.weight_quantizer", #0
        f"model.blocks.{block_idx}.attn2.to_k.weight_quantizer", #1
        f"model.blocks.{block_idx}.mul_qk2", #2
    ]

    node_dict[relevant_nodes[2]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx, HUNYUAN_TSX_COMBINE_LUT["add"]])])}

    adj_list = [
        [0, 2],
        [1, 2]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def iterate_hunyuan_crossattn2(node_dict, block_idx):
    assert block_idx >= 0 and block_idx < 40

    relevant_nodes = [
        f"model.blocks.{block_idx}.attn2.to_v.weight_quantizer", #0
        f"model.blocks.{block_idx}.mul_qk2", #1
        f"model.blocks.{block_idx}.mul_vsm2", #2
        f"model.blocks.{block_idx}.attn2.to_out.0.weight_quantizer" #3
    ]

    node_dict[relevant_nodes[1]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx, HUNYUAN_TSX_COMBINE_LUT["add"]])])}
    node_dict[relevant_nodes[2]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx, HUNYUAN_TSX_COMBINE_LUT["add"]])])}

    adj_list = [
        [0, 2],
        [1, 2],
        [2, 3]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 2)


def iterate_hunyuan_ff(node_dict, block_idx):
    assert block_idx >= 0 and block_idx < 40

    relevant_nodes = [
        f"model.blocks.{block_idx}.ff.net.0.proj.weight_quantizer",
        f"model.blocks.{block_idx}.ff.net.2.weight_quantizer"
    ]

    adj_list = [
        [0, 1],
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def iterate_hunyuan_skip(node_dict, block_idx):
    assert block_idx >= 21 and block_idx < 40

    relevant_nodes = [
        f"model.blocks.{block_idx}.skip_linear.weight_quantizer",
        f"model.blocks.{block_idx}.skip_linear.weight_quantizer_0",
        f"model.blocks.{block_idx}.add_skip"
    ]

    node_dict[relevant_nodes[2]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([block_idx, HUNYUAN_TSX_COMBINE_LUT["add"]])])}

    adj_list = [
        [0, 2],
        [1, 2]
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def iterate_hunyuan_pos_embed(node_dict):

    relevant_nodes = [
        "model.pos_embed.proj.weight_quantizer",
    ]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)


def iterate_hunyuan_proj_out(node_dict):

    relevant_nodes = [
        "model.norm_out.linear.weight_quantizer",
        "model.proj_out.weight_quantizer",
    ]

    adj_list = [
        [0, 1]
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def iterate_hunyuan_time_embedding(node_dict):

    relevant_nodes = [
        "model.time_extra_emb.timestep_embedder.linear_1.weight_quantizer", #0
        "model.time_extra_emb.timestep_embedder.linear_2.weight_quantizer", #1
        "model.time_extra_emb.extra_embedder.linear_1.weight_quantizer",    #2
        "model.time_extra_emb.extra_embedder.linear_2.weight_quantizer",    #3
        "time_extra_emb.add"                                                #4
        ]

    node_dict[relevant_nodes[4]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([-1, HUNYUAN_TSX_COMBINE_LUT["add"]])])}
        
    adj_list = [
        [0, 1],
        [1, 4],
        [2, 3],
        [3, 4]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 2)


def iterate_hunyuan_text_embedder(node_dict):

    relevant_nodes = [
        "model.text_embedder.linear_1.weight_quantizer",
        "model.text_embedder.linear_2.weight_quantizer",
        ]
        
    adj_list = [
        [0, 1],
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def iterate_hunyuan(pkl_path=PKL_FILE_PATH, nodes=False):
    with open(pkl_path, "rb") as f:
        qnn_dict = pickle.load(f)

    node_dict = get_all_node_variants(qnn_dict, net="hunyuan")

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
    module_dict['pos_embed'] = iterate_hunyuan_pos_embed(node_dict)
    module_dict['time_embedding'] = iterate_hunyuan_time_embedding(node_dict)
    module_dict['text_embedder'] = iterate_hunyuan_text_embedder(node_dict)
    module_dict['proj_out'] = iterate_hunyuan_proj_out(node_dict)

    for i in range(40):
        module_dict[f'hunyuan_blk_{i}_selfattn'] = iterate_hunyuan_selfattn(node_dict, i)
        module_dict[f'pixart_blk_{i}_crossattn1'] = iterate_hunyuan_crossattn1(node_dict, i)
        module_dict[f'pixart_blk_{i}_crossattn2'] = iterate_hunyuan_crossattn2(node_dict, i)
        module_dict[f'hunyuan_blk_{i}_ff'] = iterate_hunyuan_ff(node_dict, i)
        if i > 20:
            module_dict[f'hunyuan_blk_{i}_skip'] = iterate_hunyuan_skip(node_dict, i)

    assert len(node_dict.keys()) == 0, node_dict.keys()
    return module_dict

if __name__ == "__main__":


    with open(PKL_FILE_PATH, "rb") as f:
        qnn_dict = pickle.load(f)


    node_dict = get_all_node_variants(qnn_dict, net="hunyuan")

    running_sum = 0
    running_sum += len(iterate_hunyuan_time_embedding(node_dict))
    running_sum += len(iterate_hunyuan_text_embedder(node_dict))
    running_sum += len(iterate_hunyuan_proj_out(node_dict))
    running_sum += len(iterate_hunyuan_pos_embed(node_dict))

    for block_idx in range(40):
        running_sum += len(iterate_hunyuan_selfattn(node_dict, block_idx))
        running_sum += len(iterate_hunyuan_crossattn1(node_dict, block_idx))
        running_sum += len(iterate_hunyuan_crossattn2(node_dict, block_idx))
        running_sum += len(iterate_hunyuan_ff(node_dict, block_idx))

    for block_idx in range(21, 40):
        running_sum += len(iterate_hunyuan_skip(node_dict, block_idx))

    print(running_sum)
    assert len(node_dict.keys()) == 0, node_dict.keys()

