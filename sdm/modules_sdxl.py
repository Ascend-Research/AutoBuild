import networkx as nx
import torch as t
from sdm.graph_util import get_all_node_variants, gen_combine_node
from sdm.modules_util import _generate_variant_digraphs
from sdm.graph_sdxl import _get_stage_code, _get_block_code, _get_layer_code, _get_tsx_num_code
from itertools import product
from collections import OrderedDict
import pickle


PKL_FILE_PATH = "cache/sdxl_mq.pkl"

STAGE_BLK_IDX_TF_BLK = {
    "down_blocks": [[], [2, 2], [10, 10]],
    "mid_block": [[10]],
    "up_blocks": [[10, 10, 10], [2, 2, 2], []]
}

STAGE_BLK_IDX_RES_BLK = {
    "down_blocks": [2, 2, 2],
    "mid_block": [2],
    "up_blocks": [3, 3, 3]
}


def _iterate_sdxl_input_res_blk(node_dict, prefix):
    relevant_nodes = [
        f"{prefix}.conv1.weight_quantizer", # 0
        f"{prefix}.time_emb_proj.weight_quantizer", # 1
        f"{prefix}.conv2.weight_quantizer", # 2
    ]

    adj_list = [
        [0, 2],
        [1, 2],
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def _iterate_sdxl_input_skip_res_blk(node_dict, block_idx, unit_idx, prefix):
    relevant_nodes = [
        f"{prefix}.dummy_input", # 0
        f"model.down_blocks.{unit_idx-1}.downsamplers.{block_idx}.conv.weight_quantizer", # 1
        f"{prefix}.conv1.weight_quantizer", # 2
        f"{prefix}.time_emb_proj.weight_quantizer", # 3
        f"{prefix}.conv2.weight_quantizer", # 4
        f"{prefix}.conv_shortcut.weight_quantizer", # 5
        f"{prefix}.add" # 6
    ]

    stage_code = _get_stage_code(relevant_nodes[5])
    block_code = _get_block_code(relevant_nodes[5])
    layer_code = _get_layer_code("add")

    node_dict[relevant_nodes[6]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, -1, layer_code])])}
    node_dict[relevant_nodes[0]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, -1, layer_code])])}

    adj_list = [
        [0, 1],
        [0, 5],
        [1, 2],
        [1, 5],
        [2, 4],
        [3, 4],
        [4, 6],
        [5, 6]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 2)


def _iterate_sdxl_output_res_blk(node_dict, prefix):
    relevant_nodes = [
        f"{prefix}.dummy_input", # 0
        f"{prefix}.conv1.weight_quantizer", # 1
        f"{prefix}.time_emb_proj.weight_quantizer", # 2
        f"{prefix}.conv1.weight_quantizer_0", # 3 - NOTE SDXL shortcut is a bit different from 1.5
        f"{prefix}.conv2.weight_quantizer", # 4
        f"{prefix}.conv_shortcut.weight_quantizer", # 5
        f"{prefix}.add" # 6
    ]

    stage_code = _get_stage_code(relevant_nodes[5])
    block_code = _get_block_code(relevant_nodes[5])
    layer_code = _get_layer_code("add")

    node_dict[relevant_nodes[6]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, -1, layer_code])])}
    node_dict[relevant_nodes[0]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, -1, layer_code])])}

    adj_list = [
        [0, 1],
        [0, 5],
        [1, 4],
        [2, 4],
        [3, 4],
        [4, 6],
        [5, 6],
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 2)


def iterate_sdxl_res_blk(node_dict, block_idx, unit_idx, stage):
    prefix = _sdxl_resblock_prefix(block_idx, unit_idx, stage)
    if stage in ["down_blocks", "mid_block"]:
        if stage == "down_blocks" and unit_idx in [1, 2] and block_idx == 0:
            return _iterate_sdxl_input_skip_res_blk(node_dict, block_idx, unit_idx, prefix)
        else:
            if stage == "mid_block":
                prefix = prefix.replace('mid_block.0.', 'mid_block.')
            return _iterate_sdxl_input_res_blk(node_dict, prefix)
    else:
        return _iterate_sdxl_output_res_blk(node_dict, prefix)



def iterate_sdxl_attn2(node_dict, tblk_idx, block_idx, unit_idx, stage):
    prefix = _sdxl_transformer_prefix(tblk_idx, block_idx, unit_idx, stage)

    relevant_nodes = [
        f"{prefix}.attn2.to_q.weight_quantizer", #0
        f"{prefix}.attn2.to_k.weight_quantizer", #1
        f"{prefix}.attn2.to_v.weight_quantizer", #2
        f"{prefix}.attn2.mul_qk", # 3
        f"{prefix}.attn2.mul_vsm", # 4
        f"{prefix}.attn2.to_out.0.weight_quantizer", #5
    ]

    stage_code = _get_stage_code(relevant_nodes[3])
    block_code = _get_block_code(relevant_nodes[3])
    tf_code = _get_tsx_num_code(prefix)
    layer_code = _get_layer_code("add")

    node_dict[relevant_nodes[3]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, tf_code, layer_code])])}
    node_dict[relevant_nodes[4]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, tf_code, layer_code])])}

    adj_list = [
        [0, 3],
        [1, 3],
        [2, 4],
        [3, 4],
        [4, 5],
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 3)


def iterate_sdxl_attn1(node_dict, tblk_idx, block_idx, unit_idx, stage):
    prefix = _sdxl_transformer_prefix(tblk_idx, block_idx, unit_idx, stage)

    if tblk_idx == 0:
        adj_prefix = prefix.replace("transformer_blocks.0", "proj_in.weight_quantizer")
    else:
        adj_prefix = prefix.replace(f"transformer_blocks.{tblk_idx}", f"transformer_blocks.{tblk_idx-1}.add_ff")
    relevant_nodes = [
        f"{adj_prefix}", # 0
        f"{prefix}.attn1.to_q.weight_quantizer", #1
        f"{prefix}.attn1.to_k.weight_quantizer", #2
        f"{prefix}.attn1.to_v.weight_quantizer", #3
        f"{prefix}.attn1.mul_qk", # 4
        f"{prefix}.attn1.mul_vsm", # 5
        f"{prefix}.attn1.to_out.0.weight_quantizer", #6
    ]
    stage_code = _get_stage_code(relevant_nodes[4])
    block_code = _get_block_code(relevant_nodes[4])
    tf_code = _get_tsx_num_code(prefix)
    layer_code = _get_layer_code("add")

    if tblk_idx > 0:
        node_dict[relevant_nodes[0]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, tf_code, layer_code])])}
    node_dict[relevant_nodes[4]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, tf_code, layer_code])])}
    node_dict[relevant_nodes[5]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, tf_code, layer_code])])}

    adj_list = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [2, 4],
        [3, 5],
        [4, 5],
        [5, 6],
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 3)


def iterate_sdxl_ff(node_dict, tblk_idx, block_idx, unit_idx, stage):
    prefix = _sdxl_transformer_prefix(tblk_idx, block_idx, unit_idx, stage)
    
    relevant_nodes = [
        f"{prefix}.ff.net.0.proj.weight_quantizer",
        f"{prefix}.ff.net.2.weight_quantizer", 
    ]

    adj_list = [
        [0, 1],
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def _sdxl_transformer_prefix(tblk_idx, block_idx, unit_idx, stage):
    if stage == "mid_block":
        return f"model.{stage}.attentions.{block_idx}.transformer_blocks.{tblk_idx}"
    return f"model.{stage}.{unit_idx}.attentions.{block_idx}.transformer_blocks.{tblk_idx}"
    

def _sdxl_resblock_prefix(block_idx, unit_idx, stage):
    return f"model.{stage}.{unit_idx}.resnets.{block_idx}"
    

def iterate_sdxl_attn_proj_out(node_dict, block_idx, unit_idx, stage):
    if stage == "mid_block":
        node_name = f"model.{stage}.attentions.{block_idx}.proj_out.weight_quantizer"
    else:
        node_name = f"model.{stage}.{unit_idx}.attentions.{block_idx}.proj_out.weight_quantizer"

    relevant_nodes = [node_name]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)


def iterate_sdxl_output_blk(node_dict):

    relevant_nodes = [
        "model.conv_out.weight_quantizer",
    ]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)

def iterate_sdxl_input_blk(node_dict):

    relevant_nodes = [
        "model.conv_in.weight_quantizer",
    ]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)


def iterate_sdxl_upsample_layer(node_dict, block_idx):
    assert block_idx in [0, 1]

    relevant_nodes = [
        f"model.up_blocks.{block_idx}.upsamplers.0.conv.weight_quantizer"
    ]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)


def iterate_sdxl_time_embedding(node_dict):

    relevant_nodes = [
        "model.time_embedding.linear_1.weight_quantizer",
        "model.time_embedding.linear_2.weight_quantizer",
        "model.add_embedding.linear_1.weight_quantizer",
        "model.add_embedding.linear_2.weight_quantizer",
        "model.add_time.add"
        ]
    
    stage_code = _get_stage_code(relevant_nodes[4])
    block_code = _get_block_code(relevant_nodes[4])
    layer_code = _get_layer_code("add")

    node_dict[relevant_nodes[4]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, -1, layer_code])])}
    
    adj_list = [
        [0, 1],
        [2, 3],
        [1, 4],
        [3, 4]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 2)


def sdxl_transformer_blk_stage_iterator():
    stage_triples = []
    for s, blk_num_list in STAGE_BLK_IDX_TF_BLK.items():
        for u, blk in enumerate(blk_num_list):
            if len(blk) == 0:
                continue
            for su, num_tf_blks in enumerate(blk):
                temp = product([s], [u], [su], range(num_tf_blks))
                stage_triples.append(list(temp))
    return stage_triples


def sdxl_resnet_blk_stage_iterator():
    stage_triples = []
    for s, blk_num_list in STAGE_BLK_IDX_RES_BLK.items():
        for u, num_blks in enumerate(blk_num_list):
            stage_triples.append(product([s], [u], range(num_blks)))
    return stage_triples


def iterate_sdxl(pkl_path=PKL_FILE_PATH, nodes=False):
    with open(pkl_path, "rb") as f:
        qnn_dict = pickle.load(f)

    node_dict = get_all_node_variants(qnn_dict, net="sdxl")
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
    module_dict['time_embedding'] = iterate_sdxl_time_embedding(node_dict)
    module_dict['conv_in'] = iterate_sdxl_input_blk(node_dict)
    module_dict['conv_out'] = iterate_sdxl_output_blk(node_dict)
    for i in range(2):
        module_dict[f"upsampler_{i}"] = iterate_sdxl_upsample_layer(node_dict, i)

    all_res_combos = sdxl_resnet_blk_stage_iterator()
    for stage_iter in all_res_combos:
        for combo in stage_iter:
            module_dict[f"{combo[0]}_u{combo[1]}_b{combo[2]}_resblk"] = iterate_sdxl_res_blk(node_dict, block_idx=combo[2], unit_idx=combo[1], stage=combo[0])
    
    all_tf_combos = sdxl_transformer_blk_stage_iterator()
    for stage_iter in all_tf_combos:
        for combo in stage_iter:
            module_dict[f"{combo[0]}_u{combo[1]}_b{combo[2]}_tf{combo[3]}_attn1"] = iterate_sdxl_attn1(node_dict, tblk_idx=combo[3], block_idx=combo[2], unit_idx=combo[1], stage=combo[0])
            module_dict[f"{combo[0]}_u{combo[1]}_b{combo[2]}_tf{combo[3]}_attn2"] = iterate_sdxl_attn2(node_dict, tblk_idx=combo[3], block_idx=combo[2], unit_idx=combo[1], stage=combo[0])
            module_dict[f"{combo[0]}_u{combo[1]}_b{combo[2]}_tf{combo[3]}_ff"] = iterate_sdxl_ff(node_dict, tblk_idx=combo[3], block_idx=combo[2], unit_idx=combo[1], stage=combo[0])
            if combo[3] == 0:
                module_dict[f"{combo[0]}_u{combo[1]}_b{combo[2]}_tf{combo[3]}_proj_out"] = iterate_sdxl_attn_proj_out(node_dict, block_idx=combo[2], unit_idx=combo[1], stage=combo[0])
    
    return module_dict


if __name__ == "__main__":

    with open(PKL_FILE_PATH, "rb") as f:
        qnn_dict = pickle.load(f)

    node_dict = get_all_node_variants(qnn_dict, net="sdxl")

    running_sum = 0
    running_sum += len(iterate_sdxl_time_embedding(node_dict))

    running_sum += len(iterate_sdxl_input_blk(node_dict))
    running_sum += len(iterate_sdxl_output_blk(node_dict))
    for i in range(2):
        running_sum += len(iterate_sdxl_upsample_layer(node_dict, i))


    all_combos = sdxl_resnet_blk_stage_iterator()
    for stage_iter in all_combos:
        for combo in stage_iter:
            temp = iterate_sdxl_res_blk(node_dict, block_idx=combo[2], unit_idx=combo[1], stage=combo[0])
            running_sum += len(temp)

    # Attentions:
    all_combos = sdxl_transformer_blk_stage_iterator()
    for stage_iter in all_combos:
        for combo in stage_iter:
            running_sum += len(iterate_sdxl_attn1(node_dict, tblk_idx=combo[3], block_idx=combo[2], unit_idx=combo[1], stage=combo[0]))
            running_sum += len(iterate_sdxl_attn2(node_dict, tblk_idx=combo[3], block_idx=combo[2], unit_idx=combo[1], stage=combo[0]))
            running_sum += len(iterate_sdxl_ff(node_dict, tblk_idx=combo[3], block_idx=combo[2], unit_idx=combo[1], stage=combo[0]))
            if combo[3] == 0:
                running_sum += len(iterate_sdxl_attn_proj_out(node_dict, block_idx=combo[2], unit_idx=combo[1], stage=combo[0]))

    print(running_sum)
    assert len(node_dict.keys()) == 0, node_dict.keys()

