import networkx as nx
import torch as t
from sdm.graph_util import get_all_node_variants, gen_combine_node
from sdm.modules_util import _generate_variant_digraphs
from sdm.graph_sdv15 import _get_stage_code, _get_block_code, _get_layer_code
from itertools import product
from collections import OrderedDict
import pickle


PKL_FILE_PATH = "cache/sdv15_mq.pkl"

STAGE_BLK_IDX_TF_BLK = {
    "input": [1, 2, 4, 5, 7, 8],
    "middle": [1],
    "output": [3, 4, 5, 6, 7, 8, 9, 10, 11]
}

STAGE_BLK_IDX_RES_BLK = {
    "input": [1, 2, 4, 5, 7, 8, 10, 11],
    "middle": [0, 2],
    "output": list(range(12))
}


def _iterate_sdv15_input_res_blk(node_dict, prefix):
    relevant_nodes = [
        f"{prefix}.in_layers.2.weight_quantizer", # 0
        f"{prefix}.emb_layers.1.weight_quantizer", # 1
        f"{prefix}.out_layers.3.weight_quantizer", # 2
    ]

    adj_list = [
        [0, 2],
        [1, 2],
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def iterate_sdv15_op9(node_dict):
    
    relevant_nodes = [
        f"model.input_blocks.9.0.op.weight_quantizer"
    ]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)


def _iterate_sdv15_input_skip_res_blk(node_dict, block_idx, prefix):
    adj_prefix = prefix.replace(f"input_blocks.{block_idx}.0", f"input_blocks.{block_idx-1}.0.op.weight_quantizer")
    
    relevant_nodes = [
        adj_prefix, # 0
        f"{prefix}.in_layers.2.weight_quantizer", # 1
        f"{prefix}.emb_layers.1.weight_quantizer", # 2
        f"{prefix}.out_layers.3.weight_quantizer", # 3
        f"{prefix}.skip_connection.weight_quantizer", # 4
        f"{prefix}.add" # 5
    ]

    stage_code = _get_stage_code(relevant_nodes[5])
    block_code = _get_block_code(relevant_nodes[5])
    layer_code = _get_layer_code("add")

    node_dict[relevant_nodes[5]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, layer_code])])}

    adj_list = [
        [0, 1],
        [0, 4],
        [1, 3],
        [2, 3],
        [3, 5],
        [4, 5]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 2)


def _iterate_sdv15_output_res_blk(node_dict, prefix):
    relevant_nodes = [
        f"{prefix}.in_layers.2.weight_quantizer", # 0
        f"{prefix}.emb_layers.1.weight_quantizer", # 1
        f"{prefix}.out_layers.3.weight_quantizer", # 2
        f"{prefix}.skip_connection.weight_quantizer", # 3
        f"{prefix}.skip_connection.weight_quantizer_0", # 4
        f"{prefix}.add_qdiff", # 5
        f"{prefix}.add" # 6
    ]

    stage_code = _get_stage_code(relevant_nodes[5])
    block_code = _get_block_code(relevant_nodes[5])
    layer_code = _get_layer_code("add")

    node_dict[relevant_nodes[5]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, layer_code])])}
    node_dict[relevant_nodes[6]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, layer_code])])}

    adj_list = [
        [0, 2],
        [1, 2],
        [3, 5],
        [4, 5],
        [2, 6],
        [5, 6],
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 2)


def iterate_sdv15_res_blk(node_dict, block_idx, stage):
    prefix = _sdv15_resblock_prefix(block_idx, stage)
    if stage in ["input", "middle"]:
        if stage == "input" and block_idx in [4, 7]:
            return _iterate_sdv15_input_skip_res_blk(node_dict, block_idx, prefix)
        else:
            if stage == "middle":
                prefix = ".".join(prefix.split(".")[:-1])
            return _iterate_sdv15_input_res_blk(node_dict, prefix)
    else:
        return _iterate_sdv15_output_res_blk(node_dict, prefix)



def iterate_sdv15_attn2(node_dict, block_idx, stage):
    prefix = _sdv15_transformer_prefix(block_idx, stage)

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
    layer_code = _get_layer_code("add")

    node_dict[relevant_nodes[3]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, layer_code])])}
    node_dict[relevant_nodes[4]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, layer_code])])}

    adj_list = [
        [0, 3],
        [1, 3],
        [2, 4],
        [3, 4],
        [4, 5],
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 3)


def iterate_sdv15_attn1(node_dict, block_idx, stage):
    prefix = _sdv15_transformer_prefix(block_idx, stage)

    adj_prefix = prefix.replace("transformer_blocks.0", "")
    relevant_nodes = [
        f"{adj_prefix}proj_in.weight_quantizer", # 0
        f"{prefix}.attn1.to_q.weight_quantizer", #1
        f"{prefix}.attn1.to_k.weight_quantizer", #2
        f"{prefix}.attn1.to_v.weight_quantizer", #3
        f"{prefix}.attn1.mul_qk", # 4 
        f"{prefix}.attn1.mul_vsm", # 5
        f"{prefix}.attn1.to_out.0.weight_quantizer", #6
    ]
    stage_code = _get_stage_code(relevant_nodes[4])
    block_code = _get_block_code(relevant_nodes[4])
    layer_code = _get_layer_code("add")

    node_dict[relevant_nodes[4]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, layer_code])])}
    node_dict[relevant_nodes[5]] = {"combine-16": t.cat([gen_combine_node(), t.Tensor([stage_code, block_code, layer_code])])}

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


def iterate_sdv15_ff(node_dict, block_idx, stage):
    
    prefix = _sdv15_transformer_prefix(block_idx, stage)
    relevant_nodes = [
        f"{prefix}.ff.net.0.proj.weight_quantizer",
        f"{prefix}.ff.net.2.weight_quantizer", 
    ]

    adj_list = [
        [0, 1],
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)


def _sdv15_transformer_prefix(block_idx, stage):
    assert stage in STAGE_BLK_IDX_TF_BLK.keys()
    assert block_idx >= 0
    assert block_idx in STAGE_BLK_IDX_TF_BLK[stage]
    if stage == "middle":
        return "model.middle_block.1.transformer_blocks.0"
    else:
        return f"model.{stage}_blocks.{block_idx}.1.transformer_blocks.0"
    

def _sdv15_resblock_prefix(block_idx, stage):
    assert stage in STAGE_BLK_IDX_RES_BLK.keys()
    assert block_idx >= 0
    assert block_idx in STAGE_BLK_IDX_RES_BLK[stage]
    if stage == "middle":
        return f"model.middle_block.{block_idx}.0"
    else:
        return f"model.{stage}_blocks.{block_idx}.0"
    

def iterate_sdv15_attn_proj_out(node_dict, block_idx, stage):
    
    prefix = _sdv15_transformer_prefix(block_idx, stage).replace("transformer_blocks.0", "")
    relevant_nodes = [
        f"{prefix}proj_out.weight_quantizer"
    ]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)


def iterate_sdv15_output_blk(node_dict):

    relevant_nodes = [
        "model.out.2.weight_quantizer",
    ]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)

def iterate_sdv15_input_blk(node_dict):

    relevant_nodes = [
        "model.input_blocks.0.0.weight_quantizer",
    ]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)


def iterate_sdv15_upsample_layer(node_dict, block_idx):
    assert block_idx in [2, 5, 8]

    if block_idx == 2:
        relevant_nodes = [
            f"model.output_blocks.{block_idx}.1.conv.weight_quantizer"
        ]
    else:
        relevant_nodes = [
            f"model.output_blocks.{block_idx}.2.conv.weight_quantizer"
        ]

    adj_list = [
    ]
    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 0)


def iterate_sdv15_time_embedding(node_dict):

    relevant_nodes = [
        "model.time_embed.0.weight_quantizer",
        "model.time_embed.2.weight_quantizer"
        ]
        
    adj_list = [
        [0, 1]
    ]

    return _generate_variant_digraphs(node_dict, relevant_nodes, adj_list, 1)
    

def sdv15_transformer_blk_stage_iterator():
    list_of_stage_pairs = []
    for s, blist in STAGE_BLK_IDX_TF_BLK.items():
        list_of_stage_pairs.append(product([s], blist))
    return list_of_stage_pairs


def sdv15_resnet_blk_stage_iterator():
    list_of_stage_pairs = []
    for s, blist in STAGE_BLK_IDX_RES_BLK.items():
        list_of_stage_pairs.append(product([s], blist))
    return list_of_stage_pairs


def iterate_sdv15(pkl_path=PKL_FILE_PATH, nodes=False):
    with open(pkl_path, "rb") as f:
        qnn_dict = pickle.load(f)

    node_dict = get_all_node_variants(qnn_dict, net="sdv15")

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
    module_dict['time_embedding'] = iterate_sdv15_time_embedding(node_dict)
    module_dict['conv_in'] = iterate_sdv15_input_blk(node_dict)
    module_dict['conv_out'] = iterate_sdv15_output_blk(node_dict)
    module_dict['op9'] = iterate_sdv15_op9(node_dict)

    for out_idx in [2, 5, 8]:
        module_dict[f"upsample_blk_{out_idx}"] = iterate_sdv15_upsample_layer(node_dict, out_idx)

    all_tsx_combos = sdv15_transformer_blk_stage_iterator()
    for stage_iter in all_tsx_combos:
        for combo in stage_iter:
            module_dict[f"{combo[0]}_b{combo[1]}_tf_attn1"] = iterate_sdv15_attn1(node_dict, block_idx=combo[1], stage=combo[0])

            module_dict[f"{combo[0]}_b{combo[1]}_tf_attn2"] = iterate_sdv15_attn2(node_dict, block_idx=combo[1], stage=combo[0])
            
            module_dict[f"{combo[0]}_b{combo[1]}_tf_ff"] = iterate_sdv15_ff(node_dict, block_idx=combo[1], stage=combo[0])

            module_dict[f"{combo[0]}_b{combo[1]}_tf_proj_out"] = iterate_sdv15_attn_proj_out(node_dict, block_idx=combo[1], stage=combo[0])

    all_combos = sdv15_resnet_blk_stage_iterator()
    for stage_iter in all_combos:
        for combo in stage_iter:
            module_dict[f"{combo[0]}_b{combo[1]}_resblk"]= iterate_sdv15_res_blk(node_dict, block_idx=combo[1], stage=combo[0])
    assert len(node_dict.keys()) == 0, node_dict.keys()
    return module_dict


if __name__ == "__main__":

    with open(PKL_FILE_PATH, "rb") as f:
        qnn_dict = pickle.load(f)

    node_dict = get_all_node_variants(qnn_dict, net="sdv15")

    running_sum = 0
    running_sum += len(iterate_sdv15_time_embedding(node_dict))
    running_sum += len(iterate_sdv15_input_blk(node_dict))
    running_sum += len(iterate_sdv15_output_blk(node_dict))
    running_sum += len(iterate_sdv15_op9(node_dict))

    for out_idx in [2, 5, 8]:
        running_sum += len(iterate_sdv15_upsample_layer(node_dict, out_idx))

    # Attentions:
    all_combos = sdv15_transformer_blk_stage_iterator()
    for stage_iter in all_combos:
        for combo in stage_iter:
            running_sum += len(iterate_sdv15_attn1(node_dict, block_idx=combo[1], stage=combo[0]))
            running_sum += len(iterate_sdv15_attn2(node_dict, block_idx=combo[1], stage=combo[0]))
            running_sum += len(iterate_sdv15_ff(node_dict, block_idx=combo[1], stage=combo[0]))
            running_sum += len(iterate_sdv15_attn_proj_out(node_dict, block_idx=combo[1], stage=combo[0]))

    all_combos = sdv15_resnet_blk_stage_iterator()
    for stage_iter in all_combos:
        for combo in stage_iter:
            temp = iterate_sdv15_res_blk(node_dict, block_idx=combo[1], stage=combo[0])
            running_sum += len(temp)
    print(running_sum)
    assert len(node_dict.keys()) == 0, node_dict.keys()