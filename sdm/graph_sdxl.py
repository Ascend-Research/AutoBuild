import torch as t
import torch_geometric as tg
import networkx as nx
from sdm.graph_util import convert_universal_layer_feats_to_vector, gen_combine_node, visualize_graph
from collections import OrderedDict


SDXL_IN_TRANSFORMER_BLK_IDX = [[], [2, 2], [10, 10]]
SDXL_MID_TRANSFORMER_BLK_IDX = [[10]]
SDXL_OUT_TRANSFORMER_BLK_IDX = [[10, 10, 10], [2, 2, 2], []]

SDXL_IN_RESBLK_IDX = [2, 2, 2]
SDXL_MID_RESBLK_IDX = [2]
SDXL_OUT_RESBLK_IDX = [3, 3, 3]

SDXL_LAYERS_LUT = OrderedDict()
SDXL_LAYERS_LUT["time_embedding.linear_1"] = 0
SDXL_LAYERS_LUT["time_embedding.linear_2"] = 1
SDXL_LAYERS_LUT["conv_in"] = 2
SDXL_LAYERS_LUT["conv1"] = 3
SDXL_LAYERS_LUT["time_emb_proj"] = 4
SDXL_LAYERS_LUT["conv2"] = 5
SDXL_LAYERS_LUT["conv_shortcut"] = 6
SDXL_LAYERS_LUT["proj_in"] = 7
SDXL_LAYERS_LUT["proj_out"] = 8
SDXL_LAYERS_LUT["attn1.to_q"] = 9
SDXL_LAYERS_LUT["attn1.to_k"] = 10
SDXL_LAYERS_LUT["attn1.to_v"] = 11
SDXL_LAYERS_LUT["attn1.to_q"] = 12
SDXL_LAYERS_LUT["attn1.to_out.0"] = 13
SDXL_LAYERS_LUT["attn2.to_q"] = 14
SDXL_LAYERS_LUT["attn2.to_k"] = 15
SDXL_LAYERS_LUT["attn2.to_v"] = 16
SDXL_LAYERS_LUT["attn2.to_q"] = 17
SDXL_LAYERS_LUT["attn2.to_out.0"] = 18
SDXL_LAYERS_LUT["ff.net.0.proj"] = 19
SDXL_LAYERS_LUT["ff.net.2"] = 20
SDXL_LAYERS_LUT["conv_out"] = 21
SDXL_LAYERS_LUT["downsamplers.0.conv"] = 22
SDXL_LAYERS_LUT["upsamplers.0.conv"] = 23
SDXL_LAYERS_LUT["add_embedding.linear_1"] = 24
SDXL_LAYERS_LUT["add_embedding.linear_2"] = 25
SDXL_LAYERS_LUT["add"] = 26
SDXL_LAYERS_LUT["matmul"] = 27


def convert_sdxl_layer_to_vector(dit_k, dit_v):
    if type(dit_v) == t.Tensor:
        return dit_v
    
    stage_code = _get_stage_code(dit_k)
    blk_code = _get_block_code(dit_k)
    layer_code = _get_layer_code(dit_k)
    universal_feats = convert_universal_layer_feats_to_vector(dit_v)
    sdxl_specific_feats = t.Tensor([stage_code, blk_code, layer_code])
    return t.cat([universal_feats, sdxl_specific_feats])


def _get_layer_code(layer_key):
    layer_key = layer_key.replace(".weight_quantizer_0", "").replace(".weight_quantizer", "")
    for key in SDXL_LAYERS_LUT.keys():
        if layer_key.endswith(key):
            return SDXL_LAYERS_LUT[key]
        
        
def _get_stage_code(key):
    # conv_in is initial conv not part of any block
    if "down_blocks" in key or "time_embedding" in key or "add_embedding" in key or "model.conv_in" in key or "add_time" in key:
        return 0
    elif "mid_block" in key:
        return 1
    elif "up_blocks" in key or "conv_out" in key:
        return 2
    else:
        raise NotImplementedError


def _get_tsx_num_code(key):
    if "transformer_blocks" not in key:
        return -1
    else:
        try:
            s_key = key.split(".")
            if s_key[1] == "mid_block":
                return int(s_key[5])
            else:
                return int(s_key[6])
        except:
            print(s_key)
    

def _get_block_code(key):
    str_code = key.split(".")[2]
    try:
        return int(str_code)
    except:
        return -1


def _gen_attn_blk(sdxl_quant_dict, prefix, ca=False):

    attn = nx.DiGraph()

    q_key = ".".join([prefix, "to_q", "weight_quantizer"])
    stage_code = _get_stage_code(q_key)
    block_code = _get_block_code(q_key)
    tf_code = _get_tsx_num_code(q_key)
    q_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[q_key])
    q_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(q_key)])
    q_feats = t.cat([q_universal_feats, q_specific_feats])
    del sdxl_quant_dict[q_key]
    attn.add_node(q_key, all=q_feats)

    k_key = ".".join([prefix, "to_k", "weight_quantizer"])
    k_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[k_key])
    k_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(k_key)])
    k_feats = t.cat([k_universal_feats, k_specific_feats])
    del sdxl_quant_dict[k_key]
    attn.add_node(k_key, all=k_feats)

    v_key = ".".join([prefix, "to_v", "weight_quantizer"])
    v_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[v_key])
    v_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(v_key)])
    v_feats = t.cat([v_universal_feats, v_specific_feats])
    del sdxl_quant_dict[v_key]
    attn.add_node(v_key, all=v_feats)

    out_key = ".".join([prefix, "to_out.0", "weight_quantizer"])
    out_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[out_key])
    out_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(out_key)])
    out_feats = t.cat([out_universal_feats, out_specific_feats])
    del sdxl_quant_dict[out_key]
    attn.add_node(out_key, all=out_feats)

    qk_key = prefix + ".mul_qk"
    qk_universal_feats = gen_combine_node()
    qk_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code("matmul")])
    qk_feats = t.cat([qk_universal_feats, qk_specific_feats])
    attn.add_node(qk_key, all=qk_feats)

    vsm_key = prefix + ".mul_vsm"
    vsm_universal_feats = gen_combine_node()
    vsm_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code("matmul")])
    vsm_feats = t.cat([vsm_universal_feats, vsm_specific_feats])
    attn.add_node(vsm_key, all=vsm_feats)

    attn.add_edge(q_key, qk_key)
    attn.add_edge(k_key, qk_key)
    attn.add_edge(qk_key, vsm_key)
    attn.add_edge(v_key, vsm_key)
    attn.add_edge(vsm_key, out_key)
    in_dict = {'q': q_key}

    # If cross-attention, then k and v come from context which we do not encode
    if not ca:
        in_dict['k'] = k_key
        in_dict['v'] = v_key
    out_dict = {'out': out_key}
    return {'nx': attn, 'in_dict': in_dict, 'out_dict': out_dict}


def _gen_ff_blk(sdxl_quant_dict, prefix):
    ff = nx.DiGraph()
    ff1_key = ".".join([prefix, "ff.net.0.proj", "weight_quantizer"])
    ff1_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[ff1_key])
    stage_code = _get_stage_code(ff1_key)
    block_code = _get_block_code(ff1_key)
    tf_code = _get_tsx_num_code(ff1_key)
    ff1_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(ff1_key)])
    ff1_feats = t.cat([ff1_universal_feats, ff1_specific_feats])
    del sdxl_quant_dict[ff1_key]
    ff.add_node(ff1_key, all=ff1_feats)

    ff2_key = ".".join([prefix, "ff.net.2", "weight_quantizer"])
    ff2_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[ff2_key])
    ff2_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(ff2_key)])
    ff2_feats = t.cat([ff2_universal_feats, ff2_specific_feats])
    del sdxl_quant_dict[ff2_key]
    ff.add_node(ff2_key, all=ff2_feats)

    ff.add_edge(ff1_key, ff2_key)
    in_dict = {'ff1': ff1_key}
    out_dict = {'ff2': ff2_key}
    return {'nx': ff, 'in_dict': in_dict, 'out_dict': out_dict}


# Prefix is like "model.input_blocks.1.1.transformer_blocks"
def _gen_sdxl_transformer_block(sdxl_quant_dict, prefix):
    transformer = nx.DiGraph()

    sa_dict = _gen_attn_blk(sdxl_quant_dict, prefix=".".join([prefix, "attn1"]), ca=False)
    transformer = sa_dict['nx']
    transformer_in_dict = sa_dict['in_dict']

    sa_add_key = prefix + ".add_sa"
    sa_add_universal_features = gen_combine_node()
    sa_add_specific_features = t.Tensor([_get_stage_code(sa_add_key), _get_block_code(sa_add_key), _get_tsx_num_code(sa_add_key), _get_layer_code('add')])
    sa_add_features = t.cat([sa_add_universal_features, sa_add_specific_features])
    transformer.add_node(sa_add_key, all=sa_add_features)
    transformer_in_dict['add_sa'] = sa_add_key
    transformer.add_edge(sa_dict['out_dict']['out'], sa_add_key)

    ca_dict = _gen_attn_blk(sdxl_quant_dict, prefix=".".join([prefix, "attn2"]), ca=True)
    transformer = nx.union(transformer, ca_dict['nx'])
    for j in ca_dict['in_dict'].values():
        transformer.add_edge(sa_add_key, j)

    ca_add_key = prefix + ".add_ca"
    ca_add_universal_features = gen_combine_node()
    ca_add_specific_features = t.Tensor([_get_stage_code(ca_add_key), _get_block_code(ca_add_key), _get_tsx_num_code(ca_add_key), _get_layer_code('add')])
    ca_add_features = t.cat([ca_add_universal_features, ca_add_specific_features])
    transformer.add_node(ca_add_key, all=ca_add_features)
    transformer.add_edge(sa_add_key, ca_add_key)
    transformer.add_edge(ca_dict['out_dict']['out'], ca_add_key)
    
    ff_dict = _gen_ff_blk(sdxl_quant_dict, prefix=prefix)
    transformer = nx.union(transformer, ff_dict['nx'])
    for j in ff_dict['in_dict'].values():
        transformer.add_edge(ca_add_key, j)

    ff_add_key = prefix + ".add_ff"
    ff_add_universal_features = gen_combine_node()
    ff_add_specific_features = t.Tensor([_get_stage_code(ff_add_key), _get_block_code(ff_add_key), _get_tsx_num_code(ff_add_key), _get_layer_code('add')])
    ff_add_features = t.cat([ff_add_universal_features, ff_add_specific_features])
    transformer.add_node(ff_add_key, all=ff_add_features)
    transformer.add_edge(ca_add_key, ff_add_key)
    transformer.add_edge(ff_dict['out_dict']['ff2'], ff_add_key)

    transformer_out_dict = {'add_ff': ff_add_key}

    return {'nx': transformer, 'in_dict': transformer_in_dict, 'out_dict': transformer_out_dict}


def _gen_all_sdxl_transformer_stage_blocks(sdxl_quant_dict, stage_blk_idx, stage_str):
    transformer_dict = OrderedDict()
    for stage_id in range(len(stage_blk_idx)):
        for attentions_id in range(len(stage_blk_idx[stage_id])):
            tf_graph = nx.DiGraph()
            if stage_str == "mid_block":            
                proj_in_key = f"model.{stage_str}.attentions.{attentions_id}.proj_in.weight_quantizer"
            else:
                proj_in_key = f"model.{stage_str}.{stage_id}.attentions.{attentions_id}.proj_in.weight_quantizer"
            proj_in_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[proj_in_key])
            proj_in_specific_feats = t.Tensor([_get_stage_code(proj_in_key), _get_block_code(proj_in_key), _get_tsx_num_code(proj_in_key), _get_layer_code(proj_in_key)])
            proj_in_feats = t.cat([proj_in_universal_feats, proj_in_specific_feats])
            del sdxl_quant_dict[proj_in_key]
            tf_graph.add_node(proj_in_key, all=proj_in_feats)
            cur_input = proj_in_key
            for tf_blk_id in range(stage_blk_idx[stage_id][attentions_id]): 
                input_prefix = proj_in_key.replace("proj_in.weight_quantizer", 
                                                   f"transformer_blocks.{tf_blk_id}")
                tf_dict = _gen_sdxl_transformer_block(sdxl_quant_dict, input_prefix)
                tf_graph = nx.union(tf_graph, tf_dict['nx'])
                for j in tf_dict['in_dict'].values():
                    tf_graph.add_edge(cur_input, j)
                cur_input = tf_dict['out_dict']['add_ff']

            proj_out_key = proj_in_key.replace("_in", "_out")
            proj_out_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[proj_out_key])
            proj_out_specific_feats = t.Tensor([_get_stage_code(proj_out_key), _get_block_code(proj_out_key), _get_tsx_num_code(proj_out_key), _get_layer_code(proj_out_key)])
            proj_out_feats = t.cat([proj_out_universal_feats, proj_out_specific_feats])
            del sdxl_quant_dict[proj_out_key]
            tf_graph.add_node(proj_out_key, all=proj_out_feats)
            tf_graph.add_edge(cur_input, proj_out_key)
            
            transformer_dict[proj_in_key.replace(".proj_in.weight_quantizer", "")] = {'nx': tf_graph, 'in_dict': {'proj_in': proj_in_key}, 'out_dict': {'proj_out': proj_out_key}}
    return transformer_dict


def _gen_all_sdxl_transformer_blocks(sdxl_quant_dict):

    input_tf_dict = _gen_all_sdxl_transformer_stage_blocks(sdxl_quant_dict, SDXL_IN_TRANSFORMER_BLK_IDX, "down_blocks")

    mid_tf_dict = _gen_all_sdxl_transformer_stage_blocks(sdxl_quant_dict, SDXL_MID_TRANSFORMER_BLK_IDX, "mid_block")

    output_tf_dict = _gen_all_sdxl_transformer_stage_blocks(sdxl_quant_dict, SDXL_OUT_TRANSFORMER_BLK_IDX, "up_blocks")

    return input_tf_dict, mid_tf_dict, output_tf_dict


def _gen_sdxl_resblock(sdxl_quant_dict, prefix):

    resblk = nx.DiGraph()

    in_layer_key = ".".join([prefix, "conv1", "weight_quantizer"])
    stage_code = _get_stage_code(in_layer_key)
    block_code = _get_block_code(in_layer_key)
    tf_code = -1
    in_layer_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[in_layer_key])
    in_layer_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(in_layer_key)])
    resblk.add_node(in_layer_key, all=t.cat([in_layer_universal_feats, in_layer_specific_feats]))
    del sdxl_quant_dict[in_layer_key]

    emb_layer_key = ".".join([prefix, "time_emb_proj", "weight_quantizer"])
    emb_layer_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[emb_layer_key])
    emb_layer_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(emb_layer_key)])
    resblk.add_node(emb_layer_key, all=t.cat([emb_layer_universal_feats, emb_layer_specific_feats]))
    del sdxl_quant_dict[emb_layer_key]

    out_layer_key = ".".join([prefix, "conv2", "weight_quantizer"])
    out_layer_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[out_layer_key])
    out_layer_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(out_layer_key)])
    resblk.add_node(out_layer_key, all=t.cat([out_layer_universal_feats, out_layer_specific_feats]))
    del sdxl_quant_dict[out_layer_key]

    skip_connect_key = ".".join([prefix, "conv_shortcut", "weight_quantizer"])
    skips, skips0 = False, False
    if skip_connect_key in sdxl_quant_dict.keys():
        skips = True
        skip_connect_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[skip_connect_key])
        skip_connect_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(skip_connect_key)])
        resblk.add_node(skip_connect_key, all=t.cat([skip_connect_universal_feats, skip_connect_specific_feats]))
        del sdxl_quant_dict[skip_connect_key]

    skip_connect_key0 = in_layer_key + "_0"
    if skip_connect_key0 in sdxl_quant_dict.keys():
        skips0 = True
        skip_connect0_universal_feats = convert_universal_layer_feats_to_vector(sdxl_quant_dict[skip_connect_key0])
        skip_connect0_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code(skip_connect_key0)])
        resblk.add_node(skip_connect_key0, all=t.cat([skip_connect0_universal_feats, skip_connect0_specific_feats]))
        del sdxl_quant_dict[skip_connect_key0]

    resblk.add_edge(in_layer_key, out_layer_key)
    resblk.add_edge(emb_layer_key, out_layer_key)

    in_dict = {'time_emb_proj': emb_layer_key}

    if skips: 
        input_dummy_key = prefix + ".dummy_input"        
        input_dummy_universal_feats = gen_combine_node()
        input_dummy_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code("add")])
        input_dummy_feats = t.cat([input_dummy_universal_feats, input_dummy_specific_feats])
        resblk.add_node(input_dummy_key, all=input_dummy_feats)
        resblk.add_edge(input_dummy_key, in_layer_key)
        resblk.add_edge(input_dummy_key, skip_connect_key)
        in_dict['input'] = input_dummy_key

        skip_h_key = prefix + ".add"        
        skip_h_universal_feats = gen_combine_node()
        skip_h_specific_feats = t.Tensor([stage_code, block_code, tf_code, _get_layer_code("add")])
        skip_h_feats = t.cat([skip_h_universal_feats, skip_h_specific_feats])
        resblk.add_node(skip_h_key, all=skip_h_feats)
        resblk.add_edge(out_layer_key, skip_h_key)
        resblk.add_edge(skip_connect_key, skip_h_key)
        out_dict = {'out_layers': skip_h_key}
    else:
        in_dict['input'] = in_layer_key
        out_dict = {'out_layers': out_layer_key}

    if skips0:
        in_dict['conv_shortcut0'] = skip_connect_key0
        resblk.add_edge(skip_connect_key0, out_layer_key)
    
    return {'nx': resblk, 'in_dict': in_dict, 'out_dict': out_dict}


def _gen_all_sdxl_resblocks(sdxl_quant_dict):
    input_dict, mid_dict, output_dict = OrderedDict(), OrderedDict(), OrderedDict()
    for stage_id in range(len(SDXL_IN_RESBLK_IDX)):
        for resnet_id in range(SDXL_IN_RESBLK_IDX[stage_id]):
            input_prefix = f"model.down_blocks.{stage_id}.resnets.{resnet_id}"
            input_dict[input_prefix] = _gen_sdxl_resblock(sdxl_quant_dict, input_prefix)
    
    for stage_id in range(len(SDXL_MID_RESBLK_IDX)):
        for resnet_id in range(SDXL_MID_RESBLK_IDX[stage_id]):
            mid_prefix = f"model.mid_block.resnets.{resnet_id}"
            mid_dict[mid_prefix] = _gen_sdxl_resblock(sdxl_quant_dict, mid_prefix)

    for stage_id in range(len(SDXL_OUT_RESBLK_IDX)):
        for resnet_id in range(SDXL_OUT_RESBLK_IDX[stage_id]):
            output_prefix = f"model.up_blocks.{stage_id}.resnets.{resnet_id}"
            output_dict[output_prefix] = _gen_sdxl_resblock(sdxl_quant_dict, output_prefix)

    return input_dict, mid_dict, output_dict


def construct_sdxl_tg_graph(sdxl_quant_dict, y=None):

    input_tf_dict, mid_tf_dict, output_tf_dict = _gen_all_sdxl_transformer_blocks(sdxl_quant_dict)
    input_rn_dict, mid_rn_dict, output_rn_dict = _gen_all_sdxl_resblocks(sdxl_quant_dict)

    sdxl = nx.DiGraph()
    te0_key = "model.time_embedding.linear_1.weight_quantizer"
    te0_universal_features = convert_universal_layer_feats_to_vector(sdxl_quant_dict[te0_key])
    te0_specific_features = t.Tensor([_get_stage_code(te0_key), _get_block_code(te0_key), -1, _get_layer_code(te0_key)])
    te0_features = t.cat([te0_universal_features, te0_specific_features])
    del sdxl_quant_dict[te0_key]
    sdxl.add_node(te0_key, all=te0_features)

    te2_key = "model.time_embedding.linear_2.weight_quantizer"
    te2_universal_features = convert_universal_layer_feats_to_vector(sdxl_quant_dict[te2_key])
    te2_specific_features = t.Tensor([_get_stage_code(te2_key), _get_block_code(te2_key), -1, _get_layer_code(te2_key)])
    te2_features = t.cat([te2_universal_features, te2_specific_features])
    del sdxl_quant_dict[te2_key]
    sdxl.add_node(te2_key, all=te2_features)
    sdxl.add_edge(te0_key, te2_key)

    ae0_key = "model.add_embedding.linear_1.weight_quantizer"
    ae0_universal_features = convert_universal_layer_feats_to_vector(sdxl_quant_dict[ae0_key])
    ae0_specific_features = t.Tensor([_get_stage_code(ae0_key), _get_block_code(ae0_key), -1, _get_layer_code(ae0_key)])
    ae0_features = t.cat([ae0_universal_features, ae0_specific_features])
    del sdxl_quant_dict[ae0_key]
    sdxl.add_node(ae0_key, all=ae0_features)

    ae2_key = "model.add_embedding.linear_2.weight_quantizer"
    ae2_universal_features = convert_universal_layer_feats_to_vector(sdxl_quant_dict[ae2_key])
    ae2_specific_features = t.Tensor([_get_stage_code(ae2_key), _get_block_code(ae2_key), -1, _get_layer_code(ae2_key)])
    ae2_features = t.cat([ae2_universal_features, ae2_specific_features])
    del sdxl_quant_dict[ae2_key]
    sdxl.add_node(ae2_key, all=ae2_features)
    sdxl.add_edge(ae0_key, ae2_key)

    time_add_key = "model.add_time.add"
    time_add_universal_feats = gen_combine_node()
    time_add_specific_feats = t.Tensor([_get_stage_code(time_add_key), _get_block_code(time_add_key), -1, _get_layer_code(time_add_key)])
    time_add_feats = t.cat([time_add_universal_feats, time_add_specific_feats])
    sdxl.add_node(time_add_key, all=time_add_feats)
    sdxl.add_edge(te2_key, time_add_key)
    sdxl.add_edge(ae2_key, time_add_key)

    conv_in_key = 'model.conv_in.weight_quantizer'
    conv_in_universal_features = convert_universal_layer_feats_to_vector(sdxl_quant_dict[conv_in_key])
    conv_in_specific_features = t.Tensor([_get_stage_code(conv_in_key), _get_block_code(conv_in_key), -1, _get_layer_code(conv_in_key)])
    conv_in_features = t.cat([conv_in_universal_features, conv_in_specific_features])
    del sdxl_quant_dict[conv_in_key]
    sdxl.add_node(conv_in_key, all=conv_in_features)
    prev_output_key = [conv_in_key]

    downsample_keys = {}
    for i in [0, 1]:
        ds_key = f"model.down_blocks.{i}.downsamplers.0.conv.weight_quantizer"
        ds_universal_features = convert_universal_layer_feats_to_vector(sdxl_quant_dict[ds_key])
        ds_specific_features = t.Tensor([_get_stage_code(ds_key), _get_block_code(ds_key), -1, _get_layer_code(ds_key)])
        ds_feats = t.cat([ds_universal_features, ds_specific_features])
        del sdxl_quant_dict[ds_key]
        sdxl.add_node(ds_key, all=ds_feats)
        downsample_keys[i] = ds_key

    upsample_keys = {}
    for i in [0, 1]:
        us_key = f"model.up_blocks.{i}.upsamplers.0.conv.weight_quantizer"
        us_universal_features = convert_universal_layer_feats_to_vector(sdxl_quant_dict[us_key])
        us_specific_features = t.Tensor([_get_stage_code(us_key), _get_block_code(us_key), -1, _get_layer_code(us_key)])
        us_feats = t.cat([us_universal_features, us_specific_features])
        del sdxl_quant_dict[us_key]
        sdxl.add_node(us_key, all=us_feats)
        upsample_keys[i] = us_key

    out2_key = 'model.conv_out.weight_quantizer'
    out2_universal_features = convert_universal_layer_feats_to_vector(sdxl_quant_dict[out2_key])
    out2_specific_features = t.Tensor([_get_stage_code(out2_key), _get_block_code(out2_key), -1, _get_layer_code(out2_key)])
    out2_features = t.cat([out2_universal_features, out2_specific_features])
    del sdxl_quant_dict[out2_key]
    sdxl.add_node(out2_key, all=out2_features)

    assert len(sdxl_quant_dict.keys()) == 0, sdxl_quant_dict.keys()

    def mult_i_j(g, i_list, j):
        for i in i_list:
            g.add_edge(i, j)

    for i1 in range(len(SDXL_IN_RESBLK_IDX)):
        for i2 in range(SDXL_IN_RESBLK_IDX[i1]):
            key = f"model.down_blocks.{i1}.resnets.{i2}"
            resnet_dict = input_rn_dict[key]
            sdxl = nx.union(sdxl, resnet_dict['nx'])
            mult_i_j(sdxl, prev_output_key, resnet_dict['in_dict']['input'])
            sdxl.add_edge(time_add_key, resnet_dict['in_dict']['time_emb_proj'])
            if 'conv_shortcut' in resnet_dict['in_dict'].keys():
                mult_i_j(sdxl, prev_output_key, resnet_dict['in_dict']['conv_shortcut'])
            prev_output_key = [resnet_dict['out_dict']['out_layers']]
            del input_rn_dict[key]

            transformer_key = key.replace(f"resnets.{i2}", f"attentions.{i2}")
            if transformer_key in input_tf_dict.keys():
                transformer_dict = input_tf_dict[transformer_key]
                sdxl = nx.union(sdxl, transformer_dict['nx'])
                mult_i_j(sdxl, prev_output_key, transformer_dict['in_dict']['proj_in'])
                prev_output_key = [transformer_dict['out_dict']['proj_out']]
                del input_tf_dict[transformer_key]
            
        if i1 in [0, 1]:
            mult_i_j(sdxl, prev_output_key, downsample_keys[i1])
            prev_output_key = [downsample_keys[i1]]
            del downsample_keys[i1]

    assert len(downsample_keys.keys()) == 0, downsample_keys.keys()
    assert len(input_rn_dict.keys()) == 0, input_rn_dict.keys()
    assert len(input_tf_dict.keys()) == 0, input_tf_dict.keys()

    # Mid block - do manually
    mid_dict = mid_rn_dict['model.mid_block.resnets.0']
    sdxl = nx.union(sdxl, mid_dict['nx'])
    mult_i_j(sdxl, prev_output_key, mid_dict['in_dict']['input'])
    sdxl.add_edge(time_add_key, mid_dict['in_dict']['time_emb_proj'])
    prev_output_key = [mid_dict['out_dict']['out_layers']]

    mid_dict = mid_tf_dict['model.mid_block.attentions.0']
    sdxl = nx.union(sdxl, mid_dict['nx'])
    mult_i_j(sdxl, prev_output_key, mid_dict['in_dict']['proj_in'])
    prev_output_key = [mid_dict['out_dict']['proj_out']]

    mid_dict = mid_rn_dict['model.mid_block.resnets.1']
    sdxl = nx.union(sdxl, mid_dict['nx'])
    mult_i_j(sdxl, prev_output_key, mid_dict['in_dict']['input'])
    sdxl.add_edge(time_add_key, mid_dict['in_dict']['time_emb_proj'])
    prev_output_key = [mid_dict['out_dict']['out_layers']]

    for i1 in range(len(SDXL_OUT_RESBLK_IDX)):
        for i2 in range(SDXL_OUT_RESBLK_IDX[i1]):
            key = f"model.up_blocks.{i1}.resnets.{i2}"
            resnet_dict = output_rn_dict[key]
            sdxl = nx.union(sdxl, resnet_dict['nx'])
            mult_i_j(sdxl, prev_output_key, resnet_dict['in_dict']['input'])
            sdxl.add_edge(time_add_key, resnet_dict['in_dict']['time_emb_proj'])
        
            # NOTE This deals with long skip-connect
            i2_index = max(1-i2, 0)
            in_blk_output_skip_key = f'model.down_blocks.{2-i1}.resnets.{i2_index}.conv_shortcut.weight_quantizer'
            if in_blk_output_skip_key in sdxl:
                sdxl.add_edge(in_blk_output_skip_key, resnet_dict['in_dict']['conv_shortcut0'])
            else:
                sdxl.add_edge(f'model.down_blocks.{2-i1}.resnets.{i2_index}.conv2.weight_quantizer', resnet_dict['in_dict']['conv_shortcut0'])

            mult_i_j(sdxl, prev_output_key, resnet_dict['in_dict']['input'])
            prev_output_key = [resnet_dict['out_dict']['out_layers']]
            del output_rn_dict[key]

            transformer_key = key.replace(f"resnets.{i2}", f"attentions.{i2}")
            if transformer_key in output_tf_dict.keys():
                transformer_dict = output_tf_dict[transformer_key]
                sdxl = nx.union(sdxl, transformer_dict['nx'])
                mult_i_j(sdxl, prev_output_key, transformer_dict['in_dict']['proj_in'])
                prev_output_key = [transformer_dict['out_dict']['proj_out']]
                del output_tf_dict[transformer_key]

        if i1 in [0, 1]:
            mult_i_j(sdxl, prev_output_key, upsample_keys[i1])
            prev_output_key = [upsample_keys[i1]]
            del upsample_keys[i1]

    assert len(upsample_keys.keys()) == 0, upsample_keys.keys()
    assert len(output_rn_dict.keys()) == 0, output_rn_dict.keys()
    assert len(output_tf_dict.keys()) == 0, output_tf_dict.keys()

    mult_i_j(sdxl, prev_output_key, "model.conv_out.weight_quantizer")
    prev_output_key = ["model.conv_out.weight_quantizer"]

    #visualize_graph(sdxl, "sdxl")
    tg_sample = tg.utils.convert.from_networkx(sdxl, group_node_attrs = ["all"])
    tg_sample.y = y
    return tg_sample


if __name__ == "__main__":
    import pickle
    with open("cache/sdxl_sdm_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    print(construct_sdxl_tg_graph(cache[0]['config']))
