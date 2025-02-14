import torch as t
import torch_geometric as tg
import networkx as nx
from sdm.graph_util import convert_universal_layer_feats_to_vector, gen_combine_node, visualize_graph
from collections import OrderedDict

SDV15_IN_TRANSFORMER_BLK_IDX = [1, 2, 4, 5, 7, 8]
SDV15_MID_TRANSFORMER_BLK_IDX = [1]
SDV15_OUT_TRANSFORMER_BLK_IDX = [3, 4, 5, 6, 7, 8, 9, 10, 11]

SDV15_IN_RESBLK_IDX = [1, 2, 4, 5, 7, 8, 10, 11]
SDV15_MID_RESBLK_IDX = [0, 2]
SDV15_OUT_RESBLK_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

SDV15_LAYERS_LUT = OrderedDict()
SDV15_LAYERS_LUT["add"] = 0             
SDV15_LAYERS_LUT["matmul"] = 0
SDV15_LAYERS_LUT["time_embed.0"] = 1
SDV15_LAYERS_LUT["time_embed.2"] = 2
SDV15_LAYERS_LUT["0.0"] = 3
SDV15_LAYERS_LUT["in_layers.2"] = 4
SDV15_LAYERS_LUT["emb_layers.1"] = 5
SDV15_LAYERS_LUT["out_layers.3"] = 6
SDV15_LAYERS_LUT["0.skip_connection"] = 7
SDV15_LAYERS_LUT["1.proj_in"] = 8
SDV15_LAYERS_LUT["1.proj_out"] = 9
SDV15_LAYERS_LUT["attn1.to_q"] = 10
SDV15_LAYERS_LUT["attn1.to_k"] = 11
SDV15_LAYERS_LUT["attn1.to_v"] = 12
SDV15_LAYERS_LUT["attn1.to_q"] = 13
SDV15_LAYERS_LUT["attn1.to_out.0"] = 14
SDV15_LAYERS_LUT["attn2.to_q"] = 15
SDV15_LAYERS_LUT["attn2.to_k"] = 16
SDV15_LAYERS_LUT["attn2.to_v"] = 17
SDV15_LAYERS_LUT["attn2.to_q"] = 18
SDV15_LAYERS_LUT["attn2.to_out.0"] = 19
SDV15_LAYERS_LUT["ff.net.0.proj"] = 20
SDV15_LAYERS_LUT["ff.net.2"] = 21
SDV15_LAYERS_LUT["out.2"] = 22
SDV15_LAYERS_LUT["op"] = 23
SDV15_LAYERS_LUT["conv"] = 24


def convert_sdv15_layer_to_vector(dit_k, dit_v):
    if type(dit_v) == t.Tensor:
        return dit_v
    
    stage_code = _get_stage_code(dit_k)
    blk_code = _get_block_code(dit_k)
    layer_code = _get_layer_code(dit_k)
    universal_feats = convert_universal_layer_feats_to_vector(dit_v)
    sdv15_specific_feats = t.Tensor([stage_code, blk_code, layer_code])
    return t.cat([universal_feats, sdv15_specific_feats])


def _get_layer_code(layer_key):
    layer_key = layer_key.replace(".weight_quantizer_0", "").replace(".weight_quantizer", "")
    for key in SDV15_LAYERS_LUT.keys():
        if layer_key.endswith(key):
            return SDV15_LAYERS_LUT[key]
        
def _get_stage_code(key):
    # input_blocks.0.0 is initial conv not part of any block
    if "input_blocks" in key or "time_embed" in key or "model.input_blocks.0.0" in key:
        return 0
    elif "middle_block" in key:
        return 1
    elif "output_blocks" in key or "model.out." in key:
        return 2
    else:
        raise NotImplementedError
    

def _get_block_code(key):
    str_code = key.split(".")[2]
    try:
        return int(str_code)
    except:
        return -1


def _gen_attn_blk(sdv15_quant_dict, prefix, ca=False):

    attn = nx.DiGraph()

    q_key = ".".join([prefix, "to_q", "weight_quantizer"])
    stage_code = _get_stage_code(q_key)
    block_code = _get_block_code(q_key)
    q_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[q_key])
    q_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(q_key)])
    q_feats = t.cat([q_universal_feats, q_specific_feats])
    del sdv15_quant_dict[q_key]
    attn.add_node(q_key, all=q_feats)

    k_key = ".".join([prefix, "to_k", "weight_quantizer"])
    k_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[k_key])
    k_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(k_key)])
    k_feats = t.cat([k_universal_feats, k_specific_feats])
    del sdv15_quant_dict[k_key]
    attn.add_node(k_key, all=k_feats)

    v_key = ".".join([prefix, "to_v", "weight_quantizer"])
    v_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[v_key])
    v_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(v_key)])
    v_feats = t.cat([v_universal_feats, v_specific_feats])
    del sdv15_quant_dict[v_key]
    attn.add_node(v_key, all=v_feats)

    out_key = ".".join([prefix, "to_out.0", "weight_quantizer"])
    out_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[out_key])
    out_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(out_key)])
    out_feats = t.cat([out_universal_feats, out_specific_feats])
    del sdv15_quant_dict[out_key]
    attn.add_node(out_key, all=out_feats)

    qk_key = prefix + ".mul_qk"
    qk_universal_feats = gen_combine_node()
    qk_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code("matmul")])
    qk_feats = t.cat([qk_universal_feats, qk_specific_feats])
    attn.add_node(qk_key, all=qk_feats)

    vsm_key = prefix + ".mul_vsm"
    vsm_universal_feats = gen_combine_node()
    vsm_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code("matmul")])
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


def _gen_ff_blk(sdv15_quant_dict, prefix):
    ff = nx.DiGraph()
    ff1_key = ".".join([prefix, "ff.net.0.proj", "weight_quantizer"])
    ff1_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[ff1_key])
    stage_code = _get_stage_code(ff1_key)
    block_code = _get_block_code(ff1_key)
    ff1_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(ff1_key)])
    ff1_feats = t.cat([ff1_universal_feats, ff1_specific_feats])
    del sdv15_quant_dict[ff1_key]
    ff.add_node(ff1_key, all=ff1_feats)

    ff2_key = ".".join([prefix, "ff.net.2", "weight_quantizer"])
    ff2_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[ff2_key])
    ff2_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(ff2_key)])
    ff2_feats = t.cat([ff2_universal_feats, ff2_specific_feats])
    del sdv15_quant_dict[ff2_key]
    ff.add_node(ff2_key, all=ff2_feats)

    ff.add_edge(ff1_key, ff2_key)
    in_dict = {'ff1': ff1_key}
    out_dict = {'ff2': ff2_key}
    return {'nx': ff, 'in_dict': in_dict, 'out_dict': out_dict}


# Prefix is like "model.input_blocks.1.1.transformer_blocks"
def _gen_sdv15_transformer_block(sdv15_quant_dict, prefix):
    transformer = nx.DiGraph()

    proj_in_key = ".".join([prefix.replace(".transformer_blocks.0", ""), "proj_in", "weight_quantizer"])
    stage_code = _get_stage_code(proj_in_key)
    block_code = _get_block_code(proj_in_key)
    proj_in_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[proj_in_key])
    proj_in_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(proj_in_key)])
    proj_in_feats = t.cat([proj_in_universal_feats, proj_in_specific_feats])
    del sdv15_quant_dict[proj_in_key]
    transformer.add_node(proj_in_key, all=proj_in_feats)

    proj_out_key = ".".join([prefix.replace(".transformer_blocks.0", ""), "proj_out", "weight_quantizer"])
    proj_out_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[proj_out_key])
    proj_out_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(proj_out_key)])
    proj_out_feats = t.cat([proj_out_universal_feats, proj_out_specific_feats])
    del sdv15_quant_dict[proj_out_key]
    transformer.add_node(proj_out_key, all=proj_out_feats)

    sa_dict = _gen_attn_blk(sdv15_quant_dict, prefix=".".join([prefix, "attn1"]), ca=False)
    transformer = nx.union(transformer, sa_dict['nx'])
    for j in sa_dict['in_dict'].values():
        transformer.add_edge(proj_in_key, j)

    sa_add_key = prefix + ".add_sa"
    sa_add_universal_features = gen_combine_node()
    sa_add_specific_features = t.Tensor([stage_code, block_code, _get_layer_code('add')])
    sa_add_features = t.cat([sa_add_universal_features, sa_add_specific_features])
    transformer.add_node(sa_add_key, all=sa_add_features)
    transformer.add_edge(proj_in_key, sa_add_key)
    transformer.add_edge(sa_dict['out_dict']['out'], sa_add_key)

    ca_dict = _gen_attn_blk(sdv15_quant_dict, prefix=".".join([prefix, "attn2"]), ca=True)
    transformer = nx.union(transformer, ca_dict['nx'])
    for j in ca_dict['in_dict'].values():
        transformer.add_edge(sa_add_key, j)

    ca_add_key = prefix + ".add_ca"
    ca_add_universal_features = gen_combine_node()
    ca_add_specific_features = t.Tensor([stage_code, block_code, _get_layer_code('add')])
    ca_add_features = t.cat([ca_add_universal_features, ca_add_specific_features])
    transformer.add_node(ca_add_key, all=ca_add_features)
    transformer.add_edge(sa_add_key, ca_add_key)
    transformer.add_edge(ca_dict['out_dict']['out'], ca_add_key)
    
    ff_dict = _gen_ff_blk(sdv15_quant_dict, prefix=prefix)
    transformer = nx.union(transformer, ff_dict['nx'])
    for j in ff_dict['in_dict'].values():
        transformer.add_edge(ca_add_key, j)

    ff_add_key = prefix + ".add_ff"
    ff_add_universal_features = gen_combine_node()
    ff_add_specific_features = t.Tensor([stage_code, block_code, _get_layer_code('add')])
    ff_add_features = t.cat([ff_add_universal_features, ff_add_specific_features])
    transformer.add_node(ff_add_key, all=ff_add_features)
    transformer.add_edge(ca_add_key, ff_add_key)
    transformer.add_edge(ff_dict['out_dict']['ff2'], ff_add_key)

    transformer.add_edge(ff_add_key, proj_out_key)

    return {'nx': transformer, 'in_dict': {'proj_in': proj_in_key}, 'out_dict': {'proj_out': proj_out_key}}


def _gen_all_sdv15_transformer_blocks(sdv15_quant_dict):

    input_tf_dict, mid_tf_dict, output_tf_dict = OrderedDict(), OrderedDict(), OrderedDict()
    for i in SDV15_IN_TRANSFORMER_BLK_IDX:
        input_prefix = f"model.input_blocks.{i}.1.transformer_blocks.0"
        input_tf_dict[input_prefix] = _gen_sdv15_transformer_block(sdv15_quant_dict, input_prefix)

    for i in SDV15_MID_TRANSFORMER_BLK_IDX:
        mid_prefix = f"model.middle_block.{i}.transformer_blocks.0"
        mid_tf_dict[mid_prefix] = _gen_sdv15_transformer_block(sdv15_quant_dict, mid_prefix)

    for i in SDV15_OUT_TRANSFORMER_BLK_IDX:
        output_prefix = f"model.output_blocks.{i}.1.transformer_blocks.0"
        output_tf_dict[output_prefix] = _gen_sdv15_transformer_block(sdv15_quant_dict, output_prefix)

    return input_tf_dict, mid_tf_dict, output_tf_dict


def _gen_sdv15_resblock(sdv15_quant_dict, prefix):

    resblk = nx.DiGraph()

    in_layer_key = ".".join([prefix, "in_layers.2", "weight_quantizer"])
    stage_code = _get_stage_code(in_layer_key)
    block_code = _get_block_code(in_layer_key)
    in_layer_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[in_layer_key])
    in_layer_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(in_layer_key)])
    resblk.add_node(in_layer_key, all=t.cat([in_layer_universal_feats, in_layer_specific_feats]))
    del sdv15_quant_dict[in_layer_key]

    emb_layer_key = ".".join([prefix, "emb_layers.1", "weight_quantizer"])
    emb_layer_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[emb_layer_key])
    emb_layer_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(emb_layer_key)])
    resblk.add_node(emb_layer_key, all=t.cat([emb_layer_universal_feats, emb_layer_specific_feats]))
    del sdv15_quant_dict[emb_layer_key]

    out_layer_key = ".".join([prefix, "out_layers.3", "weight_quantizer"])
    out_layer_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[out_layer_key])
    out_layer_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(out_layer_key)])
    resblk.add_node(out_layer_key, all=t.cat([out_layer_universal_feats, out_layer_specific_feats]))
    del sdv15_quant_dict[out_layer_key]

    skip_connect_key = ".".join([prefix, "skip_connection", "weight_quantizer"])
    skips, skips0 = False, False
    if skip_connect_key in sdv15_quant_dict.keys():
        skips = True
        skip_connect_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[skip_connect_key])
        skip_connect_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(skip_connect_key)])
        resblk.add_node(skip_connect_key, all=t.cat([skip_connect_universal_feats, skip_connect_specific_feats]))
        del sdv15_quant_dict[skip_connect_key]

        skip_connect_key0 = skip_connect_key + "_0"
        if skip_connect_key0 in sdv15_quant_dict.keys():
            skips0 = True
            skip_connect0_universal_feats = convert_universal_layer_feats_to_vector(sdv15_quant_dict[skip_connect_key0])
            skip_connect0_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code(skip_connect_key0)])
            resblk.add_node(skip_connect_key0, all=t.cat([skip_connect0_universal_feats, skip_connect0_specific_feats]))
            del sdv15_quant_dict[skip_connect_key0]

    resblk.add_edge(in_layer_key, out_layer_key)
    resblk.add_edge(emb_layer_key, out_layer_key)

    # 0 is always an input, as is 3 if it exists
    in_dict = {'in_conv': in_layer_key, 'emb_layer': emb_layer_key}
    
    if skips or skips0:
        skip_h_key = prefix + ".add"
        skip_h_universal_feats = gen_combine_node()
        skip_h_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code("add")])
        skip_h_feats = t.cat([skip_h_universal_feats, skip_h_specific_feats])
        resblk.add_node(skip_h_key, all=skip_h_feats)
        resblk.add_edge(out_layer_key, skip_h_key)
        out_dict = {'out_layers': skip_h_key}
    else:
        out_dict = {'out_layers': out_layer_key}

    if skips0:
        in_dict['skip_connection'] = skip_connect_key
        in_dict['skip_connection0'] = skip_connect_key0
        skip_add_key = prefix + ".add_qdiff"
        skip_add_universal_feats = gen_combine_node()
        skip_add_specific_feats = t.Tensor([stage_code, block_code, _get_layer_code("add")])
        skip_add_feats = t.cat([skip_add_universal_feats, skip_add_specific_feats])
        resblk.add_node(skip_add_key, all=skip_add_feats)
        resblk.add_edge(skip_connect_key, skip_add_key)
        resblk.add_edge(skip_connect_key0, skip_add_key)
        resblk.add_edge(skip_add_key, out_dict['out_layers'])
    elif skips:
        in_dict['skip_connection'] = skip_connect_key
        resblk.add_edge(skip_connect_key, out_dict['out_layers'])
    
    return {'nx': resblk, 'in_dict': in_dict, 'out_dict': out_dict}


def _gen_all_sdv15_resblocks(sdv15_quant_dict):
    input_dict, mid_dict, output_dict = OrderedDict(), OrderedDict(), OrderedDict()
    for i in SDV15_IN_RESBLK_IDX:
        input_prefix = f"model.input_blocks.{i}.0"
        input_dict[input_prefix] = _gen_sdv15_resblock(sdv15_quant_dict, input_prefix)
    
    for i in SDV15_MID_RESBLK_IDX:
        mid_prefix = f"model.middle_block.{i}"
        mid_dict[mid_prefix] = _gen_sdv15_resblock(sdv15_quant_dict, mid_prefix)

    for i in SDV15_OUT_RESBLK_IDX:
        output_prefix = f"model.output_blocks.{i}.0"
        output_dict[output_prefix] = _gen_sdv15_resblock(sdv15_quant_dict, output_prefix)

    return input_dict, mid_dict, output_dict


def construct_sdv15_tg_graph(sdv15_quant_dict, y=None):

    input_tf_dict, mid_tf_dict, output_tf_dict = _gen_all_sdv15_transformer_blocks(sdv15_quant_dict)
    input_rn_dict, mid_rn_dict, output_rn_dict = _gen_all_sdv15_resblocks(sdv15_quant_dict)

    # NOTE remember how to interact with skip-connects for input/output
    # E.g., its input[i] connects to output[11-i]
    sdv15 = nx.DiGraph()
    te0_key = "model.time_embed.0.weight_quantizer"
    te0_universal_features = convert_universal_layer_feats_to_vector(sdv15_quant_dict[te0_key])
    te0_specific_features = t.Tensor([_get_stage_code(te0_key), _get_block_code(te0_key), _get_layer_code(te0_key)])
    te0_features = t.cat([te0_universal_features, te0_specific_features])
    del sdv15_quant_dict[te0_key]
    sdv15.add_node(te0_key, all=te0_features)

    te2_key = "model.time_embed.2.weight_quantizer"
    te2_universal_features = convert_universal_layer_feats_to_vector(sdv15_quant_dict[te2_key])
    te2_specific_features = t.Tensor([_get_stage_code(te2_key), _get_block_code(te2_key), _get_layer_code(te2_key)])
    te2_features = t.cat([te2_universal_features, te2_specific_features])
    del sdv15_quant_dict[te2_key]
    sdv15.add_node(te2_key, all=te2_features)
    sdv15.add_edge(te0_key, te2_key)

    downsample_keys = {}
    for i in [0, 3, 6, 9]:
        if i == 0:
            ds_key = f"model.input_blocks.{i}.0.weight_quantizer"
        else:
            ds_key = f"model.input_blocks.{i}.0.op.weight_quantizer"
        ds_universal_features = convert_universal_layer_feats_to_vector(sdv15_quant_dict[ds_key])
        ds_specific_features = t.Tensor([_get_stage_code(ds_key), _get_block_code(ds_key), _get_layer_code(ds_key)])
        ds_feats = t.cat([ds_universal_features, ds_specific_features])
        del sdv15_quant_dict[ds_key]
        sdv15.add_node(ds_key, all=ds_feats)
        downsample_keys[i] = ds_key

    upsample_keys = {}
    for i in [2, 5, 8]:
        if i == 2:
            us_key = f"model.output_blocks.{i}.1.conv.weight_quantizer"
        else:
            us_key = f"model.output_blocks.{i}.2.conv.weight_quantizer"
        us_universal_features = convert_universal_layer_feats_to_vector(sdv15_quant_dict[us_key])
        us_specific_features = t.Tensor([_get_stage_code(us_key), _get_block_code(us_key), _get_layer_code(us_key)])
        us_feats = t.cat([us_universal_features, us_specific_features])
        del sdv15_quant_dict[us_key]
        sdv15.add_node(us_key, all=us_feats)
        upsample_keys[i] = us_key

    out2_key = 'model.out.2.weight_quantizer'
    out2_universal_features = convert_universal_layer_feats_to_vector(sdv15_quant_dict[out2_key])
    out2_specific_features = t.Tensor([_get_stage_code(out2_key), _get_block_code(out2_key), _get_layer_code(out2_key)])
    out2_features = t.cat([out2_universal_features, out2_specific_features])
    del sdv15_quant_dict[out2_key]
    sdv15.add_node(out2_key, all=out2_features)

    assert len(sdv15_quant_dict.keys()) == 0, sdv15_quant_dict.keys()

    def mult_i_j(g, i_list, j):
        for i in i_list:
            g.add_edge(i, j)

    prev_output_key = ["model.input_blocks.0.0.weight_quantizer"]
    del downsample_keys[0]
    for i in range(1, 12):
        if i in [3, 6, 9]:
            mult_i_j(sdv15, prev_output_key, downsample_keys[i])
            prev_output_key = [downsample_keys[i]]
            del downsample_keys[i]
        else:
            key = f"model.input_blocks.{i}.0"
            resnet_dict = input_rn_dict[key]
            sdv15 = nx.union(sdv15, resnet_dict['nx'])
            mult_i_j(sdv15, prev_output_key, resnet_dict['in_dict']['in_conv'])
            sdv15.add_edge(te2_key, resnet_dict['in_dict']['emb_layer'])
            if 'skip_connection' in resnet_dict['in_dict'].keys():
                mult_i_j(sdv15, prev_output_key, resnet_dict['in_dict']['skip_connection'])
            prev_output_key = [resnet_dict['out_dict']['out_layers']]
            del input_rn_dict[key]

            transformer_key = key.replace(".0", ".1.transformer_blocks.0")
            if transformer_key in input_tf_dict.keys():
                transformer_dict = input_tf_dict[transformer_key]
                sdv15 = nx.union(sdv15, transformer_dict['nx'])
                mult_i_j(sdv15, prev_output_key, transformer_dict['in_dict']['proj_in'])
                prev_output_key = [transformer_dict['out_dict']['proj_out']]
                del input_tf_dict[transformer_key]

    assert len(downsample_keys.keys()) == 0, downsample_keys.keys()
    assert len(input_rn_dict.keys()) == 0, input_rn_dict.keys()
    assert len(input_tf_dict.keys()) == 0, input_tf_dict.keys()

    # Mid block - do manually
    mid_dict = mid_rn_dict['model.middle_block.0']
    sdv15 = nx.union(sdv15, mid_dict['nx'])
    mult_i_j(sdv15, prev_output_key, mid_dict['in_dict']['in_conv'])
    sdv15.add_edge(te2_key, mid_dict['in_dict']['emb_layer'])
    prev_output_key = [mid_dict['out_dict']['out_layers']]

    mid_dict = mid_tf_dict['model.middle_block.1.transformer_blocks.0']
    sdv15 = nx.union(sdv15, mid_dict['nx'])
    mult_i_j(sdv15, prev_output_key, mid_dict['in_dict']['proj_in'])
    prev_output_key = [mid_dict['out_dict']['proj_out']]

    mid_dict = mid_rn_dict['model.middle_block.2']
    sdv15 = nx.union(sdv15, mid_dict['nx'])
    mult_i_j(sdv15, prev_output_key, mid_dict['in_dict']['in_conv'])
    sdv15.add_edge(te2_key, mid_dict['in_dict']['emb_layer'])
    prev_output_key = [mid_dict['out_dict']['out_layers']]

    # E.g., its input[i] connects to output[11-i]
    for i in range(12):
        key = f"model.output_blocks.{i}.0"
        resnet_dict = output_rn_dict[key]
        sdv15 = nx.union(sdv15, resnet_dict['nx'])
        mult_i_j(sdv15, prev_output_key, resnet_dict['in_dict']['in_conv'])
        sdv15.add_edge(te2_key, resnet_dict['in_dict']['emb_layer'])
    
        # NOTE This deals with long skip-connect
        if 11-i in [3, 6, 9]:
            sdv15.add_edge(f'model.input_blocks.{11-i}.0.op.weight_quantizer', resnet_dict['in_dict']['skip_connection0'])
        elif 11-i == 0:
            sdv15.add_edge(f'model.input_blocks.{11-i}.0.weight_quantizer',
                           resnet_dict['in_dict']['skip_connection0'])
        else:
            in_blk_output_skip_key = f'model.input_blocks.{11-i}.0.skip_connection.weight_quantizer'
            if in_blk_output_skip_key in sdv15:
                sdv15.add_edge(in_blk_output_skip_key, resnet_dict['in_dict']['skip_connection0'])
            else:
                sdv15.add_edge(f'model.input_blocks.{11-i}.0.out_layers.3.weight_quantizer', resnet_dict['in_dict']['skip_connection0'])
        if 'skip_connection' in resnet_dict['in_dict'].keys():
            mult_i_j(sdv15, prev_output_key, resnet_dict['in_dict']['skip_connection'])
        else:
            mult_i_j(sdv15, prev_output_key, resnet_dict['in_dict']['out_layers'])
        prev_output_key = [resnet_dict['out_dict']['out_layers']]
        del output_rn_dict[key]

        transformer_key = key.replace(".0", ".1.transformer_blocks.0")
        if transformer_key in output_tf_dict.keys():
            transformer_dict = output_tf_dict[transformer_key]
            sdv15 = nx.union(sdv15, transformer_dict['nx'])
            mult_i_j(sdv15, prev_output_key, transformer_dict['in_dict']['proj_in'])
            prev_output_key = [transformer_dict['out_dict']['proj_out']]
            del output_tf_dict[transformer_key]

        if i in [2, 5, 8]:
            mult_i_j(sdv15, prev_output_key, upsample_keys[i])
            prev_output_key = [upsample_keys[i]]
            del upsample_keys[i]

    assert len(upsample_keys.keys()) == 0, upsample_keys.keys()
    assert len(output_rn_dict.keys()) == 0, output_rn_dict.keys()
    assert len(output_tf_dict.keys()) == 0, output_tf_dict.keys()

    mult_i_j(sdv15, prev_output_key, "model.out.2.weight_quantizer")
    prev_output_key = ["model.out.2.weight_quantizer"]

    #visualize_graph(sdv15, "sdv15")
    tg_sample = tg.utils.convert.from_networkx(sdv15, group_node_attrs = ["all"])
    tg_sample.y = y
    return tg_sample


if __name__ == "__main__":
    import pickle
    with open("sdv15_multiquant_s3_eq/2024-04-20-11-31-03/quant_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    print(construct_sdv15_tg_graph(cache[0]['config']))
