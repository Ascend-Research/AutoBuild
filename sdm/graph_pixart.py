import torch as t
import networkx as nx
import torch_geometric as tg 
from collections import OrderedDict
from sdm.graph_util import convert_universal_layer_feats_to_vector, gen_combine_node, visualize_graph

PIXART_TSX_COMBINE_LUT = OrderedDict()
PIXART_TSX_COMBINE_LUT["add"] = 0
PIXART_TSX_COMBINE_LUT["matmul"] = 0
PIXART_TSX_COMBINE_LUT["adaln"] = 0
PIXART_TSX_COMBINE_LUT["norm"] = 0
PIXART_TSX_LAYERS_LUT = OrderedDict()
PIXART_TSX_LAYERS_LUT["pos_embed.proj"] = 1
PIXART_TSX_LAYERS_LUT["adaln_single.emb.timestep_embedder.linear_1"] = 2
PIXART_TSX_LAYERS_LUT["adaln_single.emb.timestep_embedder.linear_2"] = 3
PIXART_TSX_LAYERS_LUT["adaln_single.linear"] = 4
PIXART_TSX_LAYERS_LUT["caption_projection.linear_1"] = 5
PIXART_TSX_LAYERS_LUT["caption_projection.linear_2"] = 6
PIXART_TSX_LAYERS_LUT["attn1.to_q"] = 7
PIXART_TSX_LAYERS_LUT["attn1.to_k"] = 8
PIXART_TSX_LAYERS_LUT["attn1.to_v"] = 9
PIXART_TSX_LAYERS_LUT["attn1.to_out.0"] = 10
PIXART_TSX_LAYERS_LUT["attn2.to_q"] = 11
PIXART_TSX_LAYERS_LUT["attn2.to_k"] = 12
PIXART_TSX_LAYERS_LUT["attn2.to_v"] = 13
PIXART_TSX_LAYERS_LUT["attn2.to_out.0"] = 14
PIXART_TSX_LAYERS_LUT["ff.net.0.proj"] = 15
PIXART_TSX_LAYERS_LUT["ff.net.2"] = 16
PIXART_TSX_LAYERS_LUT["proj_out"] = 17

PIXART_POS_EMBED_LAYER_0_KEY = 'model.pos_embed.proj.weight_quantizer'
PIXART_ADALN_EMB_1_KEY = "model.adaln_single.emb.timestep_embedder.linear_1.weight_quantizer"
PIXART_ADALN_EMB_2_KEY = "model.adaln_single.emb.timestep_embedder.linear_2.weight_quantizer"
PIXART_ADALN_LINEAR_KEY = "model.adaln_single.linear.weight_quantizer"
PIXART_CAPTION_LINEAR_1_KEY = "model.caption_projection.linear_1.weight_quantizer"
PIXART_CAPTION_LINEAR_2_KEY = "model.caption_projection.linear_2.weight_quantizer"


def convert_dit_layer_to_vector(dit_k, dit_v):
    if type(dit_v) == t.Tensor:
        return dit_v
    elif "transformer_blocks" in dit_k and "timestep_embedder" not in dit_k:
        return convert_dit_tsx_layer_to_vector(dit_k, dit_v)
    elif "pos_embed" in dit_k:
        blk_code = 0
        layer_code = PIXART_TSX_LAYERS_LUT["pos_embed.proj"]
    else:
        blk_code = -1
        dit_k_trimmed = dit_k.replace(".weight_quantizer", "")
        if dit_k_trimmed.endswith("proj_out_1"):
            layer_code = PIXART_TSX_COMBINE_LUT["proj_out_1"]
        elif dit_k_trimmed.endswith("proj_out_2"):
            layer_code = PIXART_TSX_COMBINE_LUT["proj_out_2"]
        elif dit_k_trimmed.endswith("timestep_embedder.linear_1"):
            layer_code = PIXART_TSX_COMBINE_LUT["transformer_blocks.0.norm1.emb.timestep_embedder.linear_1"]
        elif dit_k_trimmed.endswith("timestep_embedder.linear_2"):
            layer_code = PIXART_TSX_COMBINE_LUT["transformer_blocks.0.norm1.emb.timestep_embedder.linear_2"]
        else:
            raise NotImplementedError
    universal_feats = convert_universal_layer_feats_to_vector(dit_v)
    dit_specific_feats = t.Tensor([blk_code, layer_code])
    return t.cat([universal_feats, dit_specific_feats])


# dit_k is the name, like ''model.transformer_blocks.27.norm1.linear.weight_quantizer'
# dit_v is the value list for quantization info, e.g., ['kmeans-4', 589824, 184320, 0.00047016143798828125]
# NOTE We do not consider the timestep_embedding_layers. 
def convert_dit_tsx_layer_to_vector(dit_k, dit_v):
    assert dit_k.startswith("model.transformer_blocks.") and "timestep_embedder" not in dit_k
    dit_k = dit_k.replace("model.transformer_blocks.", "").replace(".weight_quantizer", "")
    dit_k_list = dit_k.split(".")
    blk_code = int(dit_k_list[0])
    layer_type = ".".join(dit_k_list[1:])
    layer_code = PIXART_TSX_LAYERS_LUT[layer_type]
    universal_feats = convert_universal_layer_feats_to_vector(dit_v)
    dit_specific_feats = t.Tensor([blk_code, layer_code])
    return t.cat([universal_feats, dit_specific_feats])


def _generate_pixart_adaln_single_sg(layer_1_feats, layer_2_feats, linear_feats, out_dict):
    time_embed_graph = nx.DiGraph()
    
    layer_1_universal_feats = convert_universal_layer_feats_to_vector(layer_1_feats)
    layer_1_pixart_feats = t.Tensor([-1, PIXART_TSX_LAYERS_LUT["adaln_single.emb.timestep_embedder.linear_1"]])
    layer_1_feats = t.concat([layer_1_universal_feats, layer_1_pixart_feats])
    time_embed_graph.add_node(PIXART_ADALN_EMB_1_KEY, all=layer_1_feats)

    layer_2_universal_feats = convert_universal_layer_feats_to_vector(layer_2_feats)
    layer_2_pixart_feats = t.Tensor([-1, PIXART_TSX_LAYERS_LUT["adaln_single.emb.timestep_embedder.linear_2"]])
    layer_2_feats = t.concat([layer_2_universal_feats, layer_2_pixart_feats])
    time_embed_graph.add_node(PIXART_ADALN_EMB_2_KEY, all=layer_2_feats)
    time_embed_graph.add_edge(PIXART_ADALN_EMB_1_KEY, PIXART_ADALN_EMB_2_KEY)

    linear_universal_feats = convert_universal_layer_feats_to_vector(linear_feats)
    linear_pixart_feats = t.Tensor([-1, PIXART_TSX_LAYERS_LUT["adaln_single.linear"]])
    linear_feats = t.concat([linear_universal_feats, linear_pixart_feats])
    time_embed_graph.add_node(PIXART_ADALN_LINEAR_KEY, all=linear_feats)
    time_embed_graph.add_edge(PIXART_ADALN_EMB_2_KEY, PIXART_ADALN_LINEAR_KEY)

    out_dict['emb'] = PIXART_ADALN_EMB_2_KEY      # This goes to final norm in the network.
    out_dict['linear'] = PIXART_ADALN_LINEAR_KEY   # This goes to all transformer layers

    return time_embed_graph, out_dict


def _generate_pixart_caption_sg(layer_1_feats, layer_2_feats, out_dict):
    caption_embed_graph = nx.DiGraph()
    
    layer_1_universal_feats = convert_universal_layer_feats_to_vector(layer_1_feats)
    layer_1_pixart_feats = t.Tensor([-1, PIXART_TSX_LAYERS_LUT["caption_projection.linear_1"]])
    layer_1_feats = t.concat([layer_1_universal_feats, layer_1_pixart_feats])
    caption_embed_graph.add_node(PIXART_CAPTION_LINEAR_1_KEY, all=layer_1_feats)

    layer_2_universal_feats = convert_universal_layer_feats_to_vector(layer_2_feats)
    layer_2_pixart_feats = t.Tensor([-1, PIXART_TSX_LAYERS_LUT["caption_projection.linear_2"]])
    layer_2_feats = t.concat([layer_2_universal_feats, layer_2_pixart_feats])
    caption_embed_graph.add_node(PIXART_CAPTION_LINEAR_2_KEY, all=layer_2_feats)
    caption_embed_graph.add_edge(PIXART_CAPTION_LINEAR_1_KEY, PIXART_CAPTION_LINEAR_2_KEY)

    out_dict['caption'] = PIXART_CAPTION_LINEAR_2_KEY  # This goes to cross-attn of all TSX layers

    return caption_embed_graph, out_dict


def pixart_blk_sg(transformer, pixart_quant_dict, blk_id, out_dict, prefix="model.transformer_blocks"):

    adaln0_key = f"{prefix}.{str(blk_id)}.adaln_0"
    adaln0_universal_feats = gen_combine_node()
    adaln0_specific_feats = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT["adaln"]])
    adaln0_feats = t.cat([adaln0_universal_feats, adaln0_specific_feats])
    transformer.add_node(adaln0_key, all=adaln0_feats)
    transformer.add_edge(out_dict['x'], adaln0_key)
    transformer.add_edge(out_dict['linear'], adaln0_key)
    
    q1_key = ".".join([prefix, str(blk_id), "attn1", "to_q", "weight_quantizer"])
    q1_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[q1_key])
    q1_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["attn1.to_q"]])
    q1_feats = t.cat([q1_universal_feats, q1_specific_feats])
    del pixart_quant_dict[q1_key]
    transformer.add_node(q1_key, all=q1_feats)

    k1_key = ".".join([prefix, str(blk_id), "attn1", "to_k", "weight_quantizer"])
    k1_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[k1_key])
    k1_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["attn1.to_k"]])
    k1_feats = t.cat([k1_universal_feats, k1_specific_feats])
    del pixart_quant_dict[k1_key]
    transformer.add_node(k1_key, all=k1_feats)

    v1_key = ".".join([prefix, str(blk_id), "attn1", "to_v", "weight_quantizer"])
    v1_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[v1_key])
    v1_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["attn1.to_v"]])
    v1_feats = t.cat([v1_universal_feats, v1_specific_feats])
    del pixart_quant_dict[v1_key]
    transformer.add_node(v1_key, all=v1_feats)

    out1_key = ".".join([prefix, str(blk_id), "attn1", "to_out.0", "weight_quantizer"])
    out1_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[out1_key])
    out1_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["attn1.to_out.0"]])
    out1_feats = t.cat([out1_universal_feats, out1_specific_feats])
    del pixart_quant_dict[out1_key]
    transformer.add_node(out1_key, all=out1_feats)

    transformer.add_edge(adaln0_key, q1_key)
    transformer.add_edge(adaln0_key, k1_key)
    transformer.add_edge(adaln0_key, v1_key)

    qk1_key = f"{prefix}.{str(blk_id)}.mul_qk1"
    qk1_universal_feats = gen_combine_node()
    qk1_specific_feats = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT["matmul"]])
    qk1_feats = t.cat([qk1_universal_feats, qk1_specific_feats])
    transformer.add_node(qk1_key, all=qk1_feats)
    transformer.add_edge(q1_key, qk1_key)
    transformer.add_edge(k1_key, qk1_key)

    vsm1_key = f"{prefix}.{str(blk_id)}.mul_vsm1"
    vsm1_universal_feats = gen_combine_node()
    vsm1_specific_feats = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT["matmul"]])
    vsm1_feats = t.cat([vsm1_universal_feats, vsm1_specific_feats])
    transformer.add_node(vsm1_key, all=vsm1_feats)
    transformer.add_edge(v1_key, vsm1_key)
    transformer.add_edge(qk1_key, vsm1_key)

    transformer.add_edge(vsm1_key, out1_key)

    sa_add_key = f"{prefix}.{str(blk_id)}.add_sa"
    sa_add_universal_features = gen_combine_node()
    sa_add_specific_features = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT["add"]])
    sa_add_features = t.cat([sa_add_universal_features, sa_add_specific_features])
    transformer.add_node(sa_add_key, all=sa_add_features)
    transformer.add_edge(out1_key, sa_add_key)
    transformer.add_edge(out_dict['x'], sa_add_key)

    out_dict['x'] = sa_add_key
    
    # Per Hunyuan/Pixart code, there is no adaln for Cross-Attn, just regular norm
    ca_norm_key = f"{prefix}.{str(blk_id)}.norm"
    ca_norm_universal_feats = gen_combine_node()
    ca_norm_specific_feats = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT['norm']])
    ca_norm_feats = t.cat([ca_norm_universal_feats, ca_norm_specific_feats])
    transformer.add_node(ca_norm_key, all=ca_norm_feats)
    transformer.add_edge(out_dict['x'], ca_norm_key)

    q2_key = ".".join([prefix, str(blk_id), "attn2", "to_q", "weight_quantizer"])
    q2_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[q2_key])
    q2_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["attn2.to_q"]])
    q2_feats = t.cat([q2_universal_feats, q2_specific_feats])
    del pixart_quant_dict[q2_key]
    transformer.add_node(q2_key, all=q2_feats)

    k2_key = ".".join([prefix, str(blk_id), "attn2", "to_k", "weight_quantizer"])
    k2_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[k2_key])
    k2_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["attn2.to_k"]])
    k2_feats = t.cat([k2_universal_feats, k2_specific_feats])
    del pixart_quant_dict[k2_key]
    transformer.add_node(k2_key, all=k2_feats)

    v2_key = ".".join([prefix, str(blk_id), "attn2", "to_v", "weight_quantizer"])
    v2_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[v2_key])
    v2_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["attn2.to_v"]])
    v2_feats = t.cat([v2_universal_feats, v2_specific_feats])
    del pixart_quant_dict[v2_key]
    transformer.add_node(v2_key, all=v2_feats)

    out2_key = ".".join([prefix, str(blk_id), "attn2", "to_out.0", "weight_quantizer"])
    out2_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[out2_key])
    out2_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["attn2.to_out.0"]])
    out2_feats = t.cat([out2_universal_feats, out2_specific_feats])
    del pixart_quant_dict[out2_key]
    transformer.add_node(out2_key, all=out2_feats)

    transformer.add_edge(ca_norm_key, q2_key)
    transformer.add_edge(out_dict['caption'], k2_key)
    transformer.add_edge(out_dict['caption'], v2_key)

    qk2_key = f"{prefix}.{str(blk_id)}.mul_qk2"
    qk2_universal_feats = gen_combine_node()
    qk2_specific_feats = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT["matmul"]])
    qk2_feats = t.cat([qk2_universal_feats, qk2_specific_feats])
    transformer.add_node(qk2_key, all=qk2_feats)
    transformer.add_edge(q2_key, qk2_key)
    transformer.add_edge(k2_key, qk2_key)

    vsm2_key = f"{prefix}.{str(blk_id)}.mul_vsm2"
    vsm2_universal_feats = gen_combine_node()
    vsm2_specific_feats = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT["matmul"]])
    vsm2_feats = t.cat([vsm2_universal_feats, vsm2_specific_feats])
    transformer.add_node(vsm2_key, all=vsm2_feats)
    transformer.add_edge(v2_key, vsm2_key)
    transformer.add_edge(qk2_key, vsm2_key)

    transformer.add_edge(vsm2_key, out2_key)

    ca_add_key = f"{prefix}.{str(blk_id)}.add_ca"
    ca_add_universal_features = gen_combine_node()
    ca_add_specific_features = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT["add"]])
    ca_add_features = t.cat([ca_add_universal_features, ca_add_specific_features])
    transformer.add_node(ca_add_key, all=ca_add_features)
    transformer.add_edge(out2_key, ca_add_key)
    transformer.add_edge(out_dict['x'], ca_add_key)

    out_dict['x'] = ca_add_key

    # AdaLN is used for SA and FF
    adaln1_key = f"{prefix}.{str(blk_id)}.adaln_1"
    adaln1_universal_feats = gen_combine_node()
    adaln1_specific_feats = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT["adaln"]])
    adaln1_feats = t.cat([adaln1_universal_feats, adaln1_specific_feats])
    transformer.add_node(adaln1_key, all=adaln1_feats)
    transformer.add_edge(out_dict['x'], adaln1_key)
    transformer.add_edge(out_dict['linear'], adaln1_key)

    ff1_key = ".".join([prefix, str(blk_id), "ff.net.0.proj", "weight_quantizer"])
    ff1_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[ff1_key])
    ff1_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["ff.net.0.proj"]])
    ff1_feats = t.cat([ff1_universal_feats, ff1_specific_feats])
    del pixart_quant_dict[ff1_key]
    transformer.add_node(ff1_key, all=ff1_feats)

    ff2_key = ".".join([prefix, str(blk_id), "ff.net.2", "weight_quantizer"])
    ff2_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[ff2_key])
    ff2_specific_feats = t.Tensor([blk_id, PIXART_TSX_LAYERS_LUT["ff.net.2"]])
    ff2_feats = t.cat([ff2_universal_feats, ff2_specific_feats])
    del pixart_quant_dict[ff2_key]
    transformer.add_node(ff2_key, all=ff2_feats)

    transformer.add_edge(adaln1_key, ff1_key)
    transformer.add_edge(ff1_key, ff2_key)

    ff_add_key = f"{prefix}.{str(blk_id)}.add_ff"
    ff_add_universal_features = gen_combine_node()
    ff_add_specific_features = t.Tensor([blk_id, PIXART_TSX_COMBINE_LUT["add"]])
    ff_add_features = t.cat([ff_add_universal_features, ff_add_specific_features])
    transformer.add_node(ff_add_key, all=ff_add_features)
    transformer.add_edge(ff2_key, ff_add_key)
    transformer.add_edge(out_dict['x'], ff_add_key)

    out_dict['x'] = ff_add_key

    return transformer, out_dict


def construct_pixart_tg_graph(pixart_quant_dict, y=None):

    pixart = nx.DiGraph()
    # Step 1 - encode vector for positional embedding conv layer
    layer_0_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[PIXART_POS_EMBED_LAYER_0_KEY])
    layer_0_dit_feats = t.Tensor([-1, PIXART_TSX_LAYERS_LUT["pos_embed.proj"]])
    layer_0_feats = t.concat([layer_0_universal_feats, layer_0_dit_feats])
    pixart.add_node(PIXART_POS_EMBED_LAYER_0_KEY, all=layer_0_feats)
    del pixart_quant_dict[PIXART_POS_EMBED_LAYER_0_KEY]

    out_dict = {
        "x": PIXART_POS_EMBED_LAYER_0_KEY
    }

    # Step 2 - Timestep embedding. Encode. Have that create x and edge_index
    time_embed_graph, out_dict = _generate_pixart_adaln_single_sg(
        pixart_quant_dict[PIXART_ADALN_EMB_1_KEY], 
        pixart_quant_dict[PIXART_ADALN_EMB_2_KEY],
        pixart_quant_dict[PIXART_ADALN_LINEAR_KEY],
        out_dict)
    pixart = nx.union(pixart, time_embed_graph)
    del pixart_quant_dict[PIXART_ADALN_EMB_1_KEY]
    del pixart_quant_dict[PIXART_ADALN_EMB_2_KEY]
    del pixart_quant_dict[PIXART_ADALN_LINEAR_KEY]


    # TODO Step 2.5 caption projection
    caption_graph, out_dict = _generate_pixart_caption_sg(
        pixart_quant_dict[PIXART_CAPTION_LINEAR_1_KEY],
        pixart_quant_dict[PIXART_CAPTION_LINEAR_2_KEY],
        out_dict
    )
    del pixart_quant_dict[PIXART_CAPTION_LINEAR_1_KEY]
    del pixart_quant_dict[PIXART_CAPTION_LINEAR_2_KEY]
    
    pixart = nx.union(pixart, caption_graph)

    # Step 3 - For-loop over all Transformer blocks
    for blk_id in range(28):
        pixart, out_dict = pixart_blk_sg(pixart, pixart_quant_dict, blk_id, out_dict)

    # Step 4 - final layers
    norm_out_key = f"model.norm_out"
    norm_out_universal_feats = gen_combine_node()
    norm_out_specific_feats = t.Tensor([-1, PIXART_TSX_COMBINE_LUT['norm']])
    norm_out_feats = t.cat([norm_out_universal_feats, norm_out_specific_feats])
    pixart.add_node(norm_out_key, all=norm_out_feats)
    pixart.add_edge(out_dict['x'], norm_out_key)
    pixart.add_edge(out_dict['emb'], norm_out_key)

    proj_out_key = f"model.proj_out.weight_quantizer"
    proj_out_universal_feats = convert_universal_layer_feats_to_vector(pixart_quant_dict[proj_out_key])
    proj_out_pixart_feats = t.Tensor([-1, PIXART_TSX_LAYERS_LUT["proj_out"]])
    proj_out_feats = t.concat([proj_out_universal_feats, proj_out_pixart_feats])
    pixart.add_node(proj_out_key, all=proj_out_feats)
    pixart.add_edge(norm_out_key, proj_out_key)

    del pixart_quant_dict['model.proj_out.weight_quantizer']

    #visualize_graph(pixart, "pixart")

    tg_sample = tg.utils.convert.from_networkx(pixart, group_node_attrs = ["all"])
    tg_sample.y = y
    return tg_sample


if __name__ == "__main__":
    import pickle
    with open("sdm/alpha_sdm_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    tg_graph = construct_pixart_tg_graph(cache[0]['config'])
    print(tg_graph.x.shape)
    print(tg_graph.edge_index.shape)
