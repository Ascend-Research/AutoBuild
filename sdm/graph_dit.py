import torch as t
import networkx as nx
import torch_geometric as tg
from collections import OrderedDict
from sdm.graph_util import convert_universal_layer_feats_to_vector, gen_combine_node, visualize_graph

DIT_TSX_COMBINE_LUT = OrderedDict()
DIT_TSX_COMBINE_LUT["add"] = 0
DIT_TSX_COMBINE_LUT["matmul"] = 0
DIT_TSX_COMBINE_LUT["adaln"] = 0
DIT_TSX_LAYERS_LUT = OrderedDict()
DIT_TSX_LAYERS_LUT["norm1.linear"] = 1
DIT_TSX_LAYERS_LUT["attn1.to_q"] = 2
DIT_TSX_LAYERS_LUT["attn1.to_k"] = 3
DIT_TSX_LAYERS_LUT["attn1.to_v"] = 4
DIT_TSX_LAYERS_LUT["attn1.to_out.0"] = 5
DIT_TSX_LAYERS_LUT["ff.net.0.proj"] = 6
DIT_TSX_LAYERS_LUT["ff.net.2"] = 7
DIT_TSX_LAYERS_LUT["pos_embed.proj"] = 8
DIT_TSX_COMBINE_LUT["transformer_blocks.0.norm1.emb.timestep_embedder.linear_1"] = 9
DIT_TSX_COMBINE_LUT["transformer_blocks.0.norm1.emb.timestep_embedder.linear_2"] = 10
DIT_TSX_COMBINE_LUT["proj_out_1"] = 11
DIT_TSX_COMBINE_LUT["proj_out_2"] = 12

DIT_POS_EMBED_LAYER_0_KEY = 'model.pos_embed.proj.weight_quantizer'
DIT_TIME_EMBED_LAYER_1_KEY = 'model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_1.weight_quantizer'
DIT_TIME_EMBED_LAYER_2_KEY = 'model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_2.weight_quantizer'


def convert_dit_layer_to_vector(dit_k, dit_v):
    if type(dit_v) == t.Tensor:
        return dit_v
    elif "transformer_blocks" in dit_k and "timestep_embedder" not in dit_k:
        return convert_dit_tsx_layer_to_vector(dit_k, dit_v)
    elif "pos_embed" in dit_k:
        blk_code = 0
        layer_code = DIT_TSX_LAYERS_LUT["pos_embed.proj"]
    else:
        blk_code = -1
        dit_k_trimmed = dit_k.replace(".weight_quantizer", "")
        if dit_k_trimmed.endswith("proj_out_1"):
            layer_code = DIT_TSX_COMBINE_LUT["proj_out_1"]
        elif dit_k_trimmed.endswith("proj_out_2"):
            layer_code = DIT_TSX_COMBINE_LUT["proj_out_2"]
        elif dit_k_trimmed.endswith("timestep_embedder.linear_1"):
            layer_code = DIT_TSX_COMBINE_LUT["transformer_blocks.0.norm1.emb.timestep_embedder.linear_1"]
        elif dit_k_trimmed.endswith("timestep_embedder.linear_2"):
            layer_code = DIT_TSX_COMBINE_LUT["transformer_blocks.0.norm1.emb.timestep_embedder.linear_2"]
        else:
            raise NotImplementedError
    universal_feats = convert_universal_layer_feats_to_vector(dit_v)
    dit_specific_feats = t.Tensor([blk_code, layer_code])
    return t.cat([universal_feats, dit_specific_feats])


def convert_dit_tsx_layer_to_vector(dit_k, dit_v):
    assert dit_k.startswith("model.transformer_blocks.") and "timestep_embedder" not in dit_k
    dit_k = dit_k.replace("model.transformer_blocks.", "").replace(".weight_quantizer", "")
    dit_k_list = dit_k.split(".")
    blk_code = int(dit_k_list[0])
    layer_type = ".".join(dit_k_list[1:])
    layer_code = DIT_TSX_LAYERS_LUT[layer_type]
    universal_feats = convert_universal_layer_feats_to_vector(dit_v)
    dit_specific_feats = t.Tensor([blk_code, layer_code])
    return t.cat([universal_feats, dit_specific_feats])


def _generate_dit_timestep_sg(layer_1_feats, layer_2_feats):
    time_embed_graph = nx.DiGraph()
    
    layer_1_universal_feats = convert_universal_layer_feats_to_vector(layer_1_feats)
    layer_1_dit_feats = t.Tensor([-1, DIT_TSX_COMBINE_LUT["transformer_blocks.0.norm1.emb.timestep_embedder.linear_1"]])
    layer_1_feats = t.concat([layer_1_universal_feats, layer_1_dit_feats])
    time_embed_graph.add_node(DIT_TIME_EMBED_LAYER_1_KEY, all=layer_1_feats)

    layer_2_universal_feats = convert_universal_layer_feats_to_vector(layer_2_feats)
    layer_2_dit_feats = t.Tensor([-1, DIT_TSX_COMBINE_LUT["transformer_blocks.0.norm1.emb.timestep_embedder.linear_2"]])
    layer_2_feats = t.concat([layer_2_universal_feats, layer_2_dit_feats])
    time_embed_graph.add_node(DIT_TIME_EMBED_LAYER_2_KEY, all=layer_2_feats)
    time_embed_graph.add_edge(DIT_TIME_EMBED_LAYER_1_KEY, DIT_TIME_EMBED_LAYER_2_KEY)

    return time_embed_graph


def _generate_dit_proj_out_sg(p_1_feats, p_2_feats):
    out_sg = nx.DiGraph()
    proj_1_key = "model.proj_out_1.weight_quantizer"
    proj_1_universal_feats = convert_universal_layer_feats_to_vector(p_1_feats)
    proj_1_dit_feats = t.Tensor([-1, DIT_TSX_COMBINE_LUT["proj_out_1"]])
    proj_1_feats = t.concat([proj_1_universal_feats, proj_1_dit_feats])
    out_sg.add_node(proj_1_key, all=proj_1_feats)

    proj_2_key = "model.proj_out.2.weight_quantizer"
    proj_2_universal_feats = convert_universal_layer_feats_to_vector(p_2_feats)
    proj_2_dit_feats = t.Tensor([-1, DIT_TSX_COMBINE_LUT["proj_out_2"]])
    proj_2_feats = t.concat([proj_2_universal_feats, proj_2_dit_feats])
    out_sg.add_node(proj_2_key, all=proj_2_feats)
    out_sg.add_edge(proj_1_key, proj_2_key)

    return out_sg

def dit_blk_feat_mat(dit_quant_dict, blk_id, prefix="model.transformer_blocks."):
    transformer = nx.DiGraph()

    for weight_type in DIT_TSX_LAYERS_LUT.keys():
        if "pos_embed" in weight_type:
            continue
        key_str = f"{prefix}{str(blk_id)}.{weight_type}.weight_quantizer"
        transformer.add_node(key_str, all=convert_dit_tsx_layer_to_vector(key_str, dit_quant_dict[key_str]))
        del dit_quant_dict[key_str]

    adaln0_key = f"{prefix}{str(blk_id)}.adaln_0"
    adaln0_universal_feats = gen_combine_node()
    adaln0_specific_feats = t.Tensor([blk_id, DIT_TSX_COMBINE_LUT["adaln"]])
    adaln0_feats = t.cat([adaln0_universal_feats, adaln0_specific_feats])
    transformer.add_node(adaln0_key, all=adaln0_feats)
    transformer.add_edge(f"{prefix}{str(blk_id)}.norm1.linear.weight_quantizer", adaln0_key)

    in_dict = {'adanln0': adaln0_key, "norm1": f"{prefix}{str(blk_id)}.norm1.linear.weight_quantizer"}

    transformer.add_edge(adaln0_key, f"{prefix}{str(blk_id)}.attn1.to_q.weight_quantizer")
    transformer.add_edge(adaln0_key, f"{prefix}{str(blk_id)}.attn1.to_k.weight_quantizer")
    transformer.add_edge(adaln0_key, f"{prefix}{str(blk_id)}.attn1.to_v.weight_quantizer")

    qk_key = f"{prefix}{str(blk_id)}.mul_qk"
    qk_universal_feats = gen_combine_node()
    qk_specific_feats = t.Tensor([blk_id, DIT_TSX_COMBINE_LUT["matmul"]])
    qk_feats = t.cat([qk_universal_feats, qk_specific_feats])
    transformer.add_node(qk_key, all=qk_feats)
    transformer.add_edge(f"{prefix}{str(blk_id)}.attn1.to_q.weight_quantizer", qk_key)
    transformer.add_edge(f"{prefix}{str(blk_id)}.attn1.to_k.weight_quantizer", qk_key)

    vsm_key = f"{prefix}{str(blk_id)}.mul_vsm"
    vsm_universal_feats = gen_combine_node()
    vsm_specific_feats = t.Tensor([blk_id, DIT_TSX_COMBINE_LUT["matmul"]])
    vsm_feats = t.cat([vsm_universal_feats, vsm_specific_feats])
    transformer.add_node(vsm_key, all=vsm_feats)
    transformer.add_edge(f"{prefix}{str(blk_id)}.attn1.to_v.weight_quantizer", vsm_key)
    transformer.add_edge(qk_key, vsm_key)

    transformer.add_edge(vsm_key, f"{prefix}{str(blk_id)}.attn1.to_out.0.weight_quantizer")

    sa_add_key = f"{prefix}{str(blk_id)}.add_sa"
    sa_add_universal_features = gen_combine_node()
    sa_add_specific_features = t.Tensor([blk_id, DIT_TSX_COMBINE_LUT["add"]])
    sa_add_features = t.cat([sa_add_universal_features, sa_add_specific_features])
    transformer.add_node(sa_add_key, all=sa_add_features)
    transformer.add_edge(f"{prefix}{str(blk_id)}.attn1.to_out.0.weight_quantizer", sa_add_key)

    in_dict["add_sa"] = sa_add_key
    
    adaln1_key = f"{prefix}{str(blk_id)}.adaln_1"
    adaln1_universal_feats = gen_combine_node()
    adaln1_specific_feats = t.Tensor([blk_id, DIT_TSX_COMBINE_LUT["adaln"]])
    adaln1_feats = t.cat([adaln1_universal_feats, adaln1_specific_feats])
    transformer.add_node(adaln1_key, all=adaln1_feats)
    transformer.add_edge(sa_add_key, adaln1_key)
    transformer.add_edge(f"{prefix}{str(blk_id)}.norm1.linear.weight_quantizer", adaln1_key)

    transformer.add_edge(adaln1_key, f"{prefix}{str(blk_id)}.ff.net.0.proj.weight_quantizer")
    transformer.add_edge(f"{prefix}{str(blk_id)}.ff.net.0.proj.weight_quantizer", f"{prefix}{str(blk_id)}.ff.net.2.weight_quantizer")

    ff_add_key = f"{prefix}{str(blk_id)}.add_ff"
    ff_add_universal_features = gen_combine_node()
    ff_add_specific_features = t.Tensor([blk_id, DIT_TSX_COMBINE_LUT["add"]])
    ff_add_features = t.cat([ff_add_universal_features, ff_add_specific_features])
    transformer.add_node(ff_add_key, all=ff_add_features)
    transformer.add_edge(f"{prefix}{str(blk_id)}.ff.net.2.weight_quantizer", ff_add_key)
    transformer.add_edge(sa_add_key, ff_add_key)

    out_dict = {'ff': ff_add_key}
    return {'nx': transformer, 'in_dict': in_dict, 'out_dict': out_dict}


def construct_dit_tg_graph(dit_quant_dict, y=None):

    dit = nx.DiGraph()
    # Step 1 - encode vector for positional embedding conv layer
    layer_0_universal_feats = convert_universal_layer_feats_to_vector(dit_quant_dict[DIT_POS_EMBED_LAYER_0_KEY])
    layer_0_dit_feats = t.Tensor([-1, DIT_TSX_LAYERS_LUT["pos_embed.proj"]])
    layer_0_feats = t.concat([layer_0_universal_feats, layer_0_dit_feats])
    dit.add_node(DIT_POS_EMBED_LAYER_0_KEY, all=layer_0_feats)
    del dit_quant_dict[DIT_POS_EMBED_LAYER_0_KEY]

    # Step 2 - Timestep embedding. Encode. Have that create x and edge_index
    time_embed_graph = _generate_dit_timestep_sg(dit_quant_dict[DIT_TIME_EMBED_LAYER_1_KEY],
                                                   dit_quant_dict[DIT_TIME_EMBED_LAYER_2_KEY])
    dit = nx.union(dit, time_embed_graph)
    
    cur_out_key = DIT_POS_EMBED_LAYER_0_KEY
    # Step 3 - For-loop over all Transformer blocks
    for dit_blk_id in range(28):
        transformer_blk_dict = dit_blk_feat_mat(dit_quant_dict, dit_blk_id)
        dit = nx.union(dit, transformer_blk_dict['nx'])
        dit.add_edge(cur_out_key, transformer_blk_dict['in_dict']['adanln0'])
        dit.add_edge(cur_out_key, transformer_blk_dict['in_dict']['add_sa'])
        dit.add_edge(DIT_TIME_EMBED_LAYER_2_KEY, transformer_blk_dict['in_dict']['norm1'])
        cur_out_key = transformer_blk_dict['out_dict']['ff']
        del dit_quant_dict[DIT_TIME_EMBED_LAYER_1_KEY.replace(".0.", f".{dit_blk_id}.")]
        del dit_quant_dict[DIT_TIME_EMBED_LAYER_2_KEY.replace(".0.", f".{dit_blk_id}.")]

    # Step 4 - final layers
    proj_out_sg = _generate_dit_proj_out_sg(dit_quant_dict['model.proj_out_1.weight_quantizer'], dit_quant_dict['model.proj_out_2.weight_quantizer'])

    dit = nx.union(dit, proj_out_sg)
    dit.add_edge(cur_out_key, 'model.proj_out_1.weight_quantizer')
    del dit_quant_dict['model.proj_out_1.weight_quantizer']
    del dit_quant_dict['model.proj_out_2.weight_quantizer']

    assert len(dit_quant_dict.keys()) == 0, dit_quant_dict.keys()

    #visualize_graph(dit, "dit")

    tg_sample = tg.utils.convert.from_networkx(dit, group_node_attrs = ["all"])
    tg_sample.y = y
    return tg_sample


if __name__ == "__main__":
    import pickle
    with open("cache/dit_sdm_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    tg_graph = construct_dit_tg_graph(cache[0]['config'])
    print(tg_graph.x.shape)
    print(tg_graph.edge_index.shape)
