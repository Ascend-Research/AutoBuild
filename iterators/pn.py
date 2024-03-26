from ge_utils.torch_geo_data import _mobilenet_custom_as_torch_geo
import itertools


# Units - [1, 2, 3, 4, 5] for PN
# Layers - [2, 3, 4] for PN
# e = [3, 4, 6]
# k = [3, 5, 7]
def gen_sgs(units=[1, 2, 3, 4, 5], layers=[2, 3, 4], e=[3, 4, 6], k=[3, 5, 7], resolution=224):

    stage_cfg_list = []
    subgraph_list = []
    all_valid_mbconv = list(itertools.product(*[e, k]))

    # We merge the 5th and 6th units of PN together, since the 6th unit only has 1 layer.
    # Therefore, the combined 5th and 6th units have 3-5 layers, not 2-4
    if 5 in units:
        layers.append(5)
    for layer in layers:
        layer_sequences = list(itertools.product(*[all_valid_mbconv], repeat=layer))
        for layer_seq in layer_sequences:
            stage_cfg_list.append([f"mbconv2_e{e}_k{k}" for e, k in layer_seq])
        
    for unit in units:
        for cfg in stage_cfg_list:
            if unit == 5 and len(cfg) >= 3:
                bi_unit_cfg = [cfg[:-1], [cfg[-1]]]
                subgraph = _mobilenet_custom_as_torch_geo(bi_unit_cfg, resolution=resolution, unit=unit-1, layer=0)
                subgraph_dict = {'config': bi_unit_cfg,
                                 'hops': len(bi_unit_cfg[0])}
            elif unit < 5 and len(cfg) < 5:
                subgraph = _mobilenet_custom_as_torch_geo([cfg], resolution=resolution, unit=unit-1, layer=0)
                subgraph_dict = {'config': cfg,
                                 'hops': len(cfg) - 1}
            else:
                continue
            subgraph_dict['unit'] = unit
            subgraph_dict['tg_subgraph'] = subgraph
            subgraph_list.append(subgraph_dict)
    return subgraph_list


def assemble_whole_net_cfg(u1, u2, u3, u4, u56):
    net_cfg = [u1, u2, u3, u4, *u56]
    collapsed_cfg = [item for sublist in net_cfg for item in sublist]
    assert len(collapsed_cfg) >= 11 and len(collapsed_cfg) <= 21
    return net_cfg
