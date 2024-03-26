from ge_utils.torch_geo_data import _mobilenet_custom_as_torch_geo
import itertools


# Units - [1, 2, 3, 4, 5] for MBv3
# Layers - [2, 3, 4] for MBv3
# e = [3, 4, 6]
# k = [3, 5, 7]
def gen_sgs(units=[1, 2, 3, 4, 5], layers=[2, 3, 4], e=[3, 4, 6], k=[3, 5, 7], resolution=224):

    stage_cfg_list = []
    subgraph_list = []
    all_valid_mbconv = list(itertools.product(*[e, k]))

    for layer in layers:
        layer_sequences = list(itertools.product(*[all_valid_mbconv], repeat=layer))
        for layer_seq in layer_sequences:
            stage_cfg_list.append([f"mbconv3_e{e}_k{k}" for e, k in layer_seq])
        
    for unit in units:
        for cfg in stage_cfg_list:
            subgraph = _mobilenet_custom_as_torch_geo([cfg], resolution=resolution, unit=unit-1, layer=0)
            subgraph_dict = {'config': cfg,
                             'tg_subgraph': subgraph,
                             'unit': unit,
                             'hops': len(cfg) - 1}
            subgraph_list.append(subgraph_dict)
    
    return subgraph_list


def assemble_whole_net_cfg(u1, u2, u3, u4, u5):
    net_cfg = [u1, u2, u3, u4, u5]
    collapsed_cfg = [item for sublist in net_cfg for item in sublist]
    assert len(collapsed_cfg) >= 10 and len(collapsed_cfg) <= 20
    return net_cfg
