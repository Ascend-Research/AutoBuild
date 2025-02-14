from constants import MBCONV_EXPAND_MAP
from constants import MBCONV_KERNEL_MAP
import torch
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.data import Data
from onnx_ir.node import Node
from onnx_ir.graph import Graph
from onnx_ir.encoding import get_node_features
from onnx_ir.op import OP2IDX
from ge_utils.custom_embeds import discern_custom_family
from sdm.graph_dit import construct_dit_tg_graph
from sdm.graph_sdv15 import construct_sdv15_tg_graph
from sdm.graph_sdxl import construct_sdxl_tg_graph
from sdm.graph_pixart import construct_pixart_tg_graph
from sdm.graph_hunyuan import construct_hunyuan_tg_graph

op2i = OP2IDX()


def _node_feats_as_list(node:Node, op2idx):

    node_dict = get_node_features(node, op2idx)
    node_list = [node_dict['op_type_idx']] + node_dict['input_shape_feat'] + \
        node_dict['output_shape_feat'] + node_dict['attr_feat']
    return node_list


def get_entry_as_torch_geo(entry:dict, undirected=False, format=None, y=None):

    if format is not None and format == "custom":
        name, entry = discern_custom_family(entry)
        cg_config = entry["original config"]
        if name == "mobilenet":
            return _mobilenet_custom_as_torch_geo(cg_config[1], resolution=cg_config[0], undirected=undirected, y=y)
        else:
            raise NotImplementedError("Unknown family")
    elif format is not None and format == "onnx_ir":
        assert "graph" in entry.keys()
        return _get_graph_as_torch_geo(entry['graph'], undirected=undirected, y=y)
    elif format == "sdm":
        assert "architecture" in entry.keys()
        if entry["architecture"] == 'DiT':
            return construct_dit_tg_graph(entry['config'], y=y)
        elif entry["architecture"] == 'SDv1.5':
            return construct_sdv15_tg_graph(entry['config'], y=y)
        elif entry["architecture"] == 'SDXL':
            return construct_sdxl_tg_graph(entry['config'], y=y)
        elif entry["architecture"] in ['alpha', 'sigma']:
            return construct_pixart_tg_graph(entry['config'], y=y)
        elif entry["architecture"] == "hunyuan":
            return construct_hunyuan_tg_graph(entry['config'], y=y)
        else:
            raise NotImplementedError("Unknown SDM model")
    else:
        raise NotImplementedError("Unknown entry format")
    

def _get_graph_as_torch_geo(graph:Graph, undirected=False, y=None):
    nodes = graph.nodes
    src2dst = graph.src2dst
    node2idx = {n.str_id:i for i, n in enumerate(nodes)}

    edge_pairs = []
    for src_id, dst_ids in src2dst.items():
        for dst_id in dst_ids:
            edge = (node2idx[src_id], node2idx[dst_id])
            edge_pairs.append(edge)
    edge_list = [[p[0] for p in edge_pairs], [p[1] for p in edge_pairs]]

    edge_index = torch.LongTensor(edge_list)
    assert edge_index.shape[0] == 2 and edge_index.shape[1] == len(edge_pairs)
    if undirected:
        edge_index = to_undirected(edge_index)
        assert edge_index.shape[1] == 2 * len(edge_pairs)

    feature_list = [
        _node_feats_as_list(node=node, op2idx=op2i)
        for node in nodes
    ]

    x = torch.Tensor(feature_list)
    assert x.shape[0] == len(nodes)

    if y is not None:
        y = torch.Tensor(y)

    return Data(x=x, edge_index=edge_index, y=y)


def _mobilenet_custom_as_torch_geo(config_list, resolution=224, undirected=False, y=None, unit=0, layer=0):

    feature_list = []
    res_transform = (resolution - 192) / 32
    for i in range(len(config_list)):  # Stage number positional encoding
        blk_list = config_list[i]
        for j in range(len(blk_list)): # Block number positional encoding
            # Offset added - layer offset affects first unit we consider, not others.
            node_list = [i + unit, j if unit > 0 else j + layer]  # Unit, Layer
            op_str_split = blk_list[j].split("_")
            node_list.append(int(op_str_split[0][-1]) - 2.)  # Whether is MBConv2 or MBConv3; 2 -> 0, 3-> 1
            node_list.append(MBCONV_EXPAND_MAP[int(op_str_split[1][-1])])  # Expansion Ratio
            node_list.append(MBCONV_KERNEL_MAP[int(op_str_split[-1][-1])]) # Kernel size
            node_list.append(res_transform)  # Resolution scaled between [0, 1.]
            feature_list.append(node_list)
    x = torch.Tensor(feature_list)
    edge_list = []
    for i in range(len(feature_list) - 1):
        edge_list.append([i, i+1])
        if undirected:
            edge_list.append([i+1, i])
    edge_index = torch.LongTensor(edge_list).T
    return Data(x=x, edge_index=edge_index, y=y)
