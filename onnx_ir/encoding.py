from onnx_ir.node import Node
from onnx_ir.op import OP2IDX
from onnx_ir.graph import Graph


C_AXIS = 1 # Only supports BCHW
NODE_ATTR_FEAT_SIZE = 9
MAX_N_SHAPES = 2

C_NORMALIZER = 256.
H_NORMALIZER = 128.
W_NORMALIZER = 128.
KERNEL_NORMALIZER = 3.


def _default_attr_feat_list():
    return [0. for _ in range(NODE_ATTR_FEAT_SIZE)]


def _build_conv_attr_feat(node:Node):
    """
    Reserves: first 3 dims for kernel sizes
    1 dim for group
    3 dims for dilations
    1 dim for has bias
    """
    feat = _default_attr_feat_list()
    kernel_sizes = [v / KERNEL_NORMALIZER for v in node.kernel_size]
    while len(kernel_sizes) < 3:
        kernel_sizes.append(0.)
    group = node.group / node.output_shapes[0][C_AXIS]
    dilations = node.dilations
    if dilations is None:
        dilations = [1. for _ in kernel_sizes]
    while len(dilations) < 3:
        dilations.append(0.)
    feat[0:3] = kernel_sizes
    feat[3] = group
    feat[5:8] = dilations
    feat[8] = 1 if node.has_bias else 0.
    return feat


def _build_pool_attr_feat(node:Node):
    """
    Reserves: first 3 dims for kernel sizes
    """
    feat = _default_attr_feat_list()
    kernel_sizes = [v / KERNEL_NORMALIZER for v in node.kernel_size]
    while len(kernel_sizes) < 3:
        kernel_sizes.append(0.)
    feat[0:3] = kernel_sizes
    return feat


_ATTRIBUTE_FEAT_BUILDER = {
    "Conv": _build_conv_attr_feat,
    "ConvTranspose": _build_conv_attr_feat,
    "Maxpool": _build_pool_attr_feat,
    "AveragePool": _build_pool_attr_feat,
}


def get_norm_shape_feat(shape, normalizers):
    rv = []
    for v, n in zip(shape, normalizers):
        if n is None:
            continue
        rv.append( v / n )
    return rv


def get_node_features(
        node:Node,
        op2idx,
        normalizers=(None, C_NORMALIZER, H_NORMALIZER, W_NORMALIZER),
        ext_attr_feat=None,
):
    """
    :param node:
    :param op2idx:
    :param normalizers: None means do not save that value, default is CHW
    :param ext_attr_feat:
    """
    input_shapes = node.input_shapes[:MAX_N_SHAPES]
    output_shapes = node.output_shapes[:MAX_N_SHAPES]
    input_shape_feat = [get_norm_shape_feat(s, normalizers) for s in input_shapes]
    output_shape_feat = [get_norm_shape_feat(s, normalizers) for s in output_shapes]
    input_shape_feat = [item for sublist in input_shape_feat for item in sublist]
    output_shape_feat = [item for sublist in output_shape_feat for item in sublist]
    while len(input_shape_feat) < MAX_N_SHAPES * len([v for v in normalizers if v is not None]):
        input_shape_feat.append(0.)
    while len(output_shape_feat) < MAX_N_SHAPES * len([v for v in normalizers if v is not None]):
        output_shape_feat.append(0.)

    default_attr_feat = _default_attr_feat_list()
    if node.op_type in _ATTRIBUTE_FEAT_BUILDER:
        attr_feat = _ATTRIBUTE_FEAT_BUILDER[node.op_type](node)
        assert len(attr_feat) == len(default_attr_feat)
    else:
        attr_feat = default_attr_feat
    if ext_attr_feat is not None:
        attr_feat += ext_attr_feat

    return {
        "op_type_idx": op2idx[node.op_type],
        "input_shape_feat": input_shape_feat,
        "output_shape_feat": output_shape_feat,
        "attr_feat": attr_feat,
        "node_id": node.str_id,
    }


def get_graph_features(
        graph:Graph,
        normalizers=(None, C_NORMALIZER, H_NORMALIZER, W_NORMALIZER),
        node_attr_feat_map=None,
):
    """
    Provides a default way to get some raw graph features from a onnx_ir graph
    Each node feature has 3 parts:
    1. Node op type index, as defined in op.py
    2. Node io shapes, could be more than one for input or output
    3. Node attributes, for conv that include kernel size, group, dilations, bias. Not every node has this
    Edge feat will be torch geo style edge lists, matching the nodes order
    """
    op2idx = OP2IDX()

    nodes = graph.nodes
    src2dst = graph.src2dst
    node2idx = {n.str_id:i for i, n in enumerate(nodes)}

    edge_pairs = []
    for src_id, dst_ids in src2dst.items():
        for dst_id in dst_ids:
            edge = (node2idx[src_id], node2idx[dst_id])
            edge_pairs.append(edge)
    edge_list = [[p[0] for p in edge_pairs], [p[1] for p in edge_pairs]]

    node_feats = [
        get_node_features(
            node=node,
            op2idx=op2idx,
            normalizers=normalizers,
            ext_attr_feat=node_attr_feat_map[node.str_id] if node_attr_feat_map is not None else None,
        )
        for node in nodes
    ]

    return node_feats, edge_list


def get_edge_as_node_features(
        graph:Graph,
        edge_as_nodes,
        edge_as_nodes_adj,
        normalizers=(None, C_NORMALIZER, H_NORMALIZER, W_NORMALIZER),
        node_attr_feat_map=None,
):
    """
    Given a graph, and a new graph where each edge is a node.
    The new node feature will be the combined feature of original nodes at the edge end points.
    This is useful if we need to make predictions about edges.
    """
    node_feats, edge_list = get_graph_features(
        graph=graph,
        normalizers=normalizers,
        node_attr_feat_map=node_attr_feat_map,
    )

    node_id2feat = {fd["node_id"]: fd for fd in node_feats}
    assert len(node_id2feat) == len(node_feats)

    new_node2idx = {n: i for i, n in enumerate(edge_as_nodes)}
    new_node_feats = []
    for new_node in edge_as_nodes:
        n_id_1, n_id_2 = new_node
        feat_1, feat_2 = node_id2feat[n_id_1], node_id2feat[n_id_2]
        new_node_feats.append((feat_1, feat_2))

    new_edge_pairs = []
    for src_id, dst_ids in edge_as_nodes_adj.items():
        for dst_id in dst_ids:
            edge = (new_node2idx[src_id], new_node2idx[dst_id])
            new_edge_pairs.append(edge)
    new_edge_list = [[p[0] for p in new_edge_pairs], [p[1] for p in new_edge_pairs]]

    return new_node_feats, new_edge_list
