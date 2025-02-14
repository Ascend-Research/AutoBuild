from onnx_ir.graph import Graph
import numpy as np
import os
from ge_utils.torch_geo_data import get_entry_as_torch_geo
import torch as t
from ge_utils.custom_embeds import discern_custom_family


MAX_COLOR = 255
             
                    
def _resize_graph(dot, size_per_element=1.0, min_size=18):
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def visualize_onnx_ir_nembeds(graph:Graph,
                    tg_x,
                    view=True, output_dir=None, filename=None, undirected=False,
                    img_format="png", cleanup=True,
                    use_simple_node_label=False,
                    min_size=18):
    from graphviz import Digraph, Graph
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='14',
                     ranksep='0.1',
                     height='0.2')
    if undirected:
        dot = Graph(node_attr=node_attr, filename=filename,
                    graph_attr=dict(size="15,15"))
    else:
        dot = Digraph(node_attr=node_attr, filename=filename,
                      graph_attr=dict(size="15,15"))
    node_max, node_min = np.max(tg_x), np.min(tg_x)
    for i, n in enumerate(graph.nodes):
        z = (tg_x[i] - node_min) / (node_max - node_min)
        color_decimal = int(z*MAX_COLOR)
        color_hex = hex(color_decimal)[2:]
        if len(color_hex) == 1:
            color_hex = "0" + color_hex
        color = f"#{color_hex}6262" 
        dot.node(n.str_id, n.op_type if use_simple_node_label else str(n), fillcolor=color, fontcolor="white")
    for src_id, dst_ids in graph.src2dst.items():
        for dst_id in dst_ids:
            dot.edge(src_id, dst_id)
    _resize_graph(dot, min_size=min_size)
    dot.render(view=view, directory=output_dir,
               format=img_format, cleanup=cleanup)
    

def visualize_mobilenet_nembeds(config, geo_data, tg_x,
                    view=True, output_dir=None, filename=None, undirected=False,
                    img_format="png", cleanup=True,
                    min_size=18):
    from graphviz import Digraph, Graph
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='14',
                     ranksep='0.1',
                     height='0.2')
    if undirected:
        dot = Graph(node_attr=node_attr, filename=filename,
                    graph_attr=dict(size="15,15"))
    else:
        dot = Digraph(node_attr=node_attr, filename=filename,
                      graph_attr=dict(size="15,15"))
    node_max, node_min = np.max(tg_x), np.min(tg_x)
    res, node_label_list = config['original config']
    node_label_list = [item for sublist in node_label_list for item in sublist]
    for i, n in enumerate(node_label_list):
        z = (tg_x[i] - node_min) / (node_max - node_min)
        color_decimal = int(z*MAX_COLOR)
        color_hex = hex(color_decimal)[2:]
        if len(color_hex) == 1:
            color_hex = "0" + color_hex
        color = f"#{color_hex}6262" 
        unit_layer = str(int(geo_data.x[i, 0].item())+1) + "-" + str(int(geo_data.x[i, 1].item())+1)
        node_str = "_".join([unit_layer, n, str(res)])
        dot.node(str(i), node_str, fillcolor=color, fontcolor="white")
        if i < len(node_label_list) - 1:
            dot.edge(str(i), str(i+1))
    _resize_graph(dot, min_size=min_size)
    dot.render(view=view, directory=output_dir,
               format=img_format, cleanup=cleanup)
    

def gen_custom_embed_plot(predictor, entry, fname, norm=1, gen_all=False, undirected=False):
    name, entry = discern_custom_family(entry)
    plt_func = eval(f"visualize_{name}_nembeds")
    graph_geo = get_entry_as_torch_geo(entry, undirected=undirected, format="custom")
    with t.no_grad():
        embedding_list = predictor.get_gnn_node_embeds(graph_geo)

    if gen_all:
        for i in range(len(embedding_list)):
            embedding_norm = t.linalg.norm(embedding_list[i], ord=norm, dim=-1).detach().cpu().tolist()
            plt_func(entry, graph_geo, embedding_norm, view=False, filename="_".join([fname, "layer", str(i)]), undirected=undirected)
    else:
        embedding_norm = t.linalg.norm(embedding_list[-1], ord=norm, dim=-1).detach().cpu().tolist()
        plt_func(entry, graph_geo, embedding_norm, view=False, filename=fname, undirected=undirected)
    
def gen_onnx_embed_plot(predictor, entry, fname, norm=1, gen_all=False, undirected=False):

    graph = entry['graph']
    graph_geo = get_entry_as_torch_geo(entry, undirected=undirected, format="onnx_ir")
    with t.no_grad():
        embedding_list = predictor.get_gnn_node_embeds(graph_geo)

    if gen_all:
        for i in range(len(embedding_list)):
            embedding_norm = t.linalg.norm(embedding_list[i], ord=norm, dim=-1).detach().cpu().tolist()
            visualize_onnx_ir_nembeds(graph, embedding_norm, view=False, filename="_".join([fname, "layer", str(i)]), undirected=undirected)
    else:
        embedding_norm = t.linalg.norm(embedding_list[-1], ord=norm, dim=-1).detach().cpu().tolist()
        visualize_onnx_ir_nembeds(graph, embedding_norm, view=False, filename=fname, undirected=undirected)
