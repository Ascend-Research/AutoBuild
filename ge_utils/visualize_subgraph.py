from onnx_ir.graph import Graph
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


def visualize_onnx_ir_subgraph(graph:Graph,
                    node_list,
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
    sg_node_list = [graph.nodes[i] for i in node_list]
    sg_str_ids = [n.str_id for n in sg_node_list]
    for n in sg_node_list:
        dot.node(n.str_id, n.op_type if use_simple_node_label else str(n), fillcolor="brown2", fontcolor="white")
        node_j_list = []
        if n.str_id in graph.src2dst.keys():
            node_j_list = [j for j in graph.src2dst[n.str_id] if j in sg_str_ids]
        for j in node_j_list:
            dot.edge(n.str_id, j)
    _resize_graph(dot, min_size=min_size)
    dot.render(view=view, directory=output_dir,
               format=img_format, cleanup=cleanup)
    

def visualize_mobilenet_subgraph(config, geo_data, node_list,
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

    node_list.sort()
    res, node_label_list = config['original config']
    node_label_list = [item for sublist in node_label_list for item in sublist]
    for i in node_list:
        n = node_label_list[i]
        pos_data = "-".join([str(int(x) + 1) for x in geo_data.x[i, :2].squeeze().tolist()])
        dot.node(str(i), pos_data + "_" + n + f"_{res}", fillcolor="brown2", fontcolor="white")
        if i != node_list[-1]:
            dot.edge(str(i), str(i+1))
    _resize_graph(dot, min_size=min_size)
    dot.render(view=view, directory=output_dir,
               format=img_format, cleanup=cleanup)


def visualize_subgraph(entry, node_list, filename="temp", output_dir=None, undirected=False):
    if 'graph' in entry.keys():
        visualize_onnx_ir_subgraph(entry['graph'], node_list, view=False, filename=filename, output_dir=output_dir, undirected=undirected)
    else:
        name, entry = discern_custom_family(entry)
        plt_func = eval(f"visualize_{name}_subgraph")
        graph_geo = get_entry_as_torch_geo(entry, undirected=undirected, format="custom")
        plt_func(entry, graph_geo, node_list, view=False, filename=filename, output_dir=output_dir, undirected=undirected)