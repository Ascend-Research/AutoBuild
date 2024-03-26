import copy
import networkx as nx

INPUT_DUMMY_OP_TYPE = "input_dummy"
OUTPUT_DUMMY_OP_TYPE = "output_dummy"
DUMMY_IO_OP_TYPES = {
    INPUT_DUMMY_OP_TYPE,
    OUTPUT_DUMMY_OP_TYPE
}


class Graph:
    """
    An in-memory graph for an ONNX model
    """
    def __init__(self, onnx_meta_info, nodes,
                 metadata=None, build_from_nodes=True):
        self.onnx_meta_info = onnx_meta_info
        self.metadata = metadata if metadata is not None else {}
        # The following attributes are built from the nodes
        self.nodes = None
        self.src2dst = None
        self.input_nodes = None
        self.output_nodes = None
        if build_from_nodes:
            self._build(nodes)

    @property
    def boundary_output_nodes(self):
        return [n for n in self.nodes if n.str_id not in self.src2dst or len(self.src2dst[n.str_id]) == 0]

    @property
    def name(self):
        if "onnx_graph_name" in self.metadata:
            return self.metadata["onnx_graph_name"]
        return ""

    @property
    def id2node(self):
        _id2node = {}
        for node in self.nodes:
            assert node.str_id not in _id2node
            _id2node[node.str_id] = node
        return _id2node

    @property
    def internal_nodes(self):
        """
        Nodes that are not dummy input or output
        """
        input_node_ids = {_n.str_id for _n in self.input_nodes}
        output_node_ids = {_n.str_id for _n in self.output_nodes}
        return [_n for _n in self.nodes if _n.str_id not in input_node_ids.union(output_node_ids)]

    def _build(self, nodes):
        # Build adj dict from nodes
        src2dst = self._build_src2dst_dict(nodes)

        # Build nodes with dummy input/output nodes
        nodes_complete, src2dst, input_nodes, output_nodes = \
            self.add_dummy_io_nodes(nodes, src2dst)

        # Set io nodes, these won't ever change
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.set_nodes_adj_dict(nodes_complete, src2dst)

    @property
    def num_edges(self):
        return sum(len(v) for v in self.src2dst.values())

    @property
    def num_nodes(self):
        return len(self.nodes)

    def __len__(self):
        return self.num_nodes

    def __str__(self):
        return f"Graph[name={self.name}, num_nodes={self.num_nodes}, num_edges={self.num_edges}]"

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        cg = Graph(copy.deepcopy(self.onnx_meta_info),
                   [copy.deepcopy(n) for n in self.internal_nodes],
                   build_from_nodes=False)
        cg.nodes = copy.deepcopy(self.nodes)
        cg.src2dst = copy.deepcopy(self.src2dst)
        cg.set_nodes_adj_dict(cg.nodes, cg.src2dst)
        if self.metadata is not None:
            cg.metadata = {k: v for k, v in self.metadata.items()}
        return cg
    
    def to_nx(self):
        g = nx.DiGraph()
        for node in self.nodes:
            g.add_node(node.str_id, **node.to_nx())
        for i, js in self.src2dst.items():
            for j in js:
                g.add_edge(i, j)
        return g
