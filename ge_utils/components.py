import torch
import torch_geometric as tg
from onnx_ir.op import OP2IDX
from ge_utils.misc_utils import device
from ge_utils.analysis_utils import report_categorical


class NodeEmbedding(torch.nn.Module):

    def __init__(self,
                 n_ops, input_shape_size, input_attr_size,
                 op_embed_size, shape_embed_size, attr_embed_size):
        super(NodeEmbedding, self).__init__()
        self.op_embed_layer = torch.nn.Embedding(n_ops, op_embed_size)
        self.shape_embed_layer = torch.nn.Linear(input_shape_size, shape_embed_size)
        self.attr_embed_layer = torch.nn.Linear(input_attr_size, attr_embed_size)


class PreEmbeddedGraphEncoder(torch.nn.Module):

    def __init__(self,
                 in_channels, hidden_size, out_channels,
                 gnn_constructor,
                 activ=torch.nn.Tanh(), n_layers=4, dropout_prob=0.0,
                 add_normal_prior=False):
        super(PreEmbeddedGraphEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            input_size, output_size = hidden_size, hidden_size
            if i == 0:
                input_size = in_channels
            if i == n_layers - 1:
                output_size = out_channels
            gnn_layer = gnn_constructor(input_size, output_size)
            self.gnn_layers.append(gnn_layer)
        self.activ = activ
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.init_ff = torch.nn.Linear(2 * in_channels, in_channels)
        self.add_normal_prior = add_normal_prior


class TGeoNodeEmbedding(NodeEmbedding):
    def __init__(self,
                 n_ops, input_shape_size, input_attr_size,
                 op_embed_size, shape_embed_size, attr_embed_size):
        super(TGeoNodeEmbedding, self).__init__(n_ops, input_shape_size, input_attr_size,
                 op_embed_size, shape_embed_size, attr_embed_size)
        self.op_offset = 1
        self.shape_offset = self.op_offset + input_shape_size
        self.attr_offset = self.shape_offset + input_attr_size
        self.embed_total_dim = op_embed_size + shape_embed_size + attr_embed_size

        self.op_mlp = torch.nn.Linear(op_embed_size, 1)
        self.shape_mlp = torch.nn.Linear(shape_embed_size, 1)
        self.attr_mlp = torch.nn.Linear(attr_embed_size, 1)

        self.out_embed_size = 3
        self.rectify = torch.abs
        
        op2i = OP2IDX()
        self.obs_observed = torch.nn.Parameter(torch.zeros(len(op2i)), requires_grad=False)

    def forward(self, geo_x):
        ops = geo_x[:, :self.op_offset].long()
        self._count_ops(ops)
        shapes = geo_x[:, self.op_offset:self.shape_offset]
        attrs = geo_x[:, self.shape_offset:self.attr_offset]

        op_embedding = self.op_embed_layer(ops.to(device()))
        op_embedding = self.op_mlp(op_embedding.view(op_embedding.shape[0], -1))
        shape_embedding = self.shape_mlp(self.shape_embed_layer(shapes.to(device())))
        attr_embedding = self.attr_mlp(self.attr_embed_layer(attrs.to(device())))
        output = torch.cat([op_embedding, shape_embedding, attr_embedding], dim=-1)

        return self.rectify(output)
    
    def _count_ops(self, ops):
        ops_vec = ops.view(-1)
        self.obs_observed[ops_vec] += 1
 
    def analyze_node_feats(self):
        print("Running FE-MLP Analysis for ONNX-IR:")
        self._analyze_operations()

    def _analyze_operations(self):
        from onnx_ir.op import OP2IDX
        op2i = OP2IDX()
        print("     Analyze operations:")
        op_scores = {}
        with torch.no_grad():
            for i in range(len(op2i)):
                if self.obs_observed[i] > 0.:
                    op_tensor = torch.Tensor([i]).long()
                    op_embed = self.op_embed_layer(op_tensor.to(device()))
                    op_embed = op_embed.view(op_embed.shape[0], -1)
                    mlp_output = self.rectify(self.op_mlp(op_embed))
                    op_scores[op2i.query_op(i)] = mlp_output.cpu().squeeze().item()
        report_categorical(op_scores)


class TGeoPreEmbeddedGraphEncoder(PreEmbeddedGraphEncoder):
    def __init__(self, in_channels, hidden_size, out_channels, gnn_constructor,
                 activ=torch.nn.ReLU(), n_layers=4, dropout_prob=0.0, residual=False):
        super(TGeoPreEmbeddedGraphEncoder, self).__init__(in_channels=in_channels, hidden_size=hidden_size,
                                                out_channels=out_channels, gnn_constructor=gnn_constructor,
                                                activ=activ, n_layers=n_layers, dropout_prob=dropout_prob)

        bns = []
        for _ in range(n_layers - 1):
            bns.append(tg.nn.norm.BatchNorm(hidden_size))
        bns.append(tg.nn.norm.BatchNorm(out_channels))
        self.bns = torch.nn.ModuleList(bns)
        self.residual = residual
            
    def forward(self, torch_geo_batch, node_embedding):
        return self.forward_embed_list(torch_geo_batch, node_embedding)[-1]
    
    def forward_all_hops(self, torch_geo_batch, node_embedding):
        embed_list = self.forward_embed_list(torch_geo_batch, node_embedding)
        return embed_list
    
    def forward_embed_list(self, torch_geo_batch, node_embedding):
        edge_index_tsr = torch_geo_batch.edge_index.to(device())
        embed_list = [node_embedding]
        for li, gnn_layer in enumerate(self.gnn_layers):
            curr_gnn_output = self.bns[li](gnn_layer(embed_list[-1], edge_index_tsr))
            if self.activ is not None:
                curr_gnn_output = self.activ(curr_gnn_output)
            if self.residual and li > 0:
                curr_gnn_output = curr_gnn_output + embed_list[-1]
            embed_list.append(curr_gnn_output)
        return embed_list
