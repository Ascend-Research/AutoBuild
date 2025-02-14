from ge_utils.components import TGeoNodeEmbedding, TGeoPreEmbeddedGraphEncoder
import torch as t
import torch_geometric as tg
from onnx_ir.op import OPS
from onnx_ir.encoding import MAX_N_SHAPES, NODE_ATTR_FEAT_SIZE
from ge_utils.misc_utils import device


class GraphRegressor(t.nn.Module):

    def __init__(self,
                 embed_layer, encoder, aggregator,
                 hidden_size,
                 activ=None):
        super(GraphRegressor, self).__init__()
        self.embed_layer = embed_layer
        self.encoder = encoder
        self.aggregator = aggregator
        self.activ = activ
        self.post_proj = t.nn.Sequential(
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.ReLU(),
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.Linear(hidden_size, hidden_size),
        )
        self.regressor = t.nn.Linear(hidden_size, 1)

    def forward(self,
                node_inds, node_shapes, node_attrs,
                edge_tsr_list, batch_last_node_inds,
                index=None, ext_feat=None):
        node_embedding = self.embed_layer(node_inds, node_shapes, node_attrs)
        batch_embedding = self.encoder(node_embedding, edge_tsr_list, batch_last_node_inds)
        graph_embedding = self.aggregator(batch_embedding, batch_last_node_inds, index=index)
        graph_embedding = self.post_proj(graph_embedding)
        if self.activ is not None:
            graph_embedding = self.activ(graph_embedding)
        if ext_feat is not None:
            ext_feat = ext_feat.to(device())
            graph_embedding = t.cat([graph_embedding, ext_feat], dim=-1)
        out = self.regressor(graph_embedding)
        return out


class GraphPredictor(GraphRegressor):
    def __init__(self, embed_layer, encoder, aggregator,
                 hidden_size,
                 node_level=False, activ=None):
        super(GraphPredictor, self).__init__(embed_layer,
                                            encoder, aggregator,
                                            hidden_size,
                                            activ=activ)

        if node_level:
            self.forward = self.forward_nodes
        else:
            self.forward = self.forward_graph_embeds

        num_layers = len(self.encoder.gnn_layers) + 1
        self.hop_moments = t.nn.Parameter(t.cat([t.zeros(num_layers).unsqueeze(0), t.ones(num_layers).unsqueeze(0)], dim=0), requires_grad=False)
        self.mlp_moments = t.nn.Parameter(t.Tensor([0., 1.]), requires_grad=False)
        self.test_metrics = t.nn.Parameter(t.Tensor([float("inf"), 100.0, 0.]), requires_grad=False)
        self.hop_srcc = t.nn.Parameter(t.ones(num_layers), requires_grad=False)

    def forward_nodes(self, geo_batch):
        batch_embedding = self.forward_gnn(geo_batch)
        batch_embedding = self.post_proj(batch_embedding)
        if self.activ is not None:
            batch_embedding = self.activ(batch_embedding)
        if ext_feat is not None:
            ext_feat = ext_feat.to(device())
            batch_embedding = t.cat([batch_embedding, ext_feat], dim=-1)
        out = self.regressor(batch_embedding)
        return out

    def forward_graph(self, geo_batch):
        graph_embedding = self.forward_ge(geo_batch)
        post_proj_ge = self.post_proj(graph_embedding)
        if self.activ is not None:
            post_proj_ge = self.activ(post_proj_ge) 
        out = self.regressor(post_proj_ge)
        return out.squeeze(1)
    
    def forward_graph_embeds(self, geo_batch):
        graph_embedding_list = self.forward_ge_all_hops(geo_batch)
        post_proj_ge = self.post_proj(graph_embedding_list[-1])
        if self.activ is not None:
            post_proj_ge = self.activ(post_proj_ge) 
        out = self.regressor(post_proj_ge)
        return out.squeeze(1), graph_embedding_list

    def forward_gnnexplainer(self, x=None, edge_index=None, data=None, **kwargs):
        if data is None:
            data = tg.data.Batch(x=x, edge_index=edge_index)
        if not hasattr(data, 'batch'): 
            data.batch = None
        reg_pred = self.forward_graph(data)
        p = self.dist.cdf(reg_pred)
        converted_logits = t.cat([p, 1-p])
        if len(converted_logits.shape) == 1:
            converted_logits = converted_logits.unsqueeze(0)
        return converted_logits

    def forward_ne(self, geo_batch):
        return self.embed_layer(geo_batch.x)

    def forward_gnn(self, geo_batch):
        node_embedding = self.forward_ne(geo_batch)
        return self.encoder(geo_batch, node_embedding)

    def forward_ge(self, geo_batch):
        batch_as_set = set(geo_batch.batch.tolist())
        batch_embedding = self.forward_gnn(geo_batch)
        geo_batch.batch = geo_batch.batch.to(device())
        graph_embedding = self.aggregator(x=batch_embedding, batch=geo_batch.batch)
        assert graph_embedding.shape[0] == len(batch_as_set)
        return graph_embedding
    
    def forward_ge_all_hops(self, geo_batch):
        node_embedding = self.forward_ne(geo_batch)
        hop_batch_embedding_list = self.encoder.forward_all_hops(geo_batch, node_embedding)
        batch_as_set = set(geo_batch.batch.tolist())
        geo_batch.batch = geo_batch.batch.to(device())
        graph_embedding_list = [self.aggregator(h=hop_batch_embedding_list[i], hop=i, batch=geo_batch.batch)for i in range(len(hop_batch_embedding_list))]
        assert graph_embedding_list[-1].shape[0] == len(batch_as_set)
        return graph_embedding_list

    def get_gnn_node_embeds(self, geo_batch):
        node_embedding = self.embed_layer(geo_batch.x)
        return self.encoder.forward_embed_list(geo_batch, node_embedding)

    def generate_moments(self, dataloader, ord=1):
        print("Generate hop moments")
        node_norm_list = []
        with t.no_grad():
            for data in dataloader:
                if type(data) is dict:
                    data = data['batch_data']
                embed_list = self.get_gnn_node_embeds(data)
                for i, embed_tensor in enumerate(embed_list):
                    embed_norms = t.linalg.norm(embed_tensor, ord=ord, dim=-1, keepdim=False).cpu() #.tolist()
                    if len(node_norm_list) > i:
                        node_norm_list[i] = t.cat([node_norm_list[i], embed_norms]) #+= embed_norms
                    else:
                        node_norm_list.append(embed_norms)
            for hop_level, norm_list in enumerate(node_norm_list):
                sdev, mean = t.std_mean(norm_list)
                print(f"Hop-level {hop_level}: N({mean}, {sdev})")
                self.hop_moments[0, hop_level] = mean
                self.hop_moments[1, hop_level] = sdev

    def dist_shift(self, embedding_norm, stage, hop):
        assert stage > -1
        assert hop > -1
        return self.unstandardize_embedding(self.standardize_embedding(embedding_norm, hop), hop, stage)
    
    def standardize_embedding(self, embedding_norm, hop):
        assert hop > -1
        return (embedding_norm - self.hop_moments[0, hop]) / self.hop_moments[1, hop]
    
    def unstandardize_embedding(self, embedding_norm, hop, stage):
        assert stage > -1
        assert hop > -1
        return (embedding_norm * self.embed_layer.moment_tsr[stage, hop, 1]) + self.embed_layer.moment_tsr[stage, hop, 0]
    
    def forward_denorm(self, ge_data):
        if not hasattr(ge_data, "batch") or ge_data.batch is None:
            ge_data = tg.data.batch.Batch.from_data_list([ge_data])
        return self.forward_graph(ge_data) * self.mlp_moments[1] + self.mlp_moments[0]

    def check_prediction_with_mae(self, ge_data, limit, op="+"):
        prediction = self.forward_denorm(ge_data)
        if op == "+":
            if prediction + self.test_metrics[0] > limit:
                return False
        elif op == "-":
            if prediction - self.test_metrics[0] < limit:
                return False
        return True
    
    def assign_new_moment_tsr(self, new_mt):
        self.embed_layer.moment_tsr = t.nn.Parameter(new_mt, requires_grad=False)
    

def make_predictor(node_level=False,
                   op_embed_size=8,
                   shape_embed_size=8,
                   attr_embed_size=8,
                   gnn_dim=128,
                   gnn_type="GraphConv",
                   format="custom",
                   families="ofa",
                   n_ops=len(OPS),
                   input_shape_size=2 * MAX_N_SHAPES * 3,
                   input_attr_size=NODE_ATTR_FEAT_SIZE,
                   gnn_activ="ReLU",
                   num_layers=4, dropout_prob=0.0,
                   reg_activ="ReLU", aggr_method="mean",
                   residual=False,
                   gnn_chkpt=None,
                   fe_mlp=False,
                   **kwargs):

    gnn_activ = eval(f"t.nn.{gnn_activ}()")
    reg_activ = eval(f"t.nn.{reg_activ}()")

    if format == "onnx_ir":
        embed_layer = TGeoNodeEmbedding(n_ops=n_ops,
                                    input_shape_size=input_shape_size,
                                    input_attr_size=input_attr_size,
                                    op_embed_size=op_embed_size,
                                    shape_embed_size=shape_embed_size,
                                    attr_embed_size=attr_embed_size)
    else:
        embed_layer = _make_custom_node_embed_layer(families, fe_mlp)

    gnn_constructor = eval(f"tg.nn.{gnn_type}")
    encoder = TGeoPreEmbeddedGraphEncoder(embed_layer.out_embed_size, gnn_dim, gnn_dim,
                                        gnn_constructor,
                                        activ=gnn_activ,
                                        n_layers=num_layers,
                                        dropout_prob=dropout_prob,
                                        residual=residual)
    if aggr_method == "mean":
        aggregator = tg.nn.glob.global_mean_pool
    elif aggr_method == "sum":
        aggregator = tg.nn.glob.global_add_pool
    else:
        raise NotImplementedError(f"Unknown aggregation method {aggr_method}")
    from ge_utils.custom_aggr import SELECTIVE_AGGR
    aggregator = SELECTIVE_AGGR(tg_agg=aggregator, family=families, arith=aggr_method)
    
    predictor = GraphPredictor(embed_layer, encoder, aggregator, gnn_dim,
                               node_level=node_level, activ=reg_activ)

    if gnn_chkpt is not None:
        gnn_weights = t.load(gnn_chkpt, map_location=device())
        predictor.load_state_dict(gnn_weights, strict=False)
    predictor = predictor.to(device())
    return predictor


def  _make_custom_node_embed_layer(families, fe_mlp):
    if fe_mlp:
        print("[WARNING!!!!] YOU HAVE NOT DONE BATCHNORM1D FOR FE-MLP!!!!!!")
    if "ofa" in families:
        from ge_utils.custom_embeds_femlp import MobileNetNodeEmbedding as NodeEmbedding
        if "mbv3" in families:
            empty_moment_tsr = t.cat([t.zeros(5, 4, 1), t.ones(5, 4, 1)], dim=-1)
        else:
            empty_moment_tsr = t.cat([t.zeros(5, 5, 1), t.ones(5, 5, 1)], dim=-1)
    elif "dit" in families:
        if fe_mlp:
            from ge_utils.custom_embeds_femlp import DiTNodeEmbedding as NodeEmbedding
        else:
            from ge_utils.custom_embeds import DiTNodeEmbedding as NodeEmbedding
    elif "sdv15" in families:
        if fe_mlp:
            from ge_utils.custom_embeds_femlp import SDV15NodeEmbedding as NodeEmbedding
        else:
            from ge_utils.custom_embeds import SDV15NodeEmbedding as NodeEmbedding
    elif "sdxl" in families:
        if fe_mlp:
            from ge_utils.custom_embeds_femlp import SDXLNodeEmbedding as NodeEmbedding
        else:
            from ge_utils.custom_embeds import SDXLNodeEmbedding as NodeEmbedding
    elif "alpha" in families or "sigma" in families:
        from ge_utils.custom_embeds import PixArtNodeEmbedding as NodeEmbedding
    elif "hunyuan" in families:
        from ge_utils.custom_embeds import HunYuanNodeEmbedding as NodeEmbedding
    else:
        raise NotImplementedError(f"No custom embedding for family {families}")
    ne = NodeEmbedding()
    return ne
    
