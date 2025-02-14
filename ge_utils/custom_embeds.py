import torch as t
from ge_utils.misc_utils import device
from ge_utils.analysis_utils import report_categorical
from sdm.graph_util import QUANT_METHOD_LUT, BIT_PRECISION_LUT
from sdm.graph_sdv15 import SDV15_LAYERS_LUT
from sdm.graph_pixart import PIXART_TSX_LAYERS_LUT
from sdm.graph_hunyuan import HUNYUAN_TSX_LAYERS_LUT


def discern_custom_family(entry):
    return "mobilenet", entry

class MobileNetNodeEmbedding(t.nn.Module):
    def __init__(self):
        super(MobileNetNodeEmbedding, self).__init__()
        self.stage_embed = t.nn.Embedding(6, 3)
        self.stage_mlp = t.nn.Linear(3, 1)
        self.block_embed = t.nn.Embedding(4, 2)
        self.block_mlp = t.nn.Linear(2, 1)
        self.mbconv_embed = t.nn.Embedding(2, 2)
        self.mbconv_mlp = t.nn.Linear(2, 1)
        self.exp_embed = t.nn.Embedding(3, 2)
        self.exp_mlp = t.nn.Linear(2, 1)
        self.kernel_embed = t.nn.Embedding(3, 3)
        self.kernel_mlp = t.nn.Linear(3, 1)
        self.res_mlp = t.nn.Sequential(t.nn.Linear(1, 3),
                                       t.nn.Linear(3, 1))
        self.rectify = t.abs
        
        self.out_embed_size = 6

    def forward(self, geo_x):
        stages = geo_x[:, 0:1].long()
        blocks = geo_x[:, 1:2].long()
        mbconvs = geo_x[:, 2:3].long()
        exps =   geo_x[:, 3:4].long()
        kernels =   geo_x[:, 4:5].long()
        res = geo_x[:, -1:]

        stage_embedding = self.stage_embed(stages.to(device()))
        stage_embedding = stage_embedding.view(stage_embedding.shape[0], -1)
        stage_embedding = self.stage_mlp(stage_embedding)

        block_embedding = self.block_embed(blocks.to(device()))
        block_embedding = block_embedding.view(block_embedding.shape[0], -1)
        block_embedding = self.block_mlp(block_embedding)

        mbconv_embedding = self.mbconv_embed(mbconvs.to(device()))
        mbconv_embedding = mbconv_embedding.view(mbconv_embedding.shape[0], -1)
        mbconv_embedding = self.mbconv_mlp(mbconv_embedding)

        exp_embedding = self.exp_embed(exps.to(device()))
        exp_embedding = exp_embedding.view(exp_embedding.shape[0], -1)
        exp_embedding = self.exp_mlp(exp_embedding)

        kernel_embedding = self.kernel_embed(kernels.to(device()))
        kernel_embedding = kernel_embedding.view(kernel_embedding.shape[0], -1)
        kernel_embedding = self.kernel_mlp(kernel_embedding)

        res = self.res_mlp(res.to(device()))

        output = t.cat([stage_embedding, block_embedding, 
                            mbconv_embedding, exp_embedding, 
                            kernel_embedding, res], dim=-1)
        return self.rectify(output)
    
    def analyze_node_feats(self):
        print("Running FE-MLP Analysis for MobileNets:")
        self._analyze_stages()
        self._analyze_layers()
        self._analyze_expands()
        self._analyze_kernels()
        self._analyze_resolution()

    def _analyze_stages(self):
        print("     Analyze stages:")
        stage_scores = {}
        with t.no_grad():
            for i in range(self.stage_embed.num_embeddings):
                stage_tensor = t.Tensor([i]).long()
                stage_embed = self.stage_embed(stage_tensor.to(device()))
                stage_embed = stage_embed.view(stage_embed.shape[0], -1)
                mlp_output = self.rectify(self.stage_mlp(stage_embed))
                stage_scores[f"S{i+1}"] = mlp_output.cpu().squeeze().item()
        report_categorical(stage_scores)

    def _analyze_layers(self):
        print("     Analyze layers:")
        layer_scores = {}
        with t.no_grad():
            for i in range(self.block_embed.num_embeddings):
                block_tensor = t.Tensor([i]).long()
                block_embed = self.block_embed(block_tensor.to(device()))
                block_embed = block_embed.view(block_embed.shape[0], -1)
                mlp_output = self.rectify(self.block_mlp(block_embed))
                layer_scores[f"B{i+1}"] = mlp_output.cpu().squeeze().item()
        report_categorical(layer_scores)

    def _analyze_expands(self):
        from constants import MBCONV_EXPAND_MAP
        print("     Analyze expansion ratios:")
        exp_scores = {}
        with t.no_grad():
            for k, v in MBCONV_EXPAND_MAP.items():
                exp_tensor = t.Tensor([v]).long()
                exp_embed = self.exp_embed(exp_tensor.to(device()))
                exp_embed = exp_embed.view(exp_embed.shape[0], -1)
                mlp_output = self.rectify(self.exp_mlp(exp_embed))
                exp_scores[k] = mlp_output.cpu().squeeze().item()
        report_categorical(exp_scores)

    def _analyze_kernels(self):
        from constants import MBCONV_KERNEL_MAP
        print("     Analyze kernel sizes:")
        kernel_scores = {}
        with t.no_grad():
            for k, v in MBCONV_KERNEL_MAP.items():
                kernel_tensor = t.Tensor([v]).long()
                kernel_embed = self.kernel_embed(kernel_tensor.to(device()))
                kernel_embed = kernel_embed.view(kernel_embed.shape[0], -1)
                mlp_output = self.rectify(self.kernel_mlp(kernel_embed))
                kernel_scores[k] = mlp_output.cpu().squeeze().item()
        report_categorical(kernel_scores)

    def _analyze_resolution(self):
        print("     Analyze input resolution")
        input_values = [[192], [208], [224]]
        res_scores = {}
        with t.no_grad():
            for resolution in input_values:
                res_tensor = t.Tensor(resolution)
                res_scores[tuple(resolution)] = self.rectify(self.res_mlp(res_tensor.to(device()))).cpu().squeeze().item()
        report_categorical(res_scores)


# NOTE This is a superclass to handle universal features and FE-MLP functionality
class SDMUniversalNodeEmbedding(t.nn.Module):
    def __init__(self):
        super(SDMUniversalNodeEmbedding, self).__init__()
        # 1. Quant code -> Long Embedding
        self.quant_code_embed = t.nn.Embedding(len(QUANT_METHOD_LUT.keys()) + 1, 5)
        # 2. Bit code -> Long Embedding
        self.bit_prec_embed = t.nn.Embedding(len(BIT_PRECISION_LUT.keys()) + 1, 5)
        # 3. Reduct. % -> Float
        self.reduction_mlp = t.nn.Linear(1, 5)
        # 4. Quant. Err -> Float
        self.quant_error_mlp = t.nn.Linear(1, 5)
        self.out_embed_size = 4
        
    def forward(self, geo_x, apply_act=True):
        q_codes = geo_x[:, 0:1].long()
        b_codes = geo_x[:, 1:2].long()
        reducts = geo_x[:, 2:3]
        errors = geo_x[:, 3:4]

        q_embed = self.quant_code_embed(q_codes.to(device()))
        q_embed = q_embed.view(q_embed.shape[0], -1)

        b_embed = self.bit_prec_embed(b_codes.to(device()))
        b_embed = b_embed.view(b_embed.shape[0], -1)

        r_embed = self.reduction_mlp(reducts.to(device()))
        e_embed = self.quant_error_mlp(errors.to(device()))

        output = t.cat([q_embed, b_embed, r_embed, e_embed], dim=-1)
        return output


class DiTNodeEmbedding(SDMUniversalNodeEmbedding):
    # NOTE its always torch.cat([universal_feats, specific_feats])
    def __init__(self):
        # Super provides features 1-4
        super(DiTNodeEmbedding, self).__init__()
        # 5. Block Code -> Long Embedding
        # NOTE - this is 29 because there are 28 DiT Blocks plus we encode the pos_embed, timestep_embed and proj_out as -1
        self.block_code_embed = t.nn.Embedding(29, 10)
        # 6. Layer code -> Long Embedding
        self.layer_code_embed = t.nn.Embedding(13, 10)

        self.feat_mix = t.nn.Sequential(t.nn.BatchNorm1d(20+20),
                                        t.nn.ReLU(),
                                        t.nn.Linear(20+20, 64),
                                        t.nn.BatchNorm1d(64),
                                        t.nn.ReLU(),
                                        t.nn.Linear(64, 16),
                                        t.nn.BatchNorm1d(16),
                                        t.nn.ReLU()
                                        )
        
        self.out_embed_size = 16

    def forward(self, geo_x):
        universal_feats = super().forward(geo_x, apply_act=False)

        blocks = geo_x[:, 4:5].long()
        layers = geo_x[:, 5:6].long()

        blocks += 1
        # NOTE this +1 is added because we encode some blocks with position -1, however, torch.nn.Embedding expects a minimum input of 0
        bl_embed = self.block_code_embed(blocks.to(device()))
        bl_embed = bl_embed.view(bl_embed.shape[0], -1)
        #bl_embed = self.block_code_mlp(bl_embed)

        la_embed = self.layer_code_embed(layers.to(device()))
        la_embed = la_embed.view(la_embed.shape[0], -1)
        #la_embed = self.layer_code_mlp(la_embed)

        feats = t.cat([universal_feats, bl_embed, la_embed], dim=-1)
        return self.feat_mix(feats)


class SDV15NodeEmbedding(SDMUniversalNodeEmbedding):
    # NOTE its always torch.cat([universal_feats, specific_feats])
    def __init__(self):
        # Super provides features 1-4
        super(SDV15NodeEmbedding, self).__init__()
        # 5. Stage code -> Long embedding
        self.stage_code_embed = t.nn.Embedding(3, 10)
        # 6. Block Code -> Long Embedding
        # NOTE - this is 29 because there are 28 DiT Blocks plus we encode the pos_embed, timestep_embed and proj_out as -1
        self.block_code_embed = t.nn.Embedding(12, 10)
        # 6. Layer code -> Long Embedding
        self.layer_code_embed = t.nn.Embedding(len(SDV15_LAYERS_LUT.keys()) + 1, 10)

        self.feat_mix = t.nn.Sequential(t.nn.BatchNorm1d(20+30),
                                        t.nn.ReLU(),
                                        t.nn.Linear(20+30, 64),
                                        t.nn.BatchNorm1d(64),
                                        t.nn.ReLU(),
                                        t.nn.Linear(64, 32),
                                        t.nn.BatchNorm1d(32),
                                        t.nn.ReLU()
                                        )
        
        self.out_embed_size = 32

    def forward(self, geo_x):
        universal_feats = super().forward(geo_x, apply_act=False)

        stages = geo_x[:, 4:5].long()
        blocks = geo_x[:, 5:6].long()
        layers = geo_x[:, 6:7].long()

        st_embed = self.stage_code_embed(stages.to(device()))
        st_embed = st_embed.view(st_embed.shape[0], -1)

        bl_embed = self.block_code_embed(blocks.to(device()))
        bl_embed = bl_embed.view(bl_embed.shape[0], -1)

        la_embed = self.layer_code_embed(layers.to(device()))
        la_embed = la_embed.view(la_embed.shape[0], -1)

        feats = t.cat([universal_feats, st_embed, bl_embed, la_embed], dim=-1)
        return self.feat_mix(feats)


class SDXLNodeEmbedding(SDMUniversalNodeEmbedding):
    # NOTE its always torch.cat([universal_feats, specific_feats])
    def __init__(self):
        # Super provides features 1-4
        super(SDXLNodeEmbedding, self).__init__()
        # 5. Stage code -> Long embedding
        self.stage_code_embed = t.nn.Embedding(3, 10)
        # 6. Block Code -> Long Embedding
        # NOTE - this is 29 because there are 28 DiT Blocks plus we encode the pos_embed, timestep_embed and proj_out as -1
        self.block_code_embed = t.nn.Embedding(11, 10)
        # 7. Transformer Code -> Long Embedding
        # [-1, 10] -> shift to be [0, 11] with a +1
        self.tf_code_embed = t.nn.Embedding(11, 10)
        # 8. Layer code -> Long Embedding
        self.layer_code_embed = t.nn.Embedding(28, 10)
        
        self.feat_mix = t.nn.Sequential(t.nn.BatchNorm1d(20+40),
                                        t.nn.ReLU(),
                                        t.nn.Linear(20+40, 64),
                                        t.nn.BatchNorm1d(64),
                                        t.nn.ReLU(),
                                        t.nn.Linear(64, 16),
                                        t.nn.BatchNorm1d(16),
                                        t.nn.ReLU()
                                        )
        
        self.out_embed_size = 16

    def forward(self, geo_x):
        universal_feats = super().forward(geo_x, apply_act=False)

        stages = geo_x[:, 4:5].long()
        blocks = geo_x[:, 5:6].long()
        tfs = geo_x[:, 6:7].long()
        layers = geo_x[:, 7:8].long()

        st_embed = self.stage_code_embed(stages.to(device()))
        st_embed = st_embed.view(st_embed.shape[0], -1)

        blocks += 1  # NOTE done for same reason as DiT 
        bl_embed = self.block_code_embed(blocks.to(device()))
        bl_embed = bl_embed.view(bl_embed.shape[0], -1)

        tfs += 1
        tf_embed = self.tf_code_embed(tfs.to(device()))
        tf_embed = tf_embed.view(tf_embed.shape[0], -1)

        la_embed = self.layer_code_embed(layers.to(device()))
        la_embed = la_embed.view(la_embed.shape[0], -1)

        feats = t.cat([universal_feats, st_embed, bl_embed, tf_embed, la_embed], dim=-1)
        return self.feat_mix(feats)


class PixArtNodeEmbedding(SDMUniversalNodeEmbedding):
    # NOTE its always torch.cat([universal_feats, specific_feats])
    def __init__(self):
        # Super provides features 1-4
        super(PixArtNodeEmbedding, self).__init__()
        self.block_code_embed = t.nn.Embedding(29, 10)
        self.layer_code_embed = t.nn.Embedding(len(PIXART_TSX_LAYERS_LUT)+1, 10)

        self.feat_mix = t.nn.Sequential(t.nn.BatchNorm1d(20+20),
                                        t.nn.ReLU(),
                                        t.nn.Linear(20+20, 64),
                                        t.nn.BatchNorm1d(64),
                                        t.nn.ReLU(),
                                        t.nn.Linear(64, 16),
                                        t.nn.BatchNorm1d(16),
                                        t.nn.ReLU()
                                        )
        
        self.out_embed_size = 16

    def forward(self, geo_x):
        universal_feats = super().forward(geo_x, apply_act=False)

        blocks = geo_x[:, 4:5].long()
        layers = geo_x[:, 5:6].long()

        blocks += 1
        # NOTE this +1 is added because we encode some blocks with position -1, however, torch.nn.Embedding expects a minimum input of 0
        bl_embed = self.block_code_embed(blocks.to(device()))
        bl_embed = bl_embed.view(bl_embed.shape[0], -1)

        la_embed = self.layer_code_embed(layers.to(device()))
        la_embed = la_embed.view(la_embed.shape[0], -1)

        feats = t.cat([universal_feats, bl_embed, la_embed], dim=-1)
        return self.feat_mix(feats)
    

class HunYuanNodeEmbedding(SDMUniversalNodeEmbedding):
    # NOTE its always torch.cat([universal_feats, specific_feats])
    def __init__(self):
        # Super provides features 1-4
        super(HunYuanNodeEmbedding, self).__init__()
        self.block_code_embed = t.nn.Embedding(41, 10)
        self.layer_code_embed = t.nn.Embedding(len(HUNYUAN_TSX_LAYERS_LUT)+1, 10)

        self.feat_mix = t.nn.Sequential(t.nn.BatchNorm1d(20+20),
                                        t.nn.ReLU(),
                                        t.nn.Linear(20+20, 64),
                                        t.nn.BatchNorm1d(64),
                                        t.nn.ReLU(),
                                        t.nn.Linear(64, 16),
                                        t.nn.BatchNorm1d(16),
                                        t.nn.ReLU()
                                        )
        
        self.out_embed_size = 16

    def forward(self, geo_x):
        universal_feats = super().forward(geo_x, apply_act=False)

        blocks = geo_x[:, 4:5].long()
        layers = geo_x[:, 5:6].long()

        blocks += 1
        # NOTE this +1 is added because we encode some blocks with position -1, however, torch.nn.Embedding expects a minimum input of 0
        bl_embed = self.block_code_embed(blocks.to(device()))
        bl_embed = bl_embed.view(bl_embed.shape[0], -1)

        la_embed = self.layer_code_embed(layers.to(device()))
        la_embed = la_embed.view(la_embed.shape[0], -1)

        feats = t.cat([universal_feats, bl_embed, la_embed], dim=-1)
        return self.feat_mix(feats)