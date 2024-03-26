import torch as t
from ge_utils.misc_utils import device
from ge_utils.analysis_utils import report_categorical


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
