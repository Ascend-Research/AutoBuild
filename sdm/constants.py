from collections import OrderedDict
from sdm.graph_dit import DIT_TSX_LAYERS_LUT
from sdm.graph_sdv15 import SDV15_LAYERS_LUT
from sdm.graph_sdxl import SDXL_LAYERS_LUT

QM_BIT_ORDER = OrderedDict()
QM_BIT_ORDER['kmeans-4'] = "K-Means C 4-bit"
QM_BIT_ORDER['kmeans_all-4'] = "K-Means A 4-bit"
QM_BIT_ORDER['mse-4'] = "UAQ 4-bit"
QM_BIT_ORDER['kmeans-3'] = "K-Means C 3-bit"
QM_BIT_ORDER['kmeans_all-3'] = "K-Means A 3-bit"
QM_BIT_ORDER['mse-3'] = "UAQ 3-bit"

QM_COLORS = {
    "K-Means C 4-bit": 'darkturquoise',
    "K-Means C 3-bit": 'darkcyan',
    "K-Means A 4-bit": 'lightgreen',
    "K-Means A 3-bit": 'darkolivegreen',
    "UAQ 4-bit": 'coral',
    "UAQ 3-bit": "firebrick"
}


DIT_BLOCK_TYPES = ["*0.norm1.emb.timestep_embedder*", *[f"model.transformer_blocks.{i}.*" for i in range(28)], "model.proj_out*"]

DIT_BLOCK_TYPES = OrderedDict()
DIT_BLOCK_TYPES["t-Emb"] = "*0.norm1.emb.timestep_embedder*"
DIT_BLOCK_TYPES["0"] = "model.transformer_blocks.0.*"
DIT_BLOCK_TYPES["1"] = "model.transformer_blocks.1.*"
DIT_BLOCK_TYPES["2"] = "model.transformer_blocks.2.*"
DIT_BLOCK_TYPES["3"] = "model.transformer_blocks.3.*"
DIT_BLOCK_TYPES["4"] = "model.transformer_blocks.4.*"
DIT_BLOCK_TYPES["5"] = "model.transformer_blocks.5.*"
DIT_BLOCK_TYPES["6"] = "model.transformer_blocks.6.*"
DIT_BLOCK_TYPES["7"] = "model.transformer_blocks.7.*"
DIT_BLOCK_TYPES["8"] = "model.transformer_blocks.8.*"
DIT_BLOCK_TYPES["9"] = "model.transformer_blocks.9.*"
DIT_BLOCK_TYPES["10"] = "model.transformer_blocks.10.*"
DIT_BLOCK_TYPES["11"] = "model.transformer_blocks.11.*"
DIT_BLOCK_TYPES["12"] = "model.transformer_blocks.12.*"
DIT_BLOCK_TYPES["13"] = "model.transformer_blocks.13.*"
DIT_BLOCK_TYPES["14"] = "model.transformer_blocks.14.*"
DIT_BLOCK_TYPES["15"] = "model.transformer_blocks.15.*"
DIT_BLOCK_TYPES["16"] = "model.transformer_blocks.16.*"
DIT_BLOCK_TYPES["17"] = "model.transformer_blocks.17.*"
DIT_BLOCK_TYPES["18"] = "model.transformer_blocks.18.*"
DIT_BLOCK_TYPES["19"] = "model.transformer_blocks.19.*"
DIT_BLOCK_TYPES["20"] = "model.transformer_blocks.20.*"
DIT_BLOCK_TYPES["21"] = "model.transformer_blocks.21.*"
DIT_BLOCK_TYPES["22"] = "model.transformer_blocks.22.*"
DIT_BLOCK_TYPES["23"] = "model.transformer_blocks.23.*"
DIT_BLOCK_TYPES["24"] = "model.transformer_blocks.24.*"
DIT_BLOCK_TYPES["25"] = "model.transformer_blocks.25.*"
DIT_BLOCK_TYPES["26"] = "model.transformer_blocks.26.*"
DIT_BLOCK_TYPES["27"] = "model.transformer_blocks.27.*"
DIT_BLOCK_TYPES["Out Proj."] = "model.proj_out*"


DIT_LAYER_TYPES = OrderedDict()
#DIT_LAYER_TYPES["t 1"] = "model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_1*"
#DIT_LAYER_TYPES["t 2"] = "model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_2*"
DIT_LAYER_TYPES["t-Embed"] = "model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_*"
DIT_LAYER_TYPES["Patchify"] = "*pos_embed.proj*"
DIT_LAYER_TYPES["AdaLN"] = "*norm1.linear*"
DIT_LAYER_TYPES["SA-Q"] = "*attn1.to_q*"
DIT_LAYER_TYPES["SA-K"] = "*attn1.to_k*"
DIT_LAYER_TYPES["SA-V"] = "*attn1.to_v*"
DIT_LAYER_TYPES["SA-Out"] = "*attn1.to_out.0*"
DIT_LAYER_TYPES["FF 1"] = "*ff.net.0.proj*"
DIT_LAYER_TYPES["FF 2"] = "*ff.net.2*"
#DIT_LAYER_TYPES["Out 1"] = "*proj_out_1*"
#DIT_LAYER_TYPES["Out 2"] = "*proj_out_2*"
DIT_LAYER_TYPES["Out Proj."] = "*proj_out_*"


DIT_SUBGRAPHS_TYPES = OrderedDict()
DIT_SUBGRAPHS_TYPES["t-Embed"] = "time_embedding"
DIT_SUBGRAPHS_TYPES["SA"] = "*attn*"
DIT_SUBGRAPHS_TYPES["Feedforward"] = "*ff*"
DIT_SUBGRAPHS_TYPES["Out Proj."] = "*proj_out*"

DIT_SUBGRAPHS = OrderedDict()
DIT_SUBGRAPHS['t-Emb'] = "time_embedding"
DIT_SUBGRAPHS['0'] = "dit_blk_0_*"
DIT_SUBGRAPHS['1'] = "dit_blk_1_*"
DIT_SUBGRAPHS['2'] = "dit_blk_2_*"
DIT_SUBGRAPHS['3'] = "dit_blk_3_*"
DIT_SUBGRAPHS['4'] = "dit_blk_4_*"
DIT_SUBGRAPHS['5'] = "dit_blk_5_*"
DIT_SUBGRAPHS['6'] = "dit_blk_6_*"
DIT_SUBGRAPHS['7'] = "dit_blk_7_*"
DIT_SUBGRAPHS['8'] = "dit_blk_8_*"
DIT_SUBGRAPHS['9'] = "dit_blk_9_*"
DIT_SUBGRAPHS['10'] = "dit_blk_10_*"
DIT_SUBGRAPHS['11'] = "dit_blk_11_*"
DIT_SUBGRAPHS['12'] = "dit_blk_12_*"
DIT_SUBGRAPHS['13'] = "dit_blk_13_*"
DIT_SUBGRAPHS['14'] = "dit_blk_14_*"
DIT_SUBGRAPHS['15'] = "dit_blk_15_*"
DIT_SUBGRAPHS['16'] = "dit_blk_16_*"
DIT_SUBGRAPHS['17'] = "dit_blk_17_*"
DIT_SUBGRAPHS['18'] = "dit_blk_18_*"
DIT_SUBGRAPHS['19'] = "dit_blk_19_*"
DIT_SUBGRAPHS['20'] = "dit_blk_20_*"
DIT_SUBGRAPHS['21'] = "dit_blk_21_*"
DIT_SUBGRAPHS['22'] = "dit_blk_22_*"
DIT_SUBGRAPHS['23'] = "dit_blk_23_*"
DIT_SUBGRAPHS['24'] = "dit_blk_24_*"
DIT_SUBGRAPHS['25'] = "dit_blk_25_*"
DIT_SUBGRAPHS['26'] = "dit_blk_26_*"
DIT_SUBGRAPHS['27'] = "dit_blk_27_*"
DIT_SUBGRAPHS['Out Proj.'] = 'proj_out'


PIXART_LAYER_TYPES = OrderedDict()
#PIXART_LAYER_TYPES["t 1"] = "model.adaln_single.emb.timestep_embedder.linear_1*"
#PIXART_LAYER_TYPES["t 2"] = "model.adaln_single.emb.timestep_embedder.linear_2*"
PIXART_LAYER_TYPES["t-Embed"] = "model.adaln_single.*"
#PIXART_LAYER_TYPES["c 1"] = "model.caption_projection.linear_1.*"
#PIXART_LAYER_TYPES["c 2"] = "model.caption_projection.linear_2.*"
PIXART_LAYER_TYPES["c-Embed"] = "model.caption_projection.*"
PIXART_LAYER_TYPES["Patchify"] = "*pos_embed.proj*"
PIXART_LAYER_TYPES["AdaLN"] = "model.adaln_single.linear*"
PIXART_LAYER_TYPES["SA-Q"] = "*attn1.to_q*"
PIXART_LAYER_TYPES["SA-K"] = "*attn1.to_k*"
PIXART_LAYER_TYPES["SA-V"] = "*attn1.to_v*"
PIXART_LAYER_TYPES["SA-Out"] = "*attn1.to_out.0*"
PIXART_LAYER_TYPES["CA-Q"] = "*attn2.to_q*"
PIXART_LAYER_TYPES["CA-K"] = "*attn2.to_k*"
PIXART_LAYER_TYPES["CA-V"] = "*attn2.to_v*"
PIXART_LAYER_TYPES["CA-Out"] = "*attn2.to_out.0*"
PIXART_LAYER_TYPES["FF 1"] = "*ff.net.0.proj*"
PIXART_LAYER_TYPES["FF 2"] = "*ff.net.2*"
PIXART_LAYER_TYPES["Out"] = "model.proj_out.*"

PIXART_SUBGRAPHS_TYPES = OrderedDict()
PIXART_SUBGRAPHS_TYPES["t-Embed"] = "*time_embedding*"
PIXART_SUBGRAPHS_TYPES["c-Embed"] = "*caption_embedding*"
PIXART_SUBGRAPHS_TYPES["Patchify"] = "*pos_embed*"
PIXART_SUBGRAPHS_TYPES["Self-Attn"] = "*selfattn*"
PIXART_SUBGRAPHS_TYPES["Cross-Attn"] = "*crossattn*"
PIXART_SUBGRAPHS_TYPES["Feedforward"] = "*ff*"
PIXART_SUBGRAPHS_TYPES["Out Proj."] = "*proj_out*"

# TODO I forget this bit...
PIXART_SUBGRAPHS = OrderedDict()
PIXART_SUBGRAPHS['t-Emb'] = "time_embedding"
PIXART_SUBGRAPHS['c-Emb'] = "caption_embedding"
PIXART_SUBGRAPHS['0'] = "pixart_blk_0_*"
PIXART_SUBGRAPHS['1'] = "pixart_blk_1_*"
PIXART_SUBGRAPHS['2'] = "pixart_blk_2_*"
PIXART_SUBGRAPHS['3'] = "pixart_blk_3_*"
PIXART_SUBGRAPHS['4'] = "pixart_blk_4_*"
PIXART_SUBGRAPHS['5'] = "pixart_blk_5_*"
PIXART_SUBGRAPHS['6'] = "pixart_blk_6_*"
PIXART_SUBGRAPHS['7'] = "pixart_blk_7_*"
PIXART_SUBGRAPHS['8'] = "pixart_blk_8_*"
PIXART_SUBGRAPHS['9'] = "pixart_blk_9_*"
PIXART_SUBGRAPHS['10'] = "pixart_blk_10_*"
PIXART_SUBGRAPHS['11'] = "pixart_blk_11_*"
PIXART_SUBGRAPHS['12'] = "pixart_blk_12_*"
PIXART_SUBGRAPHS['13'] = "pixart_blk_13_*"
PIXART_SUBGRAPHS['14'] = "pixart_blk_14_*"
PIXART_SUBGRAPHS['15'] = "pixart_blk_15_*"
PIXART_SUBGRAPHS['16'] = "pixart_blk_16_*"
PIXART_SUBGRAPHS['17'] = "pixart_blk_17_*"
PIXART_SUBGRAPHS['18'] = "pixart_blk_18_*"
PIXART_SUBGRAPHS['19'] = "pixart_blk_19_*"
PIXART_SUBGRAPHS['20'] = "pixart_blk_20_*"
PIXART_SUBGRAPHS['21'] = "pixart_blk_21_*"
PIXART_SUBGRAPHS['22'] = "pixart_blk_22_*"
PIXART_SUBGRAPHS['23'] = "pixart_blk_23_*"
PIXART_SUBGRAPHS['24'] = "pixart_blk_24_*"
PIXART_SUBGRAPHS['25'] = "pixart_blk_25_*"
PIXART_SUBGRAPHS['26'] = "pixart_blk_26_*"
PIXART_SUBGRAPHS['27'] = "pixart_blk_27_*"
PIXART_SUBGRAPHS['Out Proj.'] = 'proj_out'

PIXART_BLOCK_TYPES = OrderedDict()
PIXART_BLOCK_TYPES["t-Emb"] = "*adaln_single*"
PIXART_BLOCK_TYPES["c-Emb"] = "model.caption_projection*"
PIXART_BLOCK_TYPES["0"] = "model.transformer_blocks.0.*"
PIXART_BLOCK_TYPES["1"] = "model.transformer_blocks.1.*"
PIXART_BLOCK_TYPES["2"] = "model.transformer_blocks.2.*"
PIXART_BLOCK_TYPES["3"] = "model.transformer_blocks.3.*"
PIXART_BLOCK_TYPES["4"] = "model.transformer_blocks.4.*"
PIXART_BLOCK_TYPES["5"] = "model.transformer_blocks.5.*"
PIXART_BLOCK_TYPES["6"] = "model.transformer_blocks.6.*"
PIXART_BLOCK_TYPES["7"] = "model.transformer_blocks.7.*"
PIXART_BLOCK_TYPES["8"] = "model.transformer_blocks.8.*"
PIXART_BLOCK_TYPES["9"] = "model.transformer_blocks.9.*"
PIXART_BLOCK_TYPES["10"] = "model.transformer_blocks.10.*"
PIXART_BLOCK_TYPES["11"] = "model.transformer_blocks.11.*"
PIXART_BLOCK_TYPES["12"] = "model.transformer_blocks.12.*"
PIXART_BLOCK_TYPES["13"] = "model.transformer_blocks.13.*"
PIXART_BLOCK_TYPES["14"] = "model.transformer_blocks.14.*"
PIXART_BLOCK_TYPES["15"] = "model.transformer_blocks.15.*"
PIXART_BLOCK_TYPES["16"] = "model.transformer_blocks.16.*"
PIXART_BLOCK_TYPES["17"] = "model.transformer_blocks.17.*"
PIXART_BLOCK_TYPES["18"] = "model.transformer_blocks.18.*"
PIXART_BLOCK_TYPES["19"] = "model.transformer_blocks.19.*"
PIXART_BLOCK_TYPES["20"] = "model.transformer_blocks.20.*"
PIXART_BLOCK_TYPES["21"] = "model.transformer_blocks.21.*"
PIXART_BLOCK_TYPES["22"] = "model.transformer_blocks.22.*"
PIXART_BLOCK_TYPES["23"] = "model.transformer_blocks.23.*"
PIXART_BLOCK_TYPES["24"] = "model.transformer_blocks.24.*"
PIXART_BLOCK_TYPES["25"] = "model.transformer_blocks.25.*"
PIXART_BLOCK_TYPES["26"] = "model.transformer_blocks.26.*"
PIXART_BLOCK_TYPES["27"] = "model.transformer_blocks.27.*"
PIXART_BLOCK_TYPES["Out Proj."] = "model.proj_out*"


HUNYUAN_LAYER_TYPES = OrderedDict()
#HUNYUAN_LAYER_TYPES["t 1"] = "model.time_extra_emb.timestep_embedder.linear_1*"
#HUNYUAN_LAYER_TYPES["t 2"] = "model.time_extra_emb.timestep_embedder.linear_2*"
#HUNYUAN_LAYER_TYPES["e 1"] = "model.time_extra_emb.extra_embedder.linear_1*"
#HUNYUAN_LAYER_TYPES["e 2"] = "model.time_extra_emb.extra_embedder.linear_2*"
HUNYUAN_LAYER_TYPES['t-Embed'] = 'model.time_extra_emb.*'
#HUNYUAN_LAYER_TYPES["c 1"] = "model.text_embedder.linear_1.*"
#HUNYUAN_LAYER_TYPES["c 2"] = "model.text_embedder.linear_2.*"
HUNYUAN_LAYER_TYPES["c-Embed"] = "model.text_embedder.*"
HUNYUAN_LAYER_TYPES["Patchify"] = "*pos_embed.proj*"
HUNYUAN_LAYER_TYPES["AdaLN"] = "*norm1.linear*"
HUNYUAN_LAYER_TYPES["SA-Q"] = "*attn1.to_q*"
HUNYUAN_LAYER_TYPES["SA-K"] = "*attn1.to_k*"
HUNYUAN_LAYER_TYPES["SA-V"] = "*attn1.to_v*"
HUNYUAN_LAYER_TYPES["SA-Out"] = "*attn1.to_out.0*"
HUNYUAN_LAYER_TYPES["CA-Q"] = "*attn2.to_q*"
HUNYUAN_LAYER_TYPES["CA-K"] = "*attn2.to_k*"
HUNYUAN_LAYER_TYPES["CA-V"] = "*attn2.to_v*"
HUNYUAN_LAYER_TYPES["CA-Out"] = "*attn2.to_out.0*"
HUNYUAN_LAYER_TYPES["FF 1"] = "*ff.net.0.proj*"
HUNYUAN_LAYER_TYPES["FF 2"] = "*ff.net.2*"
HUNYUAN_LAYER_TYPES["Res Skip"] = "*skip*"
HUNYUAN_LAYER_TYPES["Norm"] = "model.norm_out.*"
HUNYUAN_LAYER_TYPES["Out"] = "model.proj_out.*"


HUNYUAN_SUBGRAPHS_TYPES = OrderedDict()
HUNYUAN_SUBGRAPHS_TYPES["t-Embed"] = "time_embedding"
HUNYUAN_SUBGRAPHS_TYPES["c-Embed"] = "text_embedder"
HUNYUAN_SUBGRAPHS_TYPES["Patchify"] = "*pos_embed*"
HUNYUAN_SUBGRAPHS_TYPES["Self-Attn"] = "*selfattn*"
HUNYUAN_SUBGRAPHS_TYPES["Cross-Attn"] = "*crossattn*"
HUNYUAN_SUBGRAPHS_TYPES["Skip"] = "*skip*"
HUNYUAN_SUBGRAPHS_TYPES["Feedforward"] = "*ff*"
HUNYUAN_SUBGRAPHS_TYPES["Out Proj."] = "*proj_out*"

# TODO I forget this bit...
HUNYUAN_SUBGRAPHS = OrderedDict()
HUNYUAN_SUBGRAPHS['t-Emb'] = "time_embedding"
HUNYUAN_SUBGRAPHS['c-Emb'] = "text_embedder"
HUNYUAN_SUBGRAPHS['0'] = "hunyuan_blk_0_*"
HUNYUAN_SUBGRAPHS['1'] = "hunyuan_blk_1_*"
HUNYUAN_SUBGRAPHS['2'] = "hunyuan_blk_2_*"
HUNYUAN_SUBGRAPHS['3'] = "hunyuan_blk_3_*"
HUNYUAN_SUBGRAPHS['4'] = "hunyuan_blk_4_*"
HUNYUAN_SUBGRAPHS['5'] = "hunyuan_blk_5_*"
HUNYUAN_SUBGRAPHS['6'] = "hunyuan_blk_6_*"
HUNYUAN_SUBGRAPHS['7'] = "hunyuan_blk_7_*"
HUNYUAN_SUBGRAPHS['8'] = "hunyuan_blk_8_*"
HUNYUAN_SUBGRAPHS['9'] = "hunyuan_blk_9_*"
HUNYUAN_SUBGRAPHS['10'] = "hunyuan_blk_10_*"
HUNYUAN_SUBGRAPHS['11'] = "hunyuan_blk_11_*"
HUNYUAN_SUBGRAPHS['12'] = "hunyuan_blk_12_*"
HUNYUAN_SUBGRAPHS['13'] = "hunyuan_blk_13_*"
HUNYUAN_SUBGRAPHS['14'] = "hunyuan_blk_14_*"
HUNYUAN_SUBGRAPHS['15'] = "hunyuan_blk_15_*"
HUNYUAN_SUBGRAPHS['16'] = "hunyuan_blk_16_*"
HUNYUAN_SUBGRAPHS['17'] = "hunyuan_blk_17_*"
HUNYUAN_SUBGRAPHS['18'] = "hunyuan_blk_18_*"
HUNYUAN_SUBGRAPHS['19'] = "hunyuan_blk_19_*"
HUNYUAN_SUBGRAPHS['20'] = "hunyuan_blk_20_*"
HUNYUAN_SUBGRAPHS['21'] = "hunyuan_blk_21_*"
HUNYUAN_SUBGRAPHS['22'] = "hunyuan_blk_22_*"
HUNYUAN_SUBGRAPHS['23'] = "hunyuan_blk_23_*"
HUNYUAN_SUBGRAPHS['24'] = "hunyuan_blk_24_*"
HUNYUAN_SUBGRAPHS['25'] = "hunyuan_blk_25_*"
HUNYUAN_SUBGRAPHS['26'] = "hunyuan_blk_26_*"
HUNYUAN_SUBGRAPHS['27'] = "hunyuan_blk_27_*"
HUNYUAN_SUBGRAPHS['28'] = "hunyuan_blk_28_*"
HUNYUAN_SUBGRAPHS['29'] = "hunyuan_blk_29_*"
HUNYUAN_SUBGRAPHS['30'] = "hunyuan_blk_30_*"
HUNYUAN_SUBGRAPHS['31'] = "hunyuan_blk_31_*"
HUNYUAN_SUBGRAPHS['32'] = "hunyuan_blk_32_*"
HUNYUAN_SUBGRAPHS['33'] = "hunyuan_blk_33_*"
HUNYUAN_SUBGRAPHS['34'] = "hunyuan_blk_34_*"
HUNYUAN_SUBGRAPHS['35'] = "hunyuan_blk_35_*"
HUNYUAN_SUBGRAPHS['36'] = "hunyuan_blk_36_*"
HUNYUAN_SUBGRAPHS['37'] = "hunyuan_blk_37_*"
HUNYUAN_SUBGRAPHS['38'] = "hunyuan_blk_38_*"
HUNYUAN_SUBGRAPHS['39'] = "hunyuan_blk_39_*"
HUNYUAN_SUBGRAPHS['Out Proj.'] = 'proj_out'

HUNYUAN_BLOCK_TYPES = OrderedDict()
HUNYUAN_BLOCK_TYPES["t-Emb"] = "model.time_extra_emb*"
HUNYUAN_BLOCK_TYPES["c-Emb"] = "model.text_embedder*"
HUNYUAN_BLOCK_TYPES["0"] = "model.blocks.0.*"
HUNYUAN_BLOCK_TYPES["1"] = "model.blocks.1.*"
HUNYUAN_BLOCK_TYPES["2"] = "model.blocks.2.*"
HUNYUAN_BLOCK_TYPES["3"] = "model.blocks.3.*"
HUNYUAN_BLOCK_TYPES["4"] = "model.blocks.4.*"
HUNYUAN_BLOCK_TYPES["5"] = "model.blocks.5.*"
HUNYUAN_BLOCK_TYPES["6"] = "model.blocks.6.*"
HUNYUAN_BLOCK_TYPES["7"] = "model.blocks.7.*"
HUNYUAN_BLOCK_TYPES["8"] = "model.blocks.8.*"
HUNYUAN_BLOCK_TYPES["9"] = "model.blocks.9.*"
HUNYUAN_BLOCK_TYPES["10"] = "model.blocks.10.*"
HUNYUAN_BLOCK_TYPES["11"] = "model.blocks.11.*"
HUNYUAN_BLOCK_TYPES["12"] = "model.blocks.12.*"
HUNYUAN_BLOCK_TYPES["13"] = "model.blocks.13.*"
HUNYUAN_BLOCK_TYPES["14"] = "model.blocks.14.*"
HUNYUAN_BLOCK_TYPES["15"] = "model.blocks.15.*"
HUNYUAN_BLOCK_TYPES["16"] = "model.blocks.16.*"
HUNYUAN_BLOCK_TYPES["17"] = "model.blocks.17.*"
HUNYUAN_BLOCK_TYPES["18"] = "model.blocks.18.*"
HUNYUAN_BLOCK_TYPES["19"] = "model.blocks.19.*"
HUNYUAN_BLOCK_TYPES["20"] = "model.blocks.20.*"
HUNYUAN_BLOCK_TYPES["21"] = "model.blocks.21.*"
HUNYUAN_BLOCK_TYPES["22"] = "model.blocks.22.*"
HUNYUAN_BLOCK_TYPES["23"] = "model.blocks.23.*"
HUNYUAN_BLOCK_TYPES["24"] = "model.blocks.24.*"
HUNYUAN_BLOCK_TYPES["25"] = "model.blocks.25.*"
HUNYUAN_BLOCK_TYPES["26"] = "model.blocks.26.*"
HUNYUAN_BLOCK_TYPES["27"] = "model.blocks.27.*"
HUNYUAN_BLOCK_TYPES["28"] = "model.blocks.28.*"
HUNYUAN_BLOCK_TYPES["29"] = "model.blocks.29.*"
HUNYUAN_BLOCK_TYPES["30"] = "model.blocks.30.*"
HUNYUAN_BLOCK_TYPES["31"] = "model.blocks.31.*"
HUNYUAN_BLOCK_TYPES["32"] = "model.blocks.32.*"
HUNYUAN_BLOCK_TYPES["33"] = "model.blocks.33.*"
HUNYUAN_BLOCK_TYPES["34"] = "model.blocks.34.*"
HUNYUAN_BLOCK_TYPES["35"] = "model.blocks.35.*"
HUNYUAN_BLOCK_TYPES["36"] = "model.blocks.36.*"
HUNYUAN_BLOCK_TYPES["37"] = "model.blocks.37.*"
HUNYUAN_BLOCK_TYPES["38"] = "model.blocks.38.*"
HUNYUAN_BLOCK_TYPES["39"] = "model.blocks.39.*"
HUNYUAN_BLOCK_TYPES["Out Proj."] = "model.proj_out*"


SDV15_BLOCK_TYPES = OrderedDict()
SDV15_BLOCK_TYPES["t-Embed"] = 'model.time_embed*'
SDV15_BLOCK_TYPES["Conv In"] ='model.input_blocks.0.0.*'
SDV15_BLOCK_TYPES["Input ResBlk"] = "model.input_blocks.*.0.*"
SDV15_BLOCK_TYPES["Downsample"] = '*0.op.*'
SDV15_BLOCK_TYPES["Input SA"] = "model.input_blocks.*.1.transformer_blocks.0.attn1*"
SDV15_BLOCK_TYPES["Input CA"] = "model.input_blocks.*.1.transformer_blocks.0.attn2*"
SDV15_BLOCK_TYPES["Input FF"] = "model.input_blocks.*.1.transformer_blocks.0.ff*"
SDV15_BLOCK_TYPES["Mid Res 1"] = 'model.middle_block.0.*'
SDV15_BLOCK_TYPES["Mid SA"] = 'model.middle_block.1.transformer_blocks.0.attn1*'
SDV15_BLOCK_TYPES["Mid CA"] = 'model.middle_block.1.transformer_blocks.0.attn2*'
SDV15_BLOCK_TYPES["Mid Res 2"] = 'model.middle_block.2.*'
SDV15_BLOCK_TYPES["Output ResBlk"] = "model.output_blocks.*.0.*"
SDV15_BLOCK_TYPES["Upsample"] = "*conv.weight_quantizer"
SDV15_BLOCK_TYPES["Output SA"] = "model.output_blocks.*.1.transformer_blocks.0.attn1*"
SDV15_BLOCK_TYPES["Output CA"] = "model.output_blocks.*.1.transformer_blocks.0.attn2*"
SDV15_BLOCK_TYPES["Output FF"] = "model.output_blocks.*.1.transformer_blocks.0.ff*"
SDV15_BLOCK_TYPES["Conv Out"] = 'model.out.2.weight_quantizer'

SDV15_LAYER_TYPES = [f"*{k}.weight_quantizer" for k, v in SDV15_LAYERS_LUT.items() if v > 0]

SDV15_LAYER_TYPES = OrderedDict()
#SDV15_LAYER_TYPES["t 1"] = "*time_embed.0*"
#SDV15_LAYER_TYPES["t 2"] = "*time_embed.2*"
SDV15_LAYER_TYPES["t-Embed"] = "*time_embed.*"
SDV15_LAYER_TYPES["Conv In"] = "*0.0*"
#SDV15_LAYER_TYPES["Downsample"] = "*op*"
SDV15_LAYER_TYPES["Res In"] = "*in_layers.2*"
SDV15_LAYER_TYPES["Res t-Emb"] = "*emb_layers.1*"
SDV15_LAYER_TYPES["Res Out"] = "*out_layers.3*"
SDV15_LAYER_TYPES["Res Skip"] = "*0.skip_connection*"
SDV15_LAYER_TYPES["Proj In"] = "*1.proj_in*"
SDV15_LAYER_TYPES["SA-Q"] = "*attn1.to_q*"
SDV15_LAYER_TYPES["SA-K"] = "*attn1.to_k*"
SDV15_LAYER_TYPES["SA-V"] = "*attn1.to_v*"
SDV15_LAYER_TYPES["SA-Out"] = "*attn1.to_out.0*"
SDV15_LAYER_TYPES["CA-Q"] = "*attn2.to_q*"
SDV15_LAYER_TYPES["CA-K"] = "*attn2.to_k*"
SDV15_LAYER_TYPES["CA-V"] = "*attn2.to_v*"
SDV15_LAYER_TYPES["CA-Out"] = "*attn2.to_out.0*"
SDV15_LAYER_TYPES["FF 1"] = "*ff.net.0.proj*"
SDV15_LAYER_TYPES["FF 2"] = "*ff.net.2*"
SDV15_LAYER_TYPES["Proj Out"] = "*1.proj_out*"
#SDV15_LAYER_TYPES["Upsample"] = "*conv*"
SDV15_LAYER_TYPES["Conv Out"] = "*out.2*"

SDV15_SUBGRAPH_TYPES = OrderedDict()
SDV15_SUBGRAPH_TYPES['t-Embed'] = '*time_embedding*'
SDV15_SUBGRAPH_TYPES['Conv In'] = '*conv_in*'
SDV15_SUBGRAPH_TYPES['Downsample'] = '*op9*'
SDV15_SUBGRAPH_TYPES['ResBlk'] = '*resblk*'
SDV15_SUBGRAPH_TYPES['SA'] = '*attn1*'
SDV15_SUBGRAPH_TYPES['CA'] = '*attn2*'
SDV15_SUBGRAPH_TYPES['FF'] = '*tf_ff*'
#SDV15_SUBGRAPH_TYPES['TF Proj Ou'] = '*tf_proj_out*' # I am going to skip-over this one...
SDV15_SUBGRAPH_TYPES['Upsample'] = '*upsample_blk_*'
SDV15_SUBGRAPH_TYPES['Conv Out'] = '*conv_out*'

#SDV15_SUBGRAPHS = ["", , , , , "*output*tf*"]
SDV15_SUBGRAPHS = OrderedDict()
SDV15_SUBGRAPHS['t-Embed'] = "*time_embedding*"
SDV15_SUBGRAPHS['Conv In'] = "*conv_in*"
SDV15_SUBGRAPHS['Input ResBlk'] = "*input_*resblk"
SDV15_SUBGRAPHS['Input Transformer'] = "*input_*tf_*"
SDV15_SUBGRAPHS['Mid ResBlk'] = "middle*resblk"
SDV15_SUBGRAPHS['Mid Transformer'] = "middle*tf*"
SDV15_SUBGRAPHS['Output ResBlk'] = "*output*resblk"
SDV15_SUBGRAPHS['Conv Out'] = "*conv_out*"


#### SDXL

SDXL_BLOCK_TYPES = OrderedDict() 
SDXL_BLOCK_TYPES['t-Embed'] = 'model.time_embedding*'
#SDXL_BLOCK_TYPES['t_add'] = 'model.add_embedding*'
SDXL_BLOCK_TYPES['Conv In'] = 'model.conv_in.weight_quantizer'
SDXL_BLOCK_TYPES["Input ResBlk"] = "model.down_blocks.*.resnets.*"
SDXL_BLOCK_TYPES["Downsample"] = '*downsamplers*'
SDXL_BLOCK_TYPES["Input SA"] = "model.down_blocks.*transformer_blocks.*.attn1*"
SDXL_BLOCK_TYPES["Input CA"] = "model.down_blocks.*transformer_blocks.*.attn2*"
SDXL_BLOCK_TYPES["Input FF"] = "model.down_blocks.*transformer_blocks.*.ff*"
SDXL_BLOCK_TYPES["Mid ResBlk"] = 'model.mid_block.resnets.*'
SDXL_BLOCK_TYPES["Mid SA"] = 'model.mid_block.*transformer_blocks.*.attn1*'
SDXL_BLOCK_TYPES["Mid CA"] = 'model.mid_block.*.transformer_blocks.*.attn2*'
SDXL_BLOCK_TYPES["Mid FF"] = 'model.mid_block.*.transformer_blocks.*ff*'
SDXL_BLOCK_TYPES["Output ResBlk"] = "model.up_blocks.*.resnets.*"
SDXL_BLOCK_TYPES["Output SA"] = "model.up_blocks.*.transformer_blocks.*.attn1*"
SDXL_BLOCK_TYPES["Output CA"] = "model.up_blocks.*.transformer_blocks.*.attn2*"
SDXL_BLOCK_TYPES["Output FF"] = "model.up_blocks.*.transformer_blocks.*.ff*"
SDXL_BLOCK_TYPES["Upsample"] = '*upsamplers*'
SDXL_BLOCK_TYPES["Conv Out"] = 'model.conv_out.weight_quantizer'

SDXL_LAYER_TYPES = [f"*{k}.weight_quantizer" for k, v in SDXL_LAYERS_LUT.items() if v < 26]

SDXL_LAYER_TYPES = OrderedDict()
#SDXL_LAYER_TYPES["t 1"] = "*time_embedding.linear_1*"
#SDXL_LAYER_TYPES["t 2"] = "*time_embedding.linear_2*"
SDXL_LAYER_TYPES["t-Embed"] = "*_embedding.linear_*"
#SDXL_LAYER_TYPES["a 1"] = "*add_embedding.linear_1*"
#SDXL_LAYER_TYPES["a 2"] = "*add_embedding.linear_2*"
SDXL_LAYER_TYPES["Conv In"] = "*conv_in*"
#SDXL_LAYER_TYPES["Downsample"] = "*downsamplers.0.conv*"
SDXL_LAYER_TYPES["Res In"] = "*conv1*"
SDXL_LAYER_TYPES["Res t-Emb"] = "*time_emb_proj*"
SDXL_LAYER_TYPES["Res Out"] = "*conv2*"
SDXL_LAYER_TYPES["Res Skip"] = "*conv_shortcut*"
SDXL_LAYER_TYPES["Proj In"] = "*proj_in*"
SDXL_LAYER_TYPES["SA-Q"] = "*attn1.to_q*"
SDXL_LAYER_TYPES["SA-K"] = "*attn1.to_k*"
SDXL_LAYER_TYPES["SA-V"] = "*attn1.to_v*"
SDXL_LAYER_TYPES["SA-Out"] = "*attn1.to_out.0*"
SDXL_LAYER_TYPES["CA-Q"] = "*attn2.to_q*"
SDXL_LAYER_TYPES["CA-K"] = "*attn2.to_k*"
SDXL_LAYER_TYPES["CA-V"] = "*attn2.to_v*"
SDXL_LAYER_TYPES["CA-Out"] = "*attn2.to_out.0*"
SDXL_LAYER_TYPES["FF 1"] = "*ff.net.0.proj*"
SDXL_LAYER_TYPES["FF 2"] = "*ff.net.2*"
SDXL_LAYER_TYPES["Proj Out"] = "*proj_out*"
#SDXL_LAYER_TYPES["Upsample"] = "*upsamplers.0.conv*"
SDXL_LAYER_TYPES["Conv Out"] = "*conv_out*"

# NOTE: Downsamplers incorporated into another blk.
SDXL_SUBGRAPH_TYPES = OrderedDict()
SDXL_SUBGRAPH_TYPES['t-Embed'] = '*time_embedding*'
SDXL_SUBGRAPH_TYPES['Conv In'] = '*conv_in*'
# These were commented out for bar plot
#SDXL_SUBGRAPH_TYPES['Downsample'] = '*op9*' # Its incorporated into resblk for SDXL
SDXL_SUBGRAPH_TYPES['ResBlk'] = '*resblk*'
SDXL_SUBGRAPH_TYPES['SA'] = '*attn1*'
SDXL_SUBGRAPH_TYPES['CA'] = '*attn2*'
SDXL_SUBGRAPH_TYPES['FF'] = '*ff*'
SDXL_SUBGRAPH_TYPES['Proj Out'] = '*proj*' # I am going to skip-over this one...
SDXL_SUBGRAPH_TYPES['Upsample'] = '*upsampler*' # Therefore, also skipping the upsample.
SDXL_SUBGRAPH_TYPES['Conv Out'] = '*conv_out*'

#SDXL_SUBGRAPHS = [, , , , , , , , ]
SDXL_SUBGRAPHS = OrderedDict()
SDXL_SUBGRAPHS['t-Embed'] = "*time_embedding*"
SDXL_SUBGRAPHS['Conv In'] = "*conv_in*"
SDXL_SUBGRAPHS['Conv Out'] = "*conv_out*"
SDXL_SUBGRAPHS['Input ResBlk'] = "*down_blocks*resblk"
SDXL_SUBGRAPHS['Input Transformer'] = "*down_blocks_*tf*"
SDXL_SUBGRAPHS['Mid ResBlk'] = "mid_block*resblk"
SDXL_SUBGRAPHS['Mid Transformer'] = "mid_block*tf*"
SDXL_SUBGRAPHS['Output ResBlk'] = "*up_blocks*resblk"
SDXL_SUBGRAPHS['Output Transformer'] = "*up_blocks*tf*"
SDXL_SUBGRAPHS['Proj Out'] = "*proj_out*"
