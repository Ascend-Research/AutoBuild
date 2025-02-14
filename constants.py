ONNX_IR_ATTR_FEAT_DICT = {
    'EMPTY': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Conv1x1': [1, 1, 0, 0, 0, 1, 1, 0, 1],
    'Conv3x3': [3, 3, 0, 0, 0, 1, 1, 0, 1],
    'Conv5x5': [5, 5, 0, 0, 0, 1, 1, 0, 1],
    'Conv7x7': [7, 7, 0, 0, 0, 1, 1, 0, 1],
    'Conv9x9': [9, 9, 0, 0, 0, 1, 1, 0, 1],
    'DWConv3x3': [3, 3, 0, 1, 0, 1, 1, 0, 1],
    'DWConv5x5': [5, 5, 0, 1, 0, 1, 1, 0, 1],
    'DWConv7x7': [7, 7, 0, 1, 0, 1, 1, 0, 1],
    'DWConv9x9': [9, 9, 0, 1, 0, 1, 1, 0, 1],
    'DilConv3x3': [3, 3, 0, 0, 0, 2, 2, 0, 1],
    'NoBiasConv3x3': [3, 3, 0, 0, 0, 1, 1, 0, 0],
}

MBCONV_EXPAND_MAP = {
    3: 0,
    4: 1,
    6: 2
}

MBCONV_KERNEL_MAP = {
    3: 0,
    5: 1,
    7: 2,
}

TGeoNodeEmbedding_FEAT_MAP = {
    0: "Op Type",
    1: "Shape",
    2: "Attribute"
}

MobileNetNodeEmbedding_FEAT_MAP = {
    0: "Stage",
    1: "Layer",
    2: "MBConv",
    3: "Expand",
    4: "Kernel",
    5: "Resolution"
}

DiTNodeEmbedding_FEAT_MAP = {
    0: "Quant. Method",
    1: "Bit Precision",
    2: "Reduction",
    3: "Quant. Error",
    4: "Block Idx",
    5: "Layer Type"
}

SDV15NodeEmbedding_FEAT_MAP = {
    0: "Quant. Method",
    1: "Bit Precision",
    2: "Reduction",
    3: "Quant. Error",
    4: "Stage Idx",
    5: "Block Idx",
    6: "Layer Type"
}

SDXLNodeEmbedding_FEAT_MAP = {
    0: "Quant. Method",
    1: "Bit Precision",
    2: "Reduction",
    3: "Quant. Error",
    4: "Stage Idx",
    5: "Block Idx",
    6: "Layer Type"
}