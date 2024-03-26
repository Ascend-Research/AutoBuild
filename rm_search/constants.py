DIR_TO_APPEND = "~/autobuild"

DK_BATCH_SIZE = "batch_size"
DK_BATCH_TARGET_TSR = "batch_target_tensor"
DK_BATCH_UNIQUE_STR_ID_SET = "batch_unique_id_set"
DK_BATCH_NODE_LABEL_LIST = "batch_node_label_list"
DK_BATCH_NODE_FEATURE_TSR = "batch_node_feature_tsr"
DK_BATCH_NODE_SHAPE_TSR = "batch_node_shape_tsr"


CHKPT_COMPLETED_EPOCHS = "completed_epochs"
CHKPT_MODEL = "model"
CHKPT_DISCRIMINATOR = "discriminator"
CHKPT_OPTIMIZER = "optimizer"
CHKPT_METADATA = "metadata"
CHKPT_PARAMS = "params"
CHKPT_BEST_EVAL_RESULT = "best_eval_result"
CHKPT_BEST_EVAL_EPOCH = "best_eval_epoch"
CHKPT_PAST_EVAL_RESULTS = "past_eval_results"
CHKPT_ITERATION = "iteration"
CHKPT_BEST_EVAL_ITERATION = "best_eval_iteration"


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

RESNET_EXP_MAP = {
    0.2: 0,
    0.25: 1,
    0.35: 2
}

TGeoNodeEmbedding_FEAT_MAP = {
    0: "Op Type",
    1: "Shape",
    2: "Attribute"
}

ResNetNodeEmbedding_FEAT_MAP = {
    0: "Stage",
    1: "Layer",  # Also known as block
    2: "Width",
    3: "Expansion",
    4: "Resolution"
}

MobileNetNodeEmbedding_FEAT_MAP = {
    0: "Stage",
    1: "Layer",  # Also known as block
    2: "MBConv",
    3: "Expand",
    4: "Kernel",
    5: "Resolution"
}
