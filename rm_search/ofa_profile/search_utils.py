import random
from functools import partial
from search.rm_search.ofa_profile.constants import *
from search.rm_search.ofa.my_utils import ofa_str_configs_to_subnet_args


"""
PN
"""
def get_pn_stage_level_cands(space_id, log_f=print):
    if space_id == "npu":
        stage_level_cands = PN_STAGE_WISE_BLOCK_CANDS_NPU
        log_f("Using PN_STAGE_WISE_BLOCK_CANDS_NPU")
    elif space_id == "npu_random":
        stage_level_cands = \
            [random.sample(PN_BLOCKS, len(sc)) for sc in PN_STAGE_WISE_BLOCK_CANDS_NPU]
        log_f("Using PN_STAGE_WISE_BLOCK_CANDS_NPU random comparison")

    elif space_id == "gpu":
        stage_level_cands = PN_STAGE_WISE_BLOCK_CANDS_GPU
        log_f("Using PN_STAGE_WISE_BLOCK_CANDS_GPU")
    elif space_id == "gpu_random":
        stage_level_cands = \
            [random.sample(PN_BLOCKS, len(sc)) for sc in PN_STAGE_WISE_BLOCK_CANDS_GPU]
        log_f("Using PN_STAGE_WISE_BLOCK_CANDS_GPU random comparison")

    elif space_id == "cpu":
        stage_level_cands = PN_STAGE_WISE_BLOCK_CANDS_CPU
        log_f("Using PN_STAGE_WISE_BLOCK_CANDS_CPU")
    elif space_id == "cpu_random":
        stage_level_cands = \
            [random.sample(PN_BLOCKS, len(sc)) for sc in PN_STAGE_WISE_BLOCK_CANDS_CPU]
        log_f("Using PN_STAGE_WISE_BLOCK_CANDS_CPU random comparison")

    elif space_id == "max_acc":
        stage_level_cands = PN_STAGE_WISE_BLOCK_CANDS_MAX_ACC
        log_f("Using PN_STAGE_WISE_BLOCK_CANDS_MAX_ACC")
    elif space_id == "max_acc_random":
        stage_level_cands = \
            [random.sample(PN_BLOCKS, len(sc)) for sc in PN_STAGE_WISE_BLOCK_CANDS_MAX_ACC]
        log_f("Using PN_STAGE_WISE_BLOCK_CANDS_MAX_ACC random comparison")

    elif space_id is None:
        stage_level_cands = [PN_BLOCKS for _ in range(len(OFA_PN_STAGE_N_BLOCKS))]
        log_f("Using full PN space")
    else:
        raise ValueError("Unknown space id: {}".format(space_id))
    log_f("Stage-level cands:")
    for si, cands in enumerate(stage_level_cands):
        log_f("Stage {} cands: {}".format(si + 1, cands))
    return stage_level_cands


def get_pn_stage_level_mutate_probs(prob_type,
                                    uniform_prob=0.5,
                                    log_f=print):
    if prob_type == "uniform":
        probs = [uniform_prob for _ in range(len(OFA_PN_STAGE_N_BLOCKS))]
        log_f("Using uniform stage-level mutate probs: {}".format(probs))
    elif prob_type == "npu":
        probs = PN_STAGE_WISE_MUTATE_PROBS_NPU
        log_f("Using PN NPU stage-level mutate probs: {}".format(probs))
    elif prob_type == "gpu":
        probs = PN_STAGE_WISE_MUTATE_PROBS_GPU
        log_f("Using PN GPU stage-level mutate probs: {}".format(probs))
    elif prob_type == "cpu":
        probs = PN_STAGE_WISE_MUTATE_PROBS_CPU
        log_f("Using PN CPU stage-level mutate probs: {}".format(probs))
    elif prob_type == "max_acc":
        probs = PN_STAGE_WISE_MUTATE_PROBS_MAX_ACC
        log_f("Using PN MAX ACC stage-level mutate probs: {}".format(probs))
    else:
        raise ValueError("Unknown prob type: {}".format(prob_type))
    return probs


def get_pn_search_lat_predictor(predictor_type, resolution,
                                lat_predictor_checkpoint=None,
                                supernet_checkpoint=None,
                                supernet_model_dir=None,
                                log_f=print):
    if predictor_type == "custom":
        from search.rm_search.api.ofa_lat_predictor import load_ofa_pn_op_graph_lat_predictor, \
            ofa_op_graph_lat_predict
        log_f("Using custom lat predictor")
        log_f("Specified OFA-PN lat predictor checkpoint: {}".format(lat_predictor_checkpoint))
        predictor = load_ofa_pn_op_graph_lat_predictor(lat_predictor_checkpoint)
        lat_predictor = partial(ofa_op_graph_lat_predict,
                                resolution=resolution,
                                model=predictor, sub_space="pn")
    elif predictor_type == "flops":
        from search.rm_search.ofa.model_zoo import ofa_net
        from search.rm_search.ofa.utils.pytorch_utils import count_net_flops
        _pn_supernet = ofa_net(net_id=supernet_checkpoint, pretrained=True,
                               model_dir=supernet_model_dir)
        def lat_predictor(_net_configs):
            ks, e, d = ofa_str_configs_to_subnet_args(_net_configs, 21,
                                                      expected_prefix="mbconv2")
            _pn_supernet.set_active_subnet(ks=ks, e=e, d=d)
            _pn_subnet = _pn_supernet.get_active_subnet(preserve_weight=True)
            return count_net_flops(_pn_subnet, (1, 3, resolution, resolution)) / 1e6
    elif predictor_type == "random":
        lat_predictor = lambda _a: random.random()
    else:
        raise ValueError("Unknown predictor type: {}".format(predictor_type))
    return lat_predictor


def get_pn_max_n_blocks_per_stage(type_str, log_f=print):
    if type_str == "default" or type_str == "npu":
        max_n_stage_blocks = OFA_PN_STAGE_N_BLOCKS
        log_f("Using OFA_PN_STAGE_N_BLOCKS")
    elif type_str == "gpu":
        max_n_stage_blocks = OFA_PN_STAGE_N_BLOCKS_GPU
        log_f("Using OFA_PN_STAGE_N_BLOCKS_GPU")
    elif type_str == "cpu":
        max_n_stage_blocks = OFA_PN_STAGE_N_BLOCKS_CPU
        log_f("Using OFA_PN_STAGE_N_BLOCKS_CPU")
    else:
        raise ValueError("Unknown type str: {}".format(type_str))
    log_f("Max number of blocks per stage: {}".format(max_n_stage_blocks))
    return max_n_stage_blocks


"""
MBV3
"""
def get_mbv3_stage_level_cands(space_id, log_f=print):
    if space_id == "n10":
        stage_level_cands = MBV3_STAGE_WISE_BLOCK_CANDS_N10
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_N10")
    elif space_id == "n10_random":
        stage_level_cands = \
            [random.sample(MBV3_BLOCKS, len(sc)) for sc in MBV3_STAGE_WISE_BLOCK_CANDS_N10]
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_N10 random comparison")

    elif space_id == "npu":
        stage_level_cands = MBV3_STAGE_WISE_BLOCK_CANDS_NPU
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_NPU")
    elif space_id == "npu_random":
        stage_level_cands = \
            [random.sample(MBV3_BLOCKS, len(sc)) for sc in MBV3_STAGE_WISE_BLOCK_CANDS_NPU]
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_NPU random comparison")

    elif space_id == "gpu":
        stage_level_cands = MBV3_STAGE_WISE_BLOCK_CANDS_GPU
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_GPU")
    elif space_id == "gpu_random":
        stage_level_cands = \
            [random.sample(MBV3_BLOCKS, len(sc)) for sc in MBV3_STAGE_WISE_BLOCK_CANDS_GPU]
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_GPU random comparison")

    elif space_id == "cpu":
        stage_level_cands = MBV3_STAGE_WISE_BLOCK_CANDS_CPU
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_CPU")
    elif space_id == "cpu_random":
        stage_level_cands = \
            [random.sample(MBV3_BLOCKS, len(sc)) for sc in MBV3_STAGE_WISE_BLOCK_CANDS_CPU]
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_CPU random comparison")

    elif space_id == "max_acc":
        stage_level_cands = MBV3_STAGE_WISE_BLOCK_CANDS_MAX_ACC
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_MAX_ACC")
    elif space_id == "max_acc_random":
        stage_level_cands = \
            [random.sample(PN_BLOCKS, len(sc)) for sc in MBV3_STAGE_WISE_BLOCK_CANDS_MAX_ACC]
        log_f("Using MBV3_STAGE_WISE_BLOCK_CANDS_MAX_ACC random comparison")

    elif space_id is None:
        stage_level_cands = [MBV3_BLOCKS for _ in range(len(OFA_MBV3_STAGE_N_BLOCKS))]
        log_f("Using full MBV3 space")
    else:
        raise ValueError("Unknown space id: {}".format(space_id))
    log_f("Stage-level cands:")
    for si, cands in enumerate(stage_level_cands):
        log_f("Stage {} cands: {}".format(si + 1, cands))
    return stage_level_cands


def get_mbv3_stage_level_mutate_probs(prob_type,
                                      uniform_prob=0.5,
                                      log_f=print):
    if prob_type == "uniform":
        probs = [uniform_prob for _ in range(len(OFA_MBV3_STAGE_N_BLOCKS))]
        log_f("Using uniform stage-level mutate probs: {}".format(probs))
    elif prob_type == "npu":
        probs = MBV3_STAGE_WISE_MUTATE_PROBS_NPU
        log_f("Using MBV3 NPU stage-level mutate probs: {}".format(probs))
    elif prob_type == "n10":
        probs = MBV3_STAGE_WISE_MUTATE_PROBS_N10
        log_f("Using MBV3 N10 stage-level mutate probs: {}".format(probs))
    elif prob_type == "gpu":
        probs = MBV3_STAGE_WISE_MUTATE_PROBS_GPU
        log_f("Using MBV3 GPU stage-level mutate probs: {}".format(probs))
    elif prob_type == "cpu":
        probs = MBV3_STAGE_WISE_MUTATE_PROBS_CPU
        log_f("Using MBV3 CPU stage-level mutate probs: {}".format(probs))
    else:
        raise ValueError("Unknown prob type: {}".format(prob_type))
    return probs


def get_mbv3_search_lat_predictor(predictor_type, resolution,
                                  lat_predictor_checkpoint=None,
                                  supernet_checkpoint=None,
                                  supernet_model_dir=None,
                                  log_f=print):
    if predictor_type == "custom":
        from search.rm_search.api.ofa_lat_predictor import load_ofa_mbv3_op_graph_lat_predictor, \
            ofa_op_graph_lat_predict
        log_f("Using custom OFA-MBV3 lat predictor")
        log_f("Specified lat predictor checkpoint: {}".format(lat_predictor_checkpoint))
        predictor = load_ofa_mbv3_op_graph_lat_predictor(lat_predictor_checkpoint)
        lat_predictor = partial(ofa_op_graph_lat_predict,
                                resolution=resolution,
                                model=predictor, sub_space="mbv3")
    elif predictor_type == "n10":
        log_f("Using Note 10 OFA-MBV3 lat predictor")
        from search.rm_search.ofa.tutorial.latency_table import LatencyTable
        lat_table = LatencyTable()
        def lat_predictor(_net_configs):
            ks, e, d = ofa_str_configs_to_subnet_args(_net_configs, 20,
                                                      expected_prefix="mbconv3")
            arch_dict = {"r": (resolution, ),
                         "ks": ks,
                         "e": e,
                         "d": d}
            return lat_table.predict_efficiency(arch_dict)
    elif predictor_type == "flops":
        from search.rm_search.ofa.model_zoo import ofa_net
        from search.rm_search.ofa.utils.pytorch_utils import count_net_flops
        _mbv3_supernet = ofa_net(net_id=supernet_checkpoint, pretrained=True,
                               model_dir=supernet_model_dir)
        def lat_predictor(_net_configs):
            ks, e, d = ofa_str_configs_to_subnet_args(_net_configs, 20,
                                                      expected_prefix="mbconv3")
            _mbv3_supernet.set_active_subnet(ks=ks, e=e, d=d)
            _mbv3_subnet = _mbv3_supernet.get_active_subnet(preserve_weight=True)
            return count_net_flops(_mbv3_subnet, (1, 3, resolution, resolution)) / 1e6
    elif predictor_type == "random":
        lat_predictor = lambda _a: random.random()
    else:
        raise ValueError("Unknown predictor type: {}".format(predictor_type))
    return lat_predictor


def get_mbv3_max_n_blocks_per_stage(type_str, log_f=print):
    if type_str == "default" or type_str == "npu":
        max_n_stage_blocks = OFA_MBV3_STAGE_N_BLOCKS
        log_f("Using OFA_MBV3_STAGE_N_BLOCKS")
    elif type_str == "n10":
        max_n_stage_blocks = OFA_MBV3_STAGE_N_BLOCKS_N10
        log_f("Using OFA_MBV3_STAGE_N_BLOCKS_N10")
    elif type_str == "gpu":
        max_n_stage_blocks = OFA_MBV3_STAGE_N_BLOCKS_GPU
        log_f("Using OFA_MBV3_STAGE_N_BLOCKS_GPU")
    elif type_str == "cpu":
        max_n_stage_blocks = OFA_MBV3_STAGE_N_BLOCKS_CPU
        log_f("Using OFA_MBV3_STAGE_N_BLOCKS_CPU")
    else:
        raise ValueError("Unknown type str: {}".format(type_str))
    log_f("Max number of blocks per stage: {}".format(max_n_stage_blocks))
    return max_n_stage_blocks


"""
ResNet
"""
def get_resnet_stage_level_e_cands(space_id, log_f=print):
    e_cands = [OFA_RES_EXPANSION_RATIOS for _ in range(OFA_RES_N_SEARCHABLE_STAGES)]
    e_cands[0] = ()
    log_f("Using default OFA_RES_EXPANSION_RATIOS")
    log_f("OFA-ResNet e candidates: {}".format(e_cands))
    return e_cands


def get_resnet_stage_level_w_cands(space_id, log_f=print):
    if space_id == "max_acc":
        w_cands = OFA_RES_STAGE_WISE_WIDTH_INDS_MAX_ACC
        log_f("Using OFA_RES_STAGE_WISE_WIDTH_INDS_MAX_ACC")
    else:
        w_cands = [OFA_RES_W_INDS for _ in range(OFA_RES_N_SEARCHABLE_STAGES)]
        log_f("Using default OFA_RES_W_INDS")
    log_f("OFA-ResNet w candidates: {}".format(w_cands))
    return w_cands


def get_resnet_stage_level_mutate_probs(prob_type,
                                        uniform_prob=0.5,
                                        log_f=print):
    if prob_type == "default":
        probs = [uniform_prob for _ in range(OFA_RES_N_SEARCHABLE_STAGES)]
        probs[0] /= 2.
        log_f("Using default stage-level mutate probs: {}".format(probs))
    else:
        raise ValueError("Unknown prob type: {}".format(prob_type))
    return probs


def get_resnet_max_n_blocks_per_stage(type_str, log_f=print):
    max_n_stage_blocks = tuple([1] + list(OFA_RES_STAGE_MAX_N_BLOCKS))
    log_f("Using default OFA_RES_STAGE_MAX_N_BLOCKS")
    log_f("Max number of blocks per stage: {}".format(max_n_stage_blocks))
    return max_n_stage_blocks


def get_resnet_min_n_blocks_per_stage(type_str, log_f=print):
    if type_str == "max_acc":
        min_n_stage_blocks = tuple([1] + list(OFA_RES_STAGE_MIN_N_BLOCKS_MAX_ACC))
        log_f("Using OFA_RES_STAGE_MIN_N_BLOCKS_MAX_ACC")
    else:
        min_n_stage_blocks = tuple([1] + list(OFA_RES_STAGE_MIN_N_BLOCKS))
        log_f("Using default OFA_RES_STAGE_MIN_N_BLOCKS")
    log_f("Min number of blocks per stage: {}".format(min_n_stage_blocks))
    return min_n_stage_blocks


def get_resnet_search_lat_predictor(predictor_type, resolution,
                                    supernet_checkpoint=None,
                                    supernet_model_dir=None):
    from search.rm_search.ofa_profile.arch_mutator import OFAResNetArchWrapper
    if predictor_type == "flops":
        from search.rm_search.ofa.model_zoo import ofa_net
        from search.rm_search.ofa.utils.pytorch_utils import count_net_flops
        _resnet_supernet = ofa_net(net_id=supernet_checkpoint, pretrained=True,
                                   model_dir=supernet_model_dir)
        def lat_predictor(_net_configs:OFAResNetArchWrapper):
            _resnet_supernet.set_active_subnet(d=_net_configs.d_list,
                                               e=_net_configs.e_list,
                                               w=_net_configs.w_list)
            _resnet_subnet = _resnet_supernet.get_active_subnet(preserve_weight=True)
            return count_net_flops(_resnet_subnet, (1, 3, resolution, resolution)) / 1e6
    elif predictor_type == "random":
        lat_predictor = lambda _a: random.random()
    else:
        raise ValueError("Unknown predictor type: {}".format(predictor_type))
    return lat_predictor
