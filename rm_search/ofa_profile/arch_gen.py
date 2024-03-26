import torch
import sys
import pickle
import random
from search.rm_search.constants import DIR_TO_APPEND
from tqdm import tqdm
from search.rm_search.params import *
from search.rm_search.utils.model_utils import device
from search.rm_search.ofa_profile.constants import *
from search.rm_search.ofa_profile.arch_utils import *

sys.path.append(DIR_TO_APPEND)


def sample_ofa_pn_configs(stage_n_blocks=OFA_PN_STAGE_N_BLOCKS,
                          stage_level_candidates=None):
    net_configs = []
    for si, max_n_blocks in enumerate(stage_n_blocks):
        stage_blocks = []
        choices = list(range(min(2, max_n_blocks), max_n_blocks + 1))
        random.shuffle(choices)
        n_blocks = choices[0]
        for _ in range(n_blocks):
            if stage_level_candidates is not None:
                candidates = stage_level_candidates[si]
            else:
                candidates = PN_BLOCKS
            block = random.choice(candidates)
            stage_blocks.append(block)
        assert len(stage_blocks) > 0
        net_configs.append(stage_blocks)
    return net_configs


def sample_ofa_pn_stage_configs(stages):
    net_configs = []
    for si, stage_choices in enumerate(stages):
        if si == 4:
            stages = random.choice(stage_choices)
            net_configs.append(stages[0])
            net_configs.append(stages[1])
        else:
            net_configs.append(random.choice(stage_choices))

    return net_configs


def get_ofa_pn_mbconv_io_shapes(net_configs, w=OFA_W_PN,
                                H=224, W=224, normalize=True,
                                H_max=224, W_max=224,
                                strides=(2, 2, 2, 1, 2, 1),
                                block_channel_sizes=(16, 24, 40, 80, 96, 192, 320),
                                log_f=print):
    assert len(net_configs) == len(strides)
    rv = []
    block_channel_sizes = get_final_channel_sizes(block_channel_sizes, w)
    H_in, W_in = H // 2, W // 2
    H_out, W_out = H_in, W_in
    C_in = block_channel_sizes[0]
    C_max = max(block_channel_sizes)
    H_max /= 2
    W_max /= 2
    for stage_i, blocks in enumerate(net_configs):
        C_out = block_channel_sizes[stage_i + 1]
        stride = strides[stage_i]
        stage_shapes = []
        for block_i, block in enumerate(blocks):
            if block_i == 0 and stride == 2:
                H_out = H_in // 2
                W_out = W_in // 2
            stage_shapes.append( [H_in, H_out, W_in, W_out, C_in, C_out] )
            C_in = C_out
            H_in = H_out
            W_in = W_out
        rv.append(stage_shapes)
    if normalize:
        if log_f is not None:
            log_f("Normalizing H_max={}".format(H_max))
            log_f("Normalizing W_max={}".format(W_max))
            log_f("Normalizing C_max={}".format(C_max))
        for stage_shapes in rv:
            for b_shapes in stage_shapes:
                b_shapes[0] /= H_max
                b_shapes[1] /= H_max
                b_shapes[2] /= W_max
                b_shapes[3] /= W_max
                b_shapes[4] /= C_max
                b_shapes[5] /= C_max
                assert all(0 <= v <= 1 for v in b_shapes), \
                    "Normalized block shapes: {}".format(b_shapes)
    return rv


def get_ofa_pn_net_idx_shape_feat(net_configs,
                                  H=224, W=224, normalize=True,
                                  H_max=224, W_max=224,
                                  log_f=print):
    shapes = get_ofa_pn_mbconv_io_shapes(net_configs,
                                         H=H, W=W, normalize=normalize,
                                         H_max=H_max, W_max=W_max,
                                         log_f=log_f)
    net_inds = []
    net_shapes = []
    for stage_i, blocks in enumerate(net_configs):
        for block_i, b_op in enumerate(blocks):
            type_idx = PN_OP2IDX[b_op]
            net_inds.append(type_idx)
            net_shapes.append(shapes[stage_i][block_i])
    assert len(net_inds) == len(net_shapes)
    return net_inds, net_shapes


def gen_random_ofa_pn_configs(cached_samples_file, n_nets, log_f=print):
    if os.path.isfile(cached_samples_file):
        log_f("Loading from cache configs file: {}".format(cached_samples_file))
        with open(cached_samples_file, "rb") as f:
            data = pickle.load(f)
        log_f("{} unique net configs loaded".format(len(data)))
        return data
    nets = []
    unique_nets = set()
    n_attempts = 0
    bar = tqdm(total=n_nets, desc="Sampling unique net configs", ascii=True)
    while n_attempts < 10000 and len(nets) < n_nets:
        configs = sample_ofa_pn_configs()
        n_attempts += 1
        if str(configs) in unique_nets:
            continue
        unique_nets.add(str(configs))
        nets.append(configs)
        bar.update(1)
        n_attempts = 0
    bar.close()
    log_f("Collected {} unique networks".format(len(nets)))
    with open(cached_samples_file, "wb") as f:
        pickle.dump(nets, f, protocol=4)
    return nets


def sample_ofa_mbv3_configs(stage_n_blocks=OFA_MBV3_STAGE_N_BLOCKS,
                            stage_level_candidates=None):
    net_configs = []
    for si, max_n_blocks in enumerate(stage_n_blocks):
        stage_blocks = []
        choices = list(range(2, max_n_blocks + 1))
        random.shuffle(choices)
        n_blocks = choices[0]
        for _ in range(n_blocks):
            if stage_level_candidates is not None:
                candidates = stage_level_candidates[si]
            else:
                candidates = MBV3_BLOCKS
            block = random.choice(candidates)
            stage_blocks.append(block)
        assert len(stage_blocks) > 0
        net_configs.append(stage_blocks)
    return net_configs


def sample_ofa_mbv3_stage_configs(stages):
    net_configs = []
    for si, stage_choices in enumerate(stages):
        net_configs.append(random.choice(stage_choices))
    return net_configs


def get_ofa_mbv3_mbconv_io_shapes(net_configs, w=OFA_W_MBV3,
                                  H=224, W=224, normalize=True,
                                  H_max=224, W_max=224,
                                  strides=(2, 2, 2, 1, 2),
                                  block_channel_sizes=(16, 24, 40, 80, 112, 160),
                                  log_f=print):
    assert len(net_configs) == len(strides)
    rv = []
    block_channel_sizes = get_final_channel_sizes(block_channel_sizes, w)
    H_in, W_in = H // 2, W // 2
    H_out, W_out = H_in, W_in
    C_in = block_channel_sizes[0]
    C_max = max(block_channel_sizes)
    H_max /= 2
    W_max /= 2
    for stage_i, blocks in enumerate(net_configs):
        C_out = block_channel_sizes[stage_i + 1]
        stride = strides[stage_i]
        stage_shapes = []
        for block_i, block in enumerate(blocks):
            if block_i == 0 and stride == 2:
                H_out = H_in // 2
                W_out = W_in // 2
            stage_shapes.append( [H_in, H_out, W_in, W_out, C_in, C_out] )
            C_in = C_out
            H_in = H_out
            W_in = W_out
        rv.append(stage_shapes)
    if normalize:
        if log_f is not None:
            log_f("Normalizing H_max={}".format(H_max))
            log_f("Normalizing W_max={}".format(W_max))
            log_f("Normalizing C_max={}".format(C_max))
        for stage_shapes in rv:
            for b_shapes in stage_shapes:
                b_shapes[0] /= H_max
                b_shapes[1] /= H_max
                b_shapes[2] /= W_max
                b_shapes[3] /= W_max
                b_shapes[4] /= C_max
                b_shapes[5] /= C_max
                assert all(0 <= v <= 1 for v in b_shapes), \
                    "Normalized block shapes: {}".format(b_shapes)
    return rv


def get_ofa_mbv3_net_idx_shape_feat(net_configs,
                                    H=224, W=224, normalize=True,
                                    H_max=224, W_max=224,
                                    log_f=print):
    shapes = get_ofa_mbv3_mbconv_io_shapes(net_configs,
                                           H=H, W=W, normalize=normalize,
                                           H_max=H_max, W_max=W_max,
                                           log_f=log_f)
    net_inds = []
    net_shapes = []
    for stage_i, blocks in enumerate(net_configs):
        for block_i, b_op in enumerate(blocks):
            type_idx = MBV3_OP2IDX[b_op]
            net_inds.append(type_idx)
            net_shapes.append(shapes[stage_i][block_i])
    assert len(net_inds) == len(net_shapes)
    return net_inds, net_shapes


def gen_random_ofa_mbv3_configs(cached_samples_file, n_nets, log_f=print):
    if os.path.isfile(cached_samples_file):
        log_f("Loading from cache configs file: {}".format(cached_samples_file))
        with open(cached_samples_file, "rb") as f:
            data = pickle.load(f)
        log_f("{} unique net configs loaded".format(len(data)))
        return data
    nets = []
    unique_nets = set()
    n_attempts = 0
    bar = tqdm(total=n_nets, desc="Sampling unique net configs", ascii=True)
    while n_attempts < 10000 and len(nets) < n_nets:
        configs = sample_ofa_mbv3_configs()
        n_attempts += 1
        if str(configs) in unique_nets:
            continue
        unique_nets.add(str(configs))
        nets.append(configs)
        bar.update(1)
        n_attempts = 0
    bar.close()
    log_f("Collected {} unique networks".format(len(nets)))
    with open(cached_samples_file, "wb") as f:
        pickle.dump(nets, f, protocol=4)
    return nets


def ofa_resnet_op2idx():
    # First add stems
    stem_output_channels = [
        make_divisible(64 * width_mult, 8) for width_mult in OFA_RES_WIDTH_MULTIPLIERS
    ]
    stem_hidden_channels = [
        make_divisible(channel // 2, 8) for channel in stem_output_channels
    ]
    blocks = []
    unique_blocks = set()
    for h in stem_hidden_channels:
        for o in stem_output_channels:
            for prefix in OFA_RES_STEM_PREFIXES:
                stem_block = "_".join([prefix, "h{}".format(h), "o{}".format(o)])
                if stem_block not in unique_blocks:
                    unique_blocks.add(stem_block)
                    blocks.append(stem_block)
    # Next add blocks
    for r in OFA_RES_EXPANSION_RATIOS:
        for k in OFA_RES_KERNEL_SIZES:
            if r < 1.0:
                exp_str = "e0{}".format(int(r * 100.))
            else:
                exp_str = "e{}".format(int(r * 100.))
            res_block = "_".join(["res", exp_str, "k{}".format(k)])
            blocks.append(res_block)
            assert res_block not in unique_blocks, "Duplicated name: {}".format(res_block)
            unique_blocks.add(res_block)
    return {b: i for i, b in enumerate(blocks)}


def _add_ofa_resnet_hooks(net):
    from search.rm_search.ofa_profile.arch_gpu_cpu_lat import InputOutputShapeHook
    hooks = []
    hook = InputOutputShapeHook(net.input_stem[0])
    hooks.append(hook)
    hook = InputOutputShapeHook(net.input_stem[-1])
    hooks.append(hook)
    for bi, block in enumerate(net.blocks):
        hook = InputOutputShapeHook(block)
        hooks.append(hook)
    return hooks


def _get_ofa_resnet_subnet_io_shapes(subnet, resolution):
    hooks = _add_ofa_resnet_hooks(subnet)
    net_batch = torch.ones([1, 3, resolution, resolution]).float().to(device())
    with torch.no_grad():
        subnet.eval()
        subnet(net_batch)
    net_shapes = []
    # Special treatment for stem
    C_in, H_in, W_in = hooks[0].input_shape[1:]
    C_out, H_out, W_out = hooks[1].output_shape[1:]
    net_shapes.append((H_in, H_out, W_in, W_out, C_in, C_out))
    # For blocks
    for hook in hooks[2:]:
        C_in, H_in, W_in = hook.input_shape[1:]
        C_out, H_out, W_out = hook.output_shape[1:]
        net_shapes.append((H_in, H_out, W_in, W_out, C_in, C_out))
    for hook in hooks:
        hook.close()
    return net_shapes


def get_ofa_resnet_io_shapes(d_list, e_list, w_list, resolution,
                             loaded_model=None):
    from search.rm_search.ofa.model_zoo import ofa_net
    from search.rm_search.ofa_profile.constants import OFA_RES_STAGE_MIN_N_BLOCKS
    if loaded_model is not None:
        model_dir = P_SEP.join([SAVED_MODELS_DIR, "ofa_checkpoints"])
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        ofa_network = ofa_net('ofa_resnet50', pretrained=True, model_dir=model_dir)
    else:
        ofa_network = loaded_model
    ofa_network.set_active_subnet(d=d_list, e=e_list, w=w_list)
    subnet = ofa_network.get_active_subnet(preserve_weight=True).to(device())
    net_shapes = _get_ofa_resnet_subnet_io_shapes(subnet, resolution)
    assert len(net_shapes) == sum(OFA_RES_STAGE_MIN_N_BLOCKS) + sum(d_list[1:]) + 1, \
        "{} != sum({}) + sum({}) + 1".format(len(net_shapes), OFA_RES_STAGE_MIN_N_BLOCKS, d_list[1:])
    return net_shapes


def get_ofa_resnet_arch_id(d_list, e_list, w_list,
                           stage_max_n_blocks=OFA_RES_STAGE_MAX_N_BLOCKS):
    block_d_list = d_list[1:]
    if d_list[0] != max(OFA_RES_ADDED_DEPTH_LIST):
        d_list[0] = 0 # If we do not choose 2 then choosing 0 or 1 means the same thing
    assert len(block_d_list) == len(w_list[2:]) == \
        len(OFA_RES_STAGE_MIN_N_BLOCKS) == len(stage_max_n_blocks)
    active_e_list = []
    for si, d in enumerate(block_d_list):
        n_active_blocks = OFA_RES_STAGE_MIN_N_BLOCKS[si] + d
        assert n_active_blocks <= stage_max_n_blocks[si]
        active_e_vals = e_list[:n_active_blocks]
        e_list = e_list[OFA_RES_STAGE_MAX_N_BLOCKS[si]:]
        active_e_list.extend(active_e_vals)
    assert len(e_list) == 0, "Invalid e_list input: {}".format(e_list)
    return str([d_list, active_e_list, w_list])


def sample_ofa_resnet_configs(resolutions=(192, 208, 224),
                              pre_compute_shapes=True):
    # The way to handle resnet is very different
    from search.rm_search.ofa.model_zoo import ofa_net
    model_dir = P_SEP.join([SAVED_MODELS_DIR, "ofa_checkpoints"])
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    ofa_network = ofa_net('ofa_resnet50', pretrained=True, model_dir=model_dir)
    subnet_configs = ofa_network.sample_active_subnet()
    if pre_compute_shapes:
        subnet = ofa_network.get_active_subnet(preserve_weight=True).to(device())
        # Add hooks to get i/o shapes
        res2shapes = {}
        for r in resolutions:
            net_shapes = _get_ofa_resnet_subnet_io_shapes(subnet, r)
            assert len(net_shapes) == sum(OFA_RES_STAGE_MIN_N_BLOCKS) + sum(subnet_configs["d"][1:]) + 1
            res2shapes[r] = net_shapes
    else:
        res2shapes = None
    net_configs = {
        "d": subnet_configs["d"],
        "e": subnet_configs["e"],
        "w": subnet_configs["w"],
        "shapes": sorted([(r, v) for r, v in res2shapes.items()], key=lambda t:t[0]) \
            if res2shapes is not None else None
    }
    return net_configs


def gen_random_ofa_resnet_configs(cached_samples_file, n_nets,
                                  log_f=print):
    if os.path.isfile(cached_samples_file):
        log_f("Loading from cache configs file: {}".format(cached_samples_file))
        with open(cached_samples_file, "rb") as f:
            data = pickle.load(f)
        log_f("{} unique net configs loaded".format(len(data)))
        return data
    nets = []
    unique_nets = set()
    n_attempts = 0
    bar = tqdm(total=n_nets, desc="Sampling unique net configs", ascii=True)
    while n_attempts < 1000 and len(nets) < n_nets:
        configs = sample_ofa_resnet_configs(pre_compute_shapes=True)
        n_attempts += 1
        _id = get_ofa_resnet_arch_id(configs["d"], configs["e"], configs["w"])
        if _id in unique_nets:
            continue
        unique_nets.add(_id)
        nets.append(configs)
        bar.update(1)
        n_attempts = 0
    bar.close()
    log_f("Collected {} unique networks".format(len(nets)))
    with open(cached_samples_file, "wb") as f:
        pickle.dump(nets, f, protocol=4)
    return nets


def sample_constrained_ofa_resnet(min_n_stage_blocks, max_n_stage_blocks,
                                  stem_d_cands, stage_wise_e_cands, stage_wise_w_cands):
    stem_d = random.choice(stem_d_cands)
    stem_w_list = [random.choice(stage_wise_w_cands[0]),
                   random.choice(stage_wise_w_cands[0])]
    # Cut off stem entries
    # Now sample for each stage
    min_n_stage_blocks = min_n_stage_blocks[1:]
    max_n_stage_blocks = max_n_stage_blocks[1:]
    stage_wise_e_cands = stage_wise_e_cands[1:]
    stage_wise_w_cands = stage_wise_w_cands[1:]
    d_list, e_list, w_list = [stem_d], [], [w for w in stem_w_list]
    assert len(min_n_stage_blocks) == len(max_n_stage_blocks)
    for si in range(len(max_n_stage_blocks)):
        stage_depth_min = min_n_stage_blocks[si]
        stage_depth_max = max_n_stage_blocks[si]
        assert stage_depth_min <= stage_depth_max
        base_num = stage_depth_min - OFA_RES_STAGE_MIN_N_BLOCKS[si]
        d_cands = []
        for di in range(stage_depth_max - stage_depth_min + 1):
            d_cands.append(base_num + di)
        stage_d = random.choice(d_cands)
        d_list.append(stage_d)
        stage_w = random.choice(stage_wise_w_cands[si])
        w_list.append(stage_w)
        stage_depth = OFA_RES_STAGE_MIN_N_BLOCKS[si] + stage_d
        assert stage_depth <= stage_depth_max
        for bi in range(stage_depth):
            e_list.append(random.choice(stage_wise_e_cands[si]))
        for _ in range(OFA_RES_STAGE_MAX_N_BLOCKS[si] - stage_depth):
            # Add e fillers
            e_list.append(random.choice(stage_wise_e_cands[si]))
    assert len(e_list) == sum(OFA_RES_STAGE_MAX_N_BLOCKS)
    return d_list, e_list, w_list


def sample_stages_ofa_resnet(stage_units):

    arch = []
    for unit_list in stage_units:
        arch.append(random.choice(unit_list))
    return assemble_whole_net_cfg(*arch)


if __name__ == "__main__":
    # gen_random_ofa_pn_configs("../../../cache/ofa_pn_100k_random_net_configs.pkl", n_nets=100000)
    # gen_random_ofa_mbv3_configs("../../../cache/ofa_mbv3_100k_random_net_configs.pkl", n_nets=100000)
    # gen_random_ofa_resnet_configs("../../../cache/ofa_resnet_100k_random_net_configs.pkl", n_nets=100000)
    print("done")
