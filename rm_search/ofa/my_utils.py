from search.rm_search.utils.math_utils import make_divisible
from search.rm_search.ofa_profile.constants import OFA_RES_ADDED_DEPTH_LIST, OFA_RES_WIDTH_MULTIPLIERS, \
    OFA_RES_STAGE_MIN_N_BLOCKS, OFA_RES_STAGE_MAX_N_BLOCKS


def _flatten_nested_lists(lists):
    def _recur_flat(val):
        if isinstance(val, list):
            for v in val:
                _recur_flat(v)
        else:
            rv.append(val)
    rv = []
    _recur_flat(lists)
    return rv


def ofa_subnet_args_to_str_configs(block_prefix, k_list, e_list, d_list,
                                   max_n_blocks_per_stage=4,
                                   group_by_stage=True):
    assert len(k_list) == len(e_list)
    rv = []
    k_e_pairs = zip(k_list, e_list)
    for d in d_list:
        stage_blocks = []
        for bi in range(d):
            k, e = next(k_e_pairs)
            block = block_prefix + "_e{}_k{}".format(e, k)
            stage_blocks.append(block)
        rv.append(stage_blocks)
        for _ in range(max_n_blocks_per_stage - d):
            # Ignore k, e pairs that are outside of the current depth
            try:
                _, _ = next(k_e_pairs)
            except StopIteration:
                break
    if not group_by_stage:
        rv = _flatten_nested_lists(rv)
    return rv


def ofa_str_configs_to_subnet_args(net_configs, max_n_net_blocks,
                                   max_n_blocks_per_stage=4,
                                   fill_k=0, fill_e=0,
                                   expected_prefix=None):
    # Input configs must be grouped by stages
    d_list = [len(bs) for bs in net_configs]
    k_list, e_list= [], []
    blocks = iter(_flatten_nested_lists(net_configs))
    for stage_depth in d_list:
        for bi in range(stage_depth):
            block = next(blocks)
            prefix, e_str, k_str = block.split("_")
            if expected_prefix is not None:
                assert expected_prefix == prefix, \
                    "Invalid block prefix: {}, expected: {}".format(prefix, expected_prefix)
            assert e_str.startswith("e"), "Invalid block str: {}".format(block)
            assert k_str.startswith("k"), "Invalid block str: {}".format(block)
            e = int(e_str.replace("e", ""))
            k = int(k_str.replace("k", ""))
            k_list.append(k)
            e_list.append(e)
        for fi in range(max_n_blocks_per_stage - stage_depth):
            # Fill such that k e lists are always the max length
            k_list.append(fill_k)
            e_list.append(fill_e)
    assert len(k_list) >= max_n_net_blocks
    k_list = k_list[:max_n_net_blocks]
    e_list = e_list[:max_n_net_blocks]
    return k_list, e_list, d_list


def ofa_res_subnet_args_to_str_configs(d_list, e_list, w_list,
                                       group_by_stage=True):
    assert len(d_list) == 5 == len(w_list) - 1
    net_structure = []
    block_output_channel_inds = w_list[2:]

    # First determine stem type
    if d_list[0] == max(OFA_RES_ADDED_DEPTH_LIST):
        stem_prefix = "stem+res"
    else:
        stem_prefix = "stem"
    hidden_idx, output_idx = w_list[0], w_list[1]
    stem_output_channels = [
        make_divisible(64 * width_mult, 8) for width_mult in OFA_RES_WIDTH_MULTIPLIERS
    ]
    stem_hidden_channels = [
        make_divisible(channel // 2, 8) for channel in stem_output_channels
    ]
    hidden_size = stem_hidden_channels[hidden_idx]
    output_size = stem_output_channels[output_idx]
    stem_name = "_".join([stem_prefix, "h{}".format(hidden_size), "o{}".format(output_size)])
    net_structure.append([stem_name])

    # Next determine the stage blocks
    e_vals = iter(e_list)
    for si, d in enumerate(d_list[1:]):
        stage_n_blocks = OFA_RES_STAGE_MIN_N_BLOCKS[si] + d
        stage_blocks = []
        for bi in range(stage_n_blocks):
            e = next(e_vals)
            if e < 1.0:
                exp_str = "e0{}".format(int(e * 100.))
            else:
                exp_str = "e{}".format(int(e * 100.))
            block = "res" + "_{}_k3".format(exp_str)
            stage_blocks.append(block)
        net_structure.append(stage_blocks)
        for _ in range(OFA_RES_STAGE_MAX_N_BLOCKS[si] - stage_n_blocks):
            # Ignore e vals that are outside of the current depth
            _ = next(e_vals)
    if not group_by_stage:
        net_structure = _flatten_nested_lists(net_structure)
    return net_structure, block_output_channel_inds


def get_ref_block_inds(net_configs, max_stage_block_counts):
    """
    Get the ref index of each block in net_configs, ref to the longest possible net
    net_configs must be stage-wise organized
    """
    assert len(net_configs) == len(max_stage_block_counts), \
        "{} vs. {}".format(net_configs, max_stage_block_counts)
    rv = []
    curr_idx = 0
    for cnt, blocks in zip(max_stage_block_counts, net_configs):
        stage_inds = []
        for _ in blocks:
            stage_inds.append(curr_idx)
            curr_idx += 1
        for _ in range(cnt - len(blocks)):
            curr_idx += 1
        rv.append(stage_inds)
    assert len(rv) == len(net_configs), \
        "{} vs. {}".format(rv, net_configs)
    assert all(len(v1) == len(v2) for v1, v2 in zip(rv, net_configs)), \
        "{} vs. {}".format(rv, net_configs)
    return rv


if __name__ == "__main__":

    # print(ofa_subnet_args_to_str_configs("mbconv3", # MBV3
    #                                      [7,5,7,3, 3,5,7,7, 7,3,7,3, 3,7,7,3, 7,7,5,7],
    #                                      [6,4,3,3, 4,3,4,4, 3,3,3,3, 6,6,3,3, 6,3,4,3],
    #                                      [3,       3,       4,       2,       2]))
    # print(ofa_subnet_args_to_str_configs("mbconv2", # PN
    #                                      [7,5,7,3, 3,5,7,7, 7,3,7,3, 3,7,7,3, 7,7,5,7, 7,7,5,7],
    #                                      [6,4,3,3, 4,3,4,4, 3,3,3,3, 6,6,3,3, 6,3,4,3, 7,7,5,7],
    #                                      [3,       3,       4,       2,       2,       1]))

    # print(ofa_str_configs_to_subnet_args([["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    #                                       ["mbconv3_e6_k7", "mbconv3_e6_k7",],
    #                                       ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    #                                       ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    #                                       ["mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7", "mbconv3_e6_k7",],
    #                                       ["mbconv3_e6_k7"],
    #                                      ], max_n_net_blocks=21))

    # print(ofa_res_subnet_args_to_str_configs([2, 2, 2, 2, 2],
    #                                          [0.25, 0.25, 0.35, 0.25, 0.25, 0.35, 0.25, 0.35, 0.25, 0.25, 0.25, 0.2, 0.2, 0.35, 0.35, 0.35, 0.2, 0.25],
    #                                          [2, 1, 0, 2, 2, 0]))

    print("done")
