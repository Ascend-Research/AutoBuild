from search.rm_search.ofa.model_zoo import ofa_net
from search.rm_search.ofa.utils.pytorch_utils import count_net_flops
from search.rm_search.ofa.my_utils import ofa_str_configs_to_subnet_args


def get_pn_flops(_net_configs, res=224):
    print("Input net: {}".format(_net_configs))
    _supernet = ofa_net(net_id="ofa_proxyless_d234_e346_k357_w1.3", pretrained=True,
                           model_dir="../../saved_models/ofa_checkpoints/")
    ks, e, d = ofa_str_configs_to_subnet_args(_net_configs, 21,
                                              expected_prefix="mbconv2")
    _supernet.set_active_subnet(ks=ks, e=e, d=d)
    _subnet = _supernet.get_active_subnet(preserve_weight=True)
    return count_net_flops(_subnet, (1, 3, res, res)) / 1e6


def get_mbv3_flops(_net_configs, res=224):
    print("Input net: {}".format(_net_configs))
    _supernet = ofa_net(net_id="ofa_mbv3_d234_e346_k357_w1.2", pretrained=True,
                           model_dir="../../saved_models/ofa_checkpoints/")
    ks, e, d = ofa_str_configs_to_subnet_args(_net_configs, 20,
                                              expected_prefix="mbconv3")
    _supernet.set_active_subnet(ks=ks, e=e, d=d)
    _subnet = _supernet.get_active_subnet(preserve_weight=True)
    return count_net_flops(_subnet, (1, 3, res, res)) / 1e6


def get_resnet_flops(d, e, w, res=224):
    print("Input d: {}, e: {}, w: {}".format(d, e, w))
    _supernet = ofa_net(net_id="ofa_resnet50", pretrained=True,
                        model_dir="../../saved_models/ofa_checkpoints/")
    _supernet.set_active_subnet(d=d, e=e, w=w)
    _subnet = _supernet.get_active_subnet(preserve_weight=True)
    return count_net_flops(_subnet, (1, 3, res, res)) / 1e6


if __name__ == "__main__":
    # # PN Insight
    # print("{}\n".format(get_pn_flops([['mbconv2_e4_k7', 'mbconv2_e6_k5', 'mbconv2_e4_k7', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e4_k7', 'mbconv2_e6_k7', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e4_k7', 'mbconv2_e6_k5', 'mbconv2_e4_k5'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e6_k5', 'mbconv2_e6_k5', 'mbconv2_e4_k5'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e6_k7', 'mbconv2_e6_k7', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e6_k7']])))
    # print("{}\n".format(get_pn_flops([['mbconv2_e6_k7', 'mbconv2_e6_k3', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e4_k5', 'mbconv2_e6_k7', 'mbconv2_e4_k5', 'mbconv2_e4_k7'],
    #                                   ['mbconv2_e6_k7', 'mbconv2_e6_k3', 'mbconv2_e6_k5', 'mbconv2_e4_k7'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e6_k7', 'mbconv2_e6_k5', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e6_k5', 'mbconv2_e6_k7', 'mbconv2_e6_k3', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e4_k7']])))
    # print("{}\n".format(get_pn_flops([['mbconv2_e4_k7', 'mbconv2_e4_k7', 'mbconv2_e6_k5', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e6_k3', 'mbconv2_e6_k5', 'mbconv2_e6_k7', 'mbconv2_e4_k5'],
    #                                   ['mbconv2_e6_k7', 'mbconv2_e6_k3', 'mbconv2_e4_k5', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e6_k5', 'mbconv2_e6_k5', 'mbconv2_e4_k5'],
    #                                   ['mbconv2_e6_k3', 'mbconv2_e6_k5', 'mbconv2_e6_k7', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e6_k7']])))
    # print("{}\n".format(get_pn_flops([['mbconv2_e4_k7', 'mbconv2_e6_k7', 'mbconv2_e6_k3', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e6_k5', 'mbconv2_e4_k5', 'mbconv2_e6_k7', 'mbconv2_e4_k7'],
    #                                   ['mbconv2_e6_k7', 'mbconv2_e4_k5', 'mbconv2_e6_k7', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e6_k5', 'mbconv2_e4_k7', 'mbconv2_e6_k3', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e6_k5', 'mbconv2_e6_k3', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e6_k7']])))
    # print("{}\n".format(get_pn_flops([['mbconv2_e6_k7', 'mbconv2_e4_k7', 'mbconv2_e6_k5', 'mbconv2_e4_k7'],
    #                                   ['mbconv2_e4_k5', 'mbconv2_e6_k5', 'mbconv2_e4_k7', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e6_k7', 'mbconv2_e6_k5', 'mbconv2_e6_k5', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e6_k3', 'mbconv2_e6_k7', 'mbconv2_e4_k7', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e6_k5', 'mbconv2_e6_k5', 'mbconv2_e6_k3'],
    #                                   ['mbconv2_e4_k5']])))

    # # PN Full
    # print("{}\n".format(get_pn_flops([['mbconv2_e6_k7', 'mbconv2_e6_k5', 'mbconv2_e6_k5', 'mbconv2_e3_k5'],
    #                                   ['mbconv2_e6_k5', 'mbconv2_e3_k5', 'mbconv2_e4_k3', 'mbconv2_e6_k3'],
    #                                   ['mbconv2_e4_k5', 'mbconv2_e4_k5', 'mbconv2_e6_k3', 'mbconv2_e3_k5'],
    #                                   ['mbconv2_e4_k5', 'mbconv2_e6_k5', 'mbconv2_e4_k3', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e4_k3', 'mbconv2_e6_k5', 'mbconv2_e4_k7', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e4_k7']])))
    # print("{}\n".format(get_pn_flops([['mbconv2_e6_k5', 'mbconv2_e4_k5', 'mbconv2_e3_k7', 'mbconv2_e6_k3'],
    #                                   ['mbconv2_e6_k5', 'mbconv2_e6_k7', 'mbconv2_e3_k7', 'mbconv2_e4_k3'],
    #                                   ['mbconv2_e3_k7', 'mbconv2_e6_k7', 'mbconv2_e4_k3', 'mbconv2_e4_k3'],
    #                                   ['mbconv2_e6_k5', 'mbconv2_e4_k3', 'mbconv2_e6_k5', 'mbconv2_e4_k7'],
    #                                   ['mbconv2_e3_k3', 'mbconv2_e6_k5', 'mbconv2_e4_k7', 'mbconv2_e6_k3'],
    #                                   ['mbconv2_e4_k7']])))
    # print("{}\n".format(get_pn_flops([['mbconv2_e4_k5', 'mbconv2_e3_k5', 'mbconv2_e6_k7', 'mbconv2_e4_k3'],
    #                                   ['mbconv2_e3_k7', 'mbconv2_e6_k5', 'mbconv2_e6_k3'],
    #                                   ['mbconv2_e3_k7', 'mbconv2_e6_k7', 'mbconv2_e6_k7', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e6_k7', 'mbconv2_e3_k5', 'mbconv2_e6_k7', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e6_k3', 'mbconv2_e4_k7', 'mbconv2_e6_k5', 'mbconv2_e3_k7'],
    #                                   ['mbconv2_e3_k7']])))
    # print("{}\n".format(get_pn_flops([['mbconv2_e3_k3', 'mbconv2_e3_k7', 'mbconv2_e4_k5', 'mbconv2_e3_k7'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e3_k7', 'mbconv2_e3_k3', 'mbconv2_e4_k7'],
    #                                   ['mbconv2_e6_k7', 'mbconv2_e6_k5', 'mbconv2_e3_k3', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e6_k5', 'mbconv2_e6_k5', 'mbconv2_e3_k3', 'mbconv2_e4_k5'],
    #                                   ['mbconv2_e4_k5', 'mbconv2_e6_k5', 'mbconv2_e6_k5', 'mbconv2_e6_k3'],
    #                                   ['mbconv2_e4_k7']])))
    # print("{}\n".format(get_pn_flops([['mbconv2_e6_k3', 'mbconv2_e6_k3', 'mbconv2_e4_k3', 'mbconv2_e3_k3'],
    #                                   ['mbconv2_e6_k3', 'mbconv2_e6_k3', 'mbconv2_e3_k5', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e6_k3', 'mbconv2_e6_k7', 'mbconv2_e6_k7'],
    #                                   ['mbconv2_e4_k7', 'mbconv2_e3_k3', 'mbconv2_e6_k7', 'mbconv2_e6_k5'],
    #                                   ['mbconv2_e6_k3', 'mbconv2_e3_k7', 'mbconv2_e6_k7', 'mbconv2_e6_k3'],
    #                                   ['mbconv2_e6_k7']])))

    # # MBV3 Insight
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e6_k5', 'mbconv3_e6_k7', 'mbconv3_e6_k3', 'mbconv3_e6_k3'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e6_k7', 'mbconv3_e4_k7', 'mbconv3_e6_k7'],
    #                                     ['mbconv3_e6_k5', 'mbconv3_e6_k7', 'mbconv3_e4_k5', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e4_k5', 'mbconv3_e6_k5', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e6_k3', 'mbconv3_e6_k3', 'mbconv3_e6_k3', 'mbconv3_e6_k5']])))
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e6_k5', 'mbconv3_e6_k5', 'mbconv3_e6_k5', 'mbconv3_e6_k7'],
    #                                     ['mbconv3_e6_k5', 'mbconv3_e6_k5', 'mbconv3_e4_k7', 'mbconv3_e6_k3'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k3', 'mbconv3_e6_k7', 'mbconv3_e6_k3'],
    #                                     ['mbconv3_e6_k5', 'mbconv3_e4_k5', 'mbconv3_e6_k5', 'mbconv3_e6_k3'],
    #                                     ['mbconv3_e6_k5', 'mbconv3_e6_k3', 'mbconv3_e6_k3']])))
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e6_k3', 'mbconv3_e6_k3', 'mbconv3_e4_k5'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k7', 'mbconv3_e4_k5', 'mbconv3_e6_k3'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e4_k7', 'mbconv3_e4_k5', 'mbconv3_e4_k7'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k7', 'mbconv3_e6_k7', 'mbconv3_e6_k7'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k5', 'mbconv3_e6_k3', 'mbconv3_e6_k7']])))
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e6_k5', 'mbconv3_e4_k5', 'mbconv3_e4_k7', 'mbconv3_e6_k7'],
    #                                     ['mbconv3_e4_k5', 'mbconv3_e4_k5', 'mbconv3_e4_k7', 'mbconv3_e4_k5'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k7', 'mbconv3_e6_k3', 'mbconv3_e6_k7'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e6_k5', 'mbconv3_e6_k7', 'mbconv3_e6_k7'],
    #                                     ['mbconv3_e6_k5', 'mbconv3_e6_k7', 'mbconv3_e6_k5']])))
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e4_k5', 'mbconv3_e6_k7', 'mbconv3_e6_k3', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e6_k7', 'mbconv3_e6_k7', 'mbconv3_e6_k3'],
    #                                     ['mbconv3_e6_k3', 'mbconv3_e6_k7', 'mbconv3_e4_k7', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e6_k5', 'mbconv3_e6_k3', 'mbconv3_e6_k3', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e6_k7', 'mbconv3_e6_k3', 'mbconv3_e4_k5']])))

    # # MBV3 Full
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e3_k5', 'mbconv3_e3_k7', 'mbconv3_e4_k7'],
    #                                     ['mbconv3_e6_k5', 'mbconv3_e3_k3', 'mbconv3_e4_k3', 'mbconv3_e6_k7'],
    #                                     ['mbconv3_e4_k5', 'mbconv3_e4_k7', 'mbconv3_e6_k5', 'mbconv3_e4_k3'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k7', 'mbconv3_e6_k5', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k5', 'mbconv3_e6_k5', 'mbconv3_e6_k3']])))
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e3_k3', 'mbconv3_e6_k5', 'mbconv3_e3_k7', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e4_k7', 'mbconv3_e6_k3', 'mbconv3_e3_k3'],
    #                                     ['mbconv3_e4_k5', 'mbconv3_e3_k7', 'mbconv3_e6_k7', 'mbconv3_e3_k3'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e6_k5', 'mbconv3_e4_k5', 'mbconv3_e3_k5'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k5', 'mbconv3_e4_k3', 'mbconv3_e6_k3']])))
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e6_k3', 'mbconv3_e4_k7', 'mbconv3_e4_k3', 'mbconv3_e3_k7'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e4_k3', 'mbconv3_e3_k5', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e3_k3', 'mbconv3_e6_k5', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e6_k3', 'mbconv3_e4_k7', 'mbconv3_e3_k3'],
    #                                     ['mbconv3_e6_k3', 'mbconv3_e4_k5', 'mbconv3_e6_k3', 'mbconv3_e3_k7']])))
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e6_k3', 'mbconv3_e6_k7', 'mbconv3_e3_k7', 'mbconv3_e4_k5'],
    #                                     ['mbconv3_e3_k5', 'mbconv3_e4_k5', 'mbconv3_e6_k3', 'mbconv3_e4_k5'],
    #                                     ['mbconv3_e3_k7', 'mbconv3_e4_k3', 'mbconv3_e6_k3', 'mbconv3_e3_k7'],
    #                                     ['mbconv3_e4_k7', 'mbconv3_e6_k5', 'mbconv3_e6_k7', 'mbconv3_e6_k3'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k7', 'mbconv3_e4_k5', 'mbconv3_e4_k5']])))
    # print("{}\n".format(get_mbv3_flops([['mbconv3_e4_k3', 'mbconv3_e6_k7', 'mbconv3_e4_k5', 'mbconv3_e4_k7'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e4_k5'],
    #                                     ['mbconv3_e3_k7', 'mbconv3_e6_k5', 'mbconv3_e3_k5', 'mbconv3_e6_k5'],
    #                                     ['mbconv3_e6_k3', 'mbconv3_e6_k7', 'mbconv3_e3_k3', 'mbconv3_e6_k3'],
    #                                     ['mbconv3_e6_k7', 'mbconv3_e6_k5', 'mbconv3_e6_k7']])))

    # # ResNet Insight
    # print("{}\n".format(get_resnet_flops(d=[2, 2, 2, 2, 2],
    #                                      e=[0.35, 0.35, 0.25, 0.2, 0.35, 0.25, 0.35, 0.2, 0.35, 0.35, 0.2, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.25],
    #                                      w=[0, 2, 2, 1, 2, 2])))
    # print("{}\n".format(get_resnet_flops(d=[2, 2, 2, 2, 2],
    #                                      e=[0.25, 0.35, 0.25, 0.2, 0.35, 0.25, 0.35, 0.35, 0.25, 0.35, 0.35, 0.25, 0.35, 0.35, 0.35, 0.35, 0.25, 0.2],
    #                                      w=[1, 2, 2, 2, 2, 2])))
    # print("{}\n".format(get_resnet_flops(d=[2, 2, 2, 2, 2],
    #                                      e=[0.2, 0.2, 0.35, 0.35, 0.35, 0.2, 0.2, 0.25, 0.25, 0.35, 0.2, 0.35, 0.25, 0.2, 0.35, 0.35, 0.25, 0.2],
    #                                      w=[1, 2, 2, 2, 2, 2])))
    # print("{}\n".format(get_resnet_flops(d=[2, 2, 2, 2, 2],
    #                                      e=[0.35, 0.35, 0.25, 0.2, 0.35, 0.35, 0.35, 0.2, 0.2, 0.35, 0.35, 0.25, 0.2, 0.25, 0.35, 0.35, 0.35, 0.2],
    #                                      w=[0, 2, 2, 1, 2, 2])))
    # print("{}\n".format(get_resnet_flops(d=[2, 2, 2, 2, 2],
    #                                      e=[0.2, 0.2, 0.25, 0.2, 0.35, 0.25, 0.2, 0.35, 0.25, 0.35, 0.35, 0.25, 0.2, 0.35, 0.35, 0.35, 0.35, 0.25],
    #                                      w=[0, 2, 2, 2, 2, 2])))

    # # ResNet Full
    # print("{}\n".format(get_resnet_flops(d=[2, 2, 1, 2, 2],
    #                                      e=[0.25, 0.25, 0.25, 0.35, 0.35, 0.35, 0.25, 0.2, 0.2, 0.25, 0.35, 0.2, 0.35, 0.2, 0.35, 0.35, 0.2, 0.35],
    #                                      w=[2, 2, 1, 2, 2, 2])))
    # print("{}\n".format(get_resnet_flops(d=[2, 0, 2, 2, 1],
    #                                      e=[0.35, 0.2, 0.35, 0.25, 0.35, 0.25, 0.35, 0.25, 0.35, 0.35, 0.35, 0.25, 0.2, 0.2, 0.25, 0.25, 0.35, 0.2],
    #                                      w=[2, 2, 2, 2, 2, 2])))
    # print("{}\n".format(get_resnet_flops(d=[2, 2, 2, 1, 2],
    #                                      e=[0.35, 0.35, 0.2, 0.35, 0.35, 0.2, 0.35, 0.2, 0.35, 0.35, 0.25, 0.35, 0.2, 0.2, 0.2, 0.35, 0.35, 0.25],
    #                                      w=[2, 2, 2, 2, 2, 2])))
    # print("{}\n".format(get_resnet_flops(d=[0, 2, 2, 2, 2],
    #                                      e=[0.35, 0.35, 0.35, 0.35, 0.2, 0.35, 0.25, 0.25, 0.35, 0.35, 0.25, 0.25, 0.2, 0.25, 0.35, 0.2, 0.25, 0.35],
    #                                      w=[0, 2, 0, 2, 2, 2])))
    # print("{}\n".format(get_resnet_flops(d=[0, 2, 2, 2, 1],
    #                                      e=[0.25, 0.25, 0.2, 0.35, 0.25, 0.35, 0.35, 0.35, 0.2, 0.35, 0.35, 0.25, 0.35, 0.25, 0.35, 0.35, 0.25, 0.2],
    #                                      w=[2, 2, 2, 2, 2, 2])))
    print("done")
