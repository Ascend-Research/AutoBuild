# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.


import torch
from search.rm_search.params import *
from search.rm_search.ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from search.rm_search.ofa.imagenet_classification.run_manager import ImagenetRunConfig
from search.rm_search.ofa.imagenet_classification.run_manager import RunManager
from search.rm_search.ofa.model_zoo import ofa_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='The path of imagenet',
        type=str,
        default='~/ImageNet')
    parser.add_argument(
        '-g',
        '--gpu',
        help='The gpu(s) to use',
        type=str,
        default='all')
    parser.add_argument(
        '-b',
        '--batch-size',
        help='The batch on every device for validation',
        type=int,
        default=100)
    parser.add_argument(
        '-j',
        '--workers',
        help='Number of workers',
        type=int,
        default=20)
    parser.add_argument(
        '-n',
        '--net',
        metavar='OFANET',
        default='ofa_mbv3_d234_e346_k357_w1.2',
        choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2', 'ofa_proxyless_d234_e346_k357_w1.3',
                 'ofa_resnet50'],
        help='OFA networks')

    args = parser.parse_args()
    if args.gpu == 'all':
        device_list = range(torch.cuda.device_count())
        args.gpu = ','.join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.batch_size = args.batch_size * max(len(device_list), 1)

    # Set imagenet data path
    ImagenetDataProvider.DEFAULT_PATH = args.path

    model_dir = P_SEP.join([SAVED_MODELS_DIR, "ofa_checkpoints"])
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    ofa_network = ofa_net(args.net, pretrained=True, model_dir=model_dir)

    """ Randomly sample a sub-network, 
        you can also manually set the sub-network using: 
            ofa_network.set_active_subnet(ks=7, e=6, d=4) 
    """
    subnet_configs = ofa_network.sample_active_subnet()
    print("Sampled subnet configs: {}".format(subnet_configs))
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    n_net_weights = sum(p.numel() for p in subnet.parameters() if p.requires_grad)
    print("Num weights: {}".format(n_net_weights))
    print(subnet)

    """ Test sampled subnet 
    """
    ofa_runs_dir = P_SEP.join([LOGS_DIR, "ofa_runs"])
    if not os.path.isdir(ofa_runs_dir):
        os.mkdir(ofa_runs_dir)
    test_runs_dir = P_SEP.join([ofa_runs_dir, "supernet_test_runs"])
    if not os.path.isdir(test_runs_dir):
        os.mkdir(test_runs_dir)
    run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers,
                                   valid_size=50000)
    run_manager = RunManager(test_runs_dir, subnet, run_config, init=False)
    # assign image size: 128, 132, ..., 224
    run_config.data_provider.assign_active_img_size(224)
    run_manager.reset_running_statistics(net=subnet)

    print('Test random subnet:')
    print(subnet.module_str)

    loss, (top1, top5) = run_manager.validate(net=subnet, is_test=False)
    print('Valid results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))
    loss, (top1, top5) = run_manager.validate(net=subnet, is_test=True)
    print('Test results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))

    from search.rm_search.ofa.my_utils import ofa_str_configs_to_subnet_args
    _net_configs = [['mbconv3_e6_k5', 'mbconv3_e6_k5', 'mbconv3_e6_k3', 'mbconv3_e6_k7'],
                    ['mbconv3_e4_k7', 'mbconv3_e6_k5', 'mbconv3_e6_k7', 'mbconv3_e6_k5'],
                    ['mbconv3_e6_k7', 'mbconv3_e4_k7', 'mbconv3_e6_k5', 'mbconv3_e4_k5'],
                    ['mbconv3_e6_k7', 'mbconv3_e6_k7', 'mbconv3_e6_k5', 'mbconv3_e6_k3'],
                    ['mbconv3_e6_k3', 'mbconv3_e6_k3', 'mbconv3_e6_k5']]
    ks, e, d = ofa_str_configs_to_subnet_args(_net_configs, max_n_net_blocks=20)
    print("ks: {}".format(ks))
    print("e: {}".format(e))
    print("d: {}".format(d))
    ofa_network.set_active_subnet(ks, e, d)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    run_manager.reset_running_statistics(net=subnet)
    loss, (top1, top5) = run_manager.validate(net=subnet, is_test=True)
    print('Custom test results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))


if __name__ == "__main__":

    main()

    print("done")
