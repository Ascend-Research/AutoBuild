import pickle
from search.rm_search.params import *
import search.rm_search.utils.model_utils as m_util
from search.rm_search.utils.model_utils import set_random_seed
from search.rm_search.model_helpers import BookKeeper
from search.rm_search.ofa_profile.arch_gpu_cpu_lat import measure_lats
from search.rm_search.ofa_profile.networks import OFAMbv3Net, OFAProxylessNet
from search.rm_search.ofa_profile.arch_gen import gen_random_ofa_mbv3_configs, gen_random_ofa_pn_configs


"""
Randomly samples networks in the OFA-PN/MBV3 space and get their GPU or CPU latency
"""


def prepare_local_params(parser):
    parser.add_argument("-model_name", required=False, type=str,
                        default="ofa_{}_{}_lat_labelling")
    parser.add_argument("-sub_space", required=False, type=str,
                        default="mbv3")
    parser.add_argument("-device", required=False, type=str,
                        default="cpu")
    parser.add_argument("-samples_file", required=False, type=str,
                        default=P_SEP.join([CACHE_DIR, "ofa_{}_100k_random_net_configs.pkl"]))
    parser.add_argument("-output_file", required=False, type=str,
                        default=P_SEP.join([CACHE_DIR, "ofa_{}_{}_r{}_lat_data.pkl"]))
    parser.add_argument("-resolution", required=False, type=int,
                        default=224)
    parser.add_argument("-start_idx", required=False, type=int,
                        default=0)
    parser.add_argument("-end_idx", required=False, type=int,
                        default=50000)
    parser.add_argument("-n_nets", required=False, type=int,
                        default=100000)
    return parser.parse_args()


def main(params):
    params.model_name = params.model_name.format(params.sub_space, params.device)
    params.samples_file = params.samples_file.format(params.sub_space)
    params.output_file = params.output_file.format(params.sub_space, params.device, params.resolution)
    book_keeper = BookKeeper(log_file_name=params.model_name + ".txt",
                             model_name=params.model_name,
                             saved_models_dir=params.saved_models_dir,
                             init_eval_perf=0, eval_perf_comp_func=lambda old, new: new > old,
                             saved_model_file=params.saved_model_file,
                             logs_dir=params.logs_dir)
    book_keeper.log("Params: {}".format(params), verbose=False)
    set_random_seed(params.seed, log_f=book_keeper.log)
    book_keeper.log("Specified resolution: {}".format(params.resolution))
    book_keeper.log("Specified output file: {}".format(params.output_file))
    book_keeper.log("Specified start idx: {}".format(params.start_idx))
    book_keeper.log("Specified end idx: {}".format(params.end_idx))

    if "mbv3" in params.model_name:
        book_keeper.log("Using MobileNetV3 space")
        nets = gen_random_ofa_mbv3_configs(params.samples_file, params.n_nets,
                                           log_f=book_keeper.log)
    else:
        book_keeper.log("Using ProxylessNAS space")
        nets = gen_random_ofa_pn_configs(params.samples_file, params.n_nets,
                                         log_f=book_keeper.log)

    data = measure_lats(nets,
                        start_idx=params.start_idx, end_idx=params.end_idx,
                        net_func=OFAMbv3Net if "mbv3" in params.model_name else OFAProxylessNet,
                        dev="cuda" if params.device=="gpu" else "cpu",
                        unit="ms" if params.device=="gpu" else "s",
                        H=params.resolution, W=params.resolution)
    book_keeper.log("Collected {} data".format(len(data)))
    book_keeper.log("Writing data to {}".format(params.output_file))
    with open(params.output_file, "wb") as f:
        pickle.dump(data, f, protocol=4)


if __name__ == "__main__":
    _parser = prepare_global_params()
    _args = prepare_local_params(_parser)
    m_util.DEVICE_STR_OVERRIDE = _args.device_str
    main(_args)
    print("done")
