from tqdm import tqdm
import sys
import torch as t
from search.rm_search.params import *
import search.rm_search.utils.model_utils as m_util
from search.rm_search.utils.model_utils import set_random_seed
from search.rm_search.utils.misc_utils import RunningStatMeter
from search.rm_search.model_helpers import BookKeeper
from search.rm_search.ofa.evaluators import OFANetAccLatEvaluator
from search.rm_search.ea.model_rm_custom import rm_pareto_search, ArchParetoFront
from search.rm_search.ofa_profile.arch_gen import sample_ofa_pn_stage_configs
from search.rm_search.ofa_profile.arch_mutator import OFAStageWholeMutator
from search.rm_search.ofa_profile.search_utils import get_pn_stage_level_mutate_probs
from search.rm_search.constants import DIR_TO_APPEND
import pickle

sys.path.append(DIR_TO_APPEND)
from iterators.ofa_manager import PNManager


"""
OFA-PN random mutation pareto front search
"""


def prepare_local_params(parser):
    parser.add_argument("-model_name", required=False, type=str,
                        default="ofa_pn_rm_pareto")
    parser.add_argument("-resolution", required=False, type=int,
                        default=224)
    parser.add_argument("-imagenet_data_dir", required=False, type=str,
                        default="~/ImageNet")
    parser.add_argument("-evaluator_runs_dir", required=False, type=str,
                        default=P_SEP.join([LOGS_DIR, "ofa_runs"]))
    parser.add_argument("-supernet_checkpoint_dir", required=False, type=str,
                        default="../../.torch/ofa_nets/")
    parser.add_argument("-supernet_name", required=False, type=str,
                        default="ofa_proxyless_d234_e346_k357_w1.3")
    parser.add_argument("-lat_predictor_type", required=False, type=str,
                        default="custom")
    parser.add_argument("-lat_predictor_checkpoint", required=False, type=str,
                        default=P_SEP.join(["../../models/Latency",
                                            "ofa_pn_op_graph_npu_lat_predictor_best.pt"]))
    parser.add_argument("-num_pareto_runs", required=False, type=int,
                        default=1)
    parser.add_argument("-batch_size", required=False, type=int,
                        default=128)
    parser.add_argument("-random_init_set_size", required=False, type=int,
                        default=50)
    parser.add_argument("-num_iterations", required=False, type=int,
                        default=4)
    parser.add_argument("-eval_budget", required=False, type=int,
                        default=50)
    parser.add_argument("-mutate_prob_type", required=False, type=str,
                        default="uniform")
    parser.add_argument("-stage_mutate_prob", required=False, type=float,
                        default=0.5)
    #parser.add_argument("-stage_block_count_type", required=False, type=str,
    #                    default="default")
    #parser.add_argument("-space_id", required=False, type=str,
    #                    default=None)
    parser.add_argument("-mutation_depth", required=False, type=int,
                        default=1)
    parser.add_argument("-completed_iter", required=False, type=int,
                        default=0)
    parser.add_argument("-pareto_init_checkpoint", required=False, type=str,
                        default=None)
    parser.add_argument("-fast", action="store_true", default=False)
    parser.add_argument("-metric", required=False, default="2080Ti_predicted")
    parser.add_argument("-unit_corpus", type=str, nargs="+", default=None)
    return parser.parse_args()


def main(params):
    params.model_name += "_{}_r{}_seed{}".format(params.metric, params.resolution, params.seed)
    if not os.path.isdir(params.evaluator_runs_dir):
        os.mkdir(params.evaluator_runs_dir)
    params.evaluator_runs_dir = P_SEP.join([params.evaluator_runs_dir,
                                            params.model_name])
    if not os.path.isdir(params.evaluator_runs_dir):
        os.mkdir(params.evaluator_runs_dir)

    book_keeper = BookKeeper(log_file_name=params.model_name + ".txt",
                             model_name=params.model_name,
                             saved_models_dir=params.saved_models_dir,
                             init_eval_perf=float("-inf"), eval_perf_comp_func=lambda old, new: new > old,
                             saved_model_file=params.saved_model_file,
                             logs_dir=params.logs_dir)
    book_keeper.log("Params: {}".format(params), verbose=False)
    set_random_seed(params.seed, log_f=book_keeper.log)

    # Load lat predictor
    lat_predictor = PNManager(params.metric)

    # Prepare evaluator
    book_keeper.log("Specified ImageNet data dir: {}".format(params.imagenet_data_dir))
    book_keeper.log("Specified evaluator runs dir: {}".format(params.evaluator_runs_dir))
    book_keeper.log("Specified supernet checkpoint dir: {}".format(params.supernet_checkpoint_dir))
    book_keeper.log("Specified supernet name: {}".format(params.supernet_name))
    book_keeper.log("Specified resolution: {}".format(params.resolution))
    evaluator = OFANetAccLatEvaluator(data_path=params.imagenet_data_dir,
                                      runs_dir=params.evaluator_runs_dir,
                                      net_name=params.supernet_name,
                                      model_dir=params.supernet_checkpoint_dir,
                                      max_n_net_blocks=21, resolution=params.resolution,
                                      block_prefix="mbconv2", ext_lat_predictor=lat_predictor,
                                      batch_size=params.batch_size, use_cached_loader=params.fast,
                                      metric=params.metric)

    # # Dummy for debugging
    # from model_src.model_helpers import RandomEvaluator
    # evaluator = RandomEvaluator()

    # Select stage-level candidates
    #stage_level_cands = get_pn_stage_level_cands(params.space_id,
    #                                             log_f=book_keeper.log)

    # Select stage-level block counts
    #max_stage_block_counts = get_pn_max_n_blocks_per_stage(params.stage_block_count_type,
    #                                                       log_f=book_keeper.log)

    # Select stage-level mutation probs
    stage_mutate_probs = get_pn_stage_level_mutate_probs(params.mutate_prob_type,
                                                         uniform_prob=params.stage_mutate_prob,
                                                         log_f=book_keeper.log)
    
    if params.unit_corpus is not None:
        for i, unit_corpus in enumerate(params.unit_corpus):
            with open(unit_corpus, "rb") as f:
                if i == 0:
                    stage_units = pickle.load(f)
                else:
                    additional_units = pickle.load(f)
                    for j, au in enumerate(additional_units):
                        stage_units[j] += au
    else:
        from iterators.pn import gen_sgs
        subgraph_list_u14 = gen_sgs(units=[1])
        sgs14 = [x['config'] for x in subgraph_list_u14]
        subgraph_list_u5 = gen_sgs(units=[5])
        sgs5 = [x['config'] for x in subgraph_list_u5]
        stage_units = [sgs14, sgs14, sgs14, sgs14, sgs5]

    # Get random init set
    init_net2perfs = {}
    if params.pareto_init_checkpoint is None:
        unique_net_ids = set()
        acc_meter = RunningStatMeter()
        lat_meter = RunningStatMeter()
        bar = tqdm(total=params.random_init_set_size,
                   desc="Sampling init nets", ascii=True)
        while len(init_net2perfs) < params.random_init_set_size:
            net_configs = sample_ofa_pn_stage_configs(stage_units)
            if str(net_configs) in unique_net_ids:
                continue
            unique_net_ids.add(str(net_configs))
            acc, lat = evaluator.get_perf_values(net_configs)
            init_net2perfs[str(net_configs)] = (net_configs, (acc, lat))
            acc_meter.update(acc)
            lat_meter.update(lat)
            bar.desc = "Sampling init nets, " \
                       "curr avg acc={}, lat={}".format(round(acc_meter.avg, 5),
                                                        round(lat_meter.avg, 3))
            bar.update(1)
        bar.close()

    pareto_front = ArchParetoFront(init_net2perfs, [False, True],
                                   num_pareto_runs=params.num_pareto_runs,
                                   ignore_init_update=params.pareto_init_checkpoint is not None,
                                   log_f=book_keeper.log)

    if params.pareto_init_checkpoint is not None:
        book_keeper.log("Loading pareto checkpoint: {}".format(params.pareto_init_checkpoint))
        book_keeper.load_state_dict_checkpoint(pareto_front,
                                               params.pareto_init_checkpoint)

    # Prepare mutator
    mutator = OFAStageWholeMutator(stage_level_candidates=stage_units,
                                   stage_mutate_probs=stage_mutate_probs)

    # Search
    book_keeper.log("Search iterations: {}".format(params.num_iterations))
    book_keeper.log("Completed iterations: {}".format(params.completed_iter))
    book_keeper.log("Budget per iter: {}".format(params.eval_budget))
    book_keeper.log("Mutation depth: {}".format(params.mutation_depth))
    try:
        rm_pareto_search(mutator=mutator,
                         num_iterations=params.num_iterations,
                         budget_per_iter=params.eval_budget,
                         pareto_front=pareto_front, evaluator=evaluator,
                         book_keeper=book_keeper,
                         completed_iter=params.completed_iter,
                         mutation_depth=params.mutation_depth)
    except KeyboardInterrupt:
        book_keeper.log("Search interrupted")

    # Generate some simple info for the top-most pareto
    book_keeper.log("Showing top-most pareto")
    book_keeper.log("Arch|Acc|Lat")
    archs_w_perf = pareto_front.get_pareto_archs_with_perf(num_runs=1)
    for arch, perf in archs_w_perf:
        acc, lat = perf
        book_keeper.log("{}|{}|{}".format(str(arch), acc, lat))


if __name__ == "__main__":
    _parser = prepare_global_params()
    _args = prepare_local_params(_parser)
    m_util.DEVICE_STR_OVERRIDE = _args.device_str
    main(_args)
    print("done")
