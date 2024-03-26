from tqdm import tqdm
from search.rm_search.params import *
import search.rm_search.utils.model_utils as m_util
from search.rm_search.utils.model_utils import set_random_seed
from search.rm_search.utils.misc_utils import RunningStatMeter
from search.rm_search.model_helpers import BookKeeper
from search.rm_search.ofa_profile.constants import *
from search.rm_search.ofa.evaluators import OFANetAccLatEvaluator
from search.rm_search.ea.model_rm_custom import rm_cons_acc_search, TopKAccSet
from search.rm_search.ofa_profile.arch_gen import sample_ofa_pn_configs
from search.rm_search.ofa_profile.arch_mutator import OFAStageLevelMutator
from search.rm_search.ofa_profile.search_utils import get_pn_stage_level_cands, get_pn_stage_level_mutate_probs, \
    get_pn_search_lat_predictor, get_pn_max_n_blocks_per_stage


"""
OFA-PN random mutation constrained/un-constrained acc search
"""


def prepare_local_params(parser):
    parser.add_argument("-model_name", required=False, type=str,
                        default="ofa_pn_rm_cons_acc")
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
    parser.add_argument("-max_cons_score", required=False, type=float,
                        default=None)
    parser.add_argument("-top_k", required=False, type=int,
                        default=10)
    parser.add_argument("-batch_size", required=False, type=int,
                        default=128)
    parser.add_argument("-random_init_set_size", required=False, type=int,
                        default=50)
    parser.add_argument("-num_iterations", required=False, type=int,
                        default=10)
    parser.add_argument("-eval_budget", required=False, type=int,
                        default=20)
    parser.add_argument("-mutate_prob_type", required=False, type=str,
                        default="uniform")
    parser.add_argument("-stage_mutate_prob", required=False, type=float,
                        default=0.5)
    parser.add_argument("-stage_block_count_type", required=False, type=str,
                        default="default")
    parser.add_argument("-space_id", required=False, type=str,
                        default=None)
    parser.add_argument("-mutation_depth", required=False, type=int,
                        default=1)
    parser.add_argument("-completed_iter", required=False, type=int,
                        default=0)
    parser.add_argument("-top_set_init_checkpoint", required=False, type=str,
                        default=None)
    parser.add_argument("-fast", action="store_true", default=False)
    return parser.parse_args()


def main(params):
    params.model_name += "_r{}_seed{}".format(params.resolution, params.seed)
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
    lat_predictor = None #get_pn_search_lat_predictor(params.lat_predictor_type,
                         #                       params.resolution,
                         #                       lat_predictor_checkpoint=params.lat_predictor_checkpoint,
                         #                       log_f=book_keeper.log)

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
                                      block_prefix="mbconv2",
                                      dummy_lat=0.0, ext_lat_predictor=None,  # De-activate built-in lat predictor
                                      batch_size=params.batch_size, use_cached_loader=params.fast)

    # # Dummy for debugging
    # from model_src.model_helpers import RandomEvaluator
    # evaluator = RandomEvaluator()

    target_nets = PN_CONS_ACC_SEARCH_TARGET_NETS
    book_keeper.log("Specified {} target net(s)".format(len(PN_CONS_ACC_SEARCH_TARGET_NETS)))
    for c in PN_CONS_ACC_SEARCH_TARGET_NETS:
        book_keeper.log("Target net: {}".format(str(c)))

    # Set max cons score
    if params.max_cons_score is not None:
        max_cons_score = params.max_cons_score
        book_keeper.log("Using specified max cons score: {}".format(params.max_cons_score))
    elif len(PN_CONS_ACC_SEARCH_TARGET_NETS) > 0:
        max_cons_score = min([lat_predictor(_cfg) for _cfg in PN_CONS_ACC_SEARCH_TARGET_NETS])
        book_keeper.log("Using target nets min lat as max cons score: {}".format(max_cons_score))
    else:
        max_cons_score = None
        book_keeper.log("No max cons score")

    # If is unconstrained search, use this dummy cons_predictor
    if max_cons_score is None:
        book_keeper.log("Running un-constrained search!")
        cons_predictor = lambda _a : 0.0
        max_cons_score = 1.0
    else:
        book_keeper.log("Running lat-constrained search!")
        book_keeper.log("Max cons score: {}".format(max_cons_score))
        cons_predictor = lat_predictor

    # Select stage-level candidates
    stage_level_cands = get_pn_stage_level_cands(params.space_id,
                                                 log_f=book_keeper.log)

    # Select stage-level block counts
    max_stage_block_counts = get_pn_max_n_blocks_per_stage(params.stage_block_count_type,
                                                           log_f=book_keeper.log)

    # Select stage-level mutation probs
    stage_mutate_probs = get_pn_stage_level_mutate_probs(params.mutate_prob_type,
                                                         uniform_prob=params.stage_mutate_prob,
                                                         log_f=book_keeper.log)

    # Get random init set
    init_net2acc = {}
    if params.top_set_init_checkpoint is None:
        unique_net_ids = set()
        acc_meter = RunningStatMeter()
        bar = tqdm(total=params.random_init_set_size,
                   desc="Sampling init nets", ascii=True)
        while len(init_net2acc) < params.random_init_set_size:
            if len(init_net2acc) < len(target_nets):
                net_configs = target_nets[len(init_net2acc)]
            else:
                net_configs = sample_ofa_pn_configs(stage_n_blocks=max_stage_block_counts,
                                                    stage_level_candidates=stage_level_cands)
            if str(net_configs) in unique_net_ids:
                continue
            unique_net_ids.add(str(net_configs))

            cons_score = cons_predictor(net_configs)

            if len(init_net2acc) < len(target_nets) and \
                    params.max_cons_score is not None and \
                    cons_score - max_cons_score > 1e-5:
                # Special treatment for target net
                book_keeper.log("Overriding target net cons score from {} to {}".format(cons_score,
                                                                                        max_cons_score))
                cons_score = params.max_cons_score

            if cons_score - max_cons_score > 1e-5:
                continue

            acc, _ = evaluator.get_perf_values(net_configs)
            init_net2acc[str(net_configs)] = (net_configs, acc)
            acc_meter.update(acc)
            bar.desc = "Sampling init nets, " \
                       "curr avg acc={}".format(round(acc_meter.avg, 5))
            bar.update(1)
        bar.close()

    acc_set = TopKAccSet(init_net2acc, params.top_k,
                         ignore_init_update=params.top_set_init_checkpoint is not None,
                         log_f=book_keeper.log)

    if params.top_set_init_checkpoint is not None:
        book_keeper.log("Loading top arch set checkpoint: {}".format(params.top_set_init_checkpoint))
        book_keeper.load_state_dict_checkpoint(acc_set,
                                               params.top_set_init_checkpoint)

    # Prepare mutator
    mutator = OFAStageLevelMutator(min_stage_block_counts=(2,2,2,2,2,1),
                                   max_stage_block_counts=max_stage_block_counts,
                                   stage_level_candidates=stage_level_cands,
                                   stage_mutate_probs=stage_mutate_probs)

    # Search
    book_keeper.log("Search iterations: {}".format(params.num_iterations))
    book_keeper.log("Completed iterations: {}".format(params.completed_iter))
    book_keeper.log("Budget per iter: {}".format(params.eval_budget))
    book_keeper.log("Mutation depth: {}".format(params.mutation_depth))
    try:
        rm_cons_acc_search(mutator=mutator,
                           num_iterations=params.num_iterations,
                           budget_per_iter=params.eval_budget,
                           acc_set=acc_set, evaluator=evaluator,
                           max_cons_score=max_cons_score,
                           cons_predictor=cons_predictor,
                           book_keeper=book_keeper,
                           completed_iter=params.completed_iter,
                           mutation_depth=params.mutation_depth)
    except KeyboardInterrupt:
        book_keeper.log("Search interrupted")

    # Generate some simple info for the top archs
    book_keeper.log("Showing top-10 archs")
    book_keeper.log("Arch|Acc")
    archs_w_acc = acc_set.get_top_arch_with_acc()
    for arch, acc in archs_w_acc[:10]:
        book_keeper.log("{}|{}".format(str(arch), acc))


if __name__ == "__main__":
    _parser = prepare_global_params()
    _args = prepare_local_params(_parser)
    m_util.DEVICE_STR_OVERRIDE = _args.device_str
    main(_args)
    print("done")
