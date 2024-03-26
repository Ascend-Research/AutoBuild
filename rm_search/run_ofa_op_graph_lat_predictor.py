import pickle
import random
from search.rm_search.params import *
import search.rm_search.utils.model_utils as m_util
from search.rm_search.utils.model_utils import set_random_seed
from search.rm_search.model_helpers import BookKeeper
from search.rm_search.predictor.model_perf_predictor import *
from search.rm_search.predictor.dataloaders import RNNShapeRegressDataLoader
from search.rm_search.ofa_profile.arch_gen import get_ofa_pn_net_idx_shape_feat, \
    get_ofa_mbv3_net_idx_shape_feat, PN_OP2IDX, MBV3_OP2IDX


"""
End-to-end latency predictor trainer for the OFA-PN/MBV3 space
The input to the predictor is a single-path block graph of a macro net
"""


def prepare_local_params(parser):
    parser.add_argument("-model_name", required=False, type=str,
                        default="ofa_{}_op_graph_{}_lat_predictor")
    parser.add_argument("-sub_space", required=False, type=str,
                        default="mbv3")
    parser.add_argument("-lat_device", required=False, type=str,
                        default="npu")
    parser.add_argument("-lat_key", required=False, type=str,
                        default="overall npu lat")
    parser.add_argument("-lat_truth_max", required=False, type=float,
                        default=None)
    parser.add_argument("-data_file", required=False, type=str,
                        default=P_SEP.join([CACHE_DIR, "ofa_{}_npu_lat_data.pkl"]))
    parser.add_argument("-train_dev_data_size", required=False, type=int,
                        default=14500)
    parser.add_argument("-dev_data_size", required=False, type=int,
                        default=500)
    parser.add_argument("-test_data_size", required=False, type=int,
                        default=500)
    parser.add_argument("-epochs", required=False, type=int,
                        default=200)
    parser.add_argument("-batch_size", required=False, type=int,
                        default=16)
    parser.add_argument("-initial_lr", required=False, type=float,
                        default=0.001)
    parser.add_argument("-in_channels", help="", type=int,
                        default=128, required=False)
    parser.add_argument("-hidden_size", help="", type=int,
                        default=512, required=False)
    parser.add_argument("-num_layers", help="", type=int,
                        default=1, required=False)
    parser.add_argument("-dropout_prob", help="", type=float,
                        default=0.5, required=False)
    parser.add_argument("-add_output_activ", action="store_true", required=False,
                        default=False)
    return parser.parse_args()


def main(params):
    params.model_name = params.model_name.format(params.sub_space, params.lat_device)
    params.data_file = params.data_file.format(params.sub_space)
    book_keeper = BookKeeper(log_file_name=params.model_name + ".txt",
                             model_name=params.model_name,
                             saved_models_dir=params.saved_models_dir,
                             init_eval_perf=float("inf"), eval_perf_comp_func=lambda old, new: new < old,
                             saved_model_file=params.saved_model_file,
                             logs_dir=params.logs_dir)
    book_keeper.log("Params: {}".format(params), verbose=False)
    book_keeper.log("Specified input data file: {}".format(params.data_file))
    with open(params.data_file, "rb") as f:
        data = pickle.load(f)
    book_keeper.log("Loaded {} data instances".format(len(data)))
    set_random_seed(params.seed, log_f=book_keeper.log)
    train_dev_data = data[:params.train_dev_data_size]
    data = data[params.train_dev_data_size:]
    test_data = data[:params.test_data_size]
    random.shuffle(train_dev_data)

    # Take only what I need from the data list
    _train_dev_data, _test_data = [], []
    for data_dict in train_dev_data:
        net_configs = data_dict["net"]
        resolution = data_dict["resolution"]
        lat = data_dict[params.lat_key]
        _train_dev_data.append((net_configs, resolution, lat))
    for data_dict in test_data:
        net_configs = data_dict["net"]
        resolution = data_dict["resolution"]
        lat = data_dict[params.lat_key]
        _test_data.append((net_configs, resolution, lat))
    train_dev_data = _train_dev_data
    test_data = _test_data
    dev_data = train_dev_data[:params.dev_data_size]
    train_data = train_dev_data[params.dev_data_size:]
    book_keeper.log("Train size: {}".format(len(train_data)))
    book_keeper.log("Dev size: {}".format(len(dev_data)))
    book_keeper.log("Test size: {}".format(len(test_data)))

    # Sub-space select
    if params.sub_space == "mbv3":
        feat_func = get_ofa_mbv3_net_idx_shape_feat
        op2idx = MBV3_OP2IDX
    elif params.sub_space == "pn":
        feat_func = get_ofa_pn_net_idx_shape_feat
        op2idx = PN_OP2IDX
    else:
        raise ValueError("Unknown subspace: {}".format(params.sub_space))

    # Build op graph
    train_g_data = []
    dev_g_data = []
    test_g_data = []
    for configs, res, tgt in train_data:
        op_inds, shapes = feat_func(configs,
                                    H=res, W=res, H_max=224, W_max=224,
                                    log_f=lambda _m:_m)
        train_g_data.append([op_inds, shapes, tgt])
    for configs, res, tgt in dev_data:
        op_inds, shapes = feat_func(configs,
                                    H=res, W=res, H_max=224, W_max=224,
                                    log_f=lambda _m:_m)
        dev_g_data.append([op_inds, shapes, tgt])
    for configs, res, tgt in test_data:
        op_inds, shapes = feat_func(configs,
                                    H=res, W=res, H_max=224, W_max=224,
                                    log_f=lambda _m:_m)
        test_g_data.append([op_inds, shapes, tgt])
    assert len(train_g_data) == len(train_data)
    assert len(dev_g_data) == len(dev_data)
    assert len(test_g_data) == len(test_data)

    node_meter = RunningStatMeter()
    tgt_meter = RunningStatMeter()
    for op_list, _, _ in train_g_data + dev_g_data + test_g_data:
        node_meter.update(len(op_list))
    for _, _, tgt in train_g_data + dev_g_data:
        tgt_meter.update(tgt)
    book_keeper.log("Max num nodes: {}".format(node_meter.max))
    book_keeper.log("Min num nodes: {}".format(node_meter.min))
    book_keeper.log("Avg num nodes: {}".format(node_meter.avg))
    book_keeper.log("Max lat: {}".format(tgt_meter.max))
    book_keeper.log("Min lat: {}".format(tgt_meter.min))
    book_keeper.log("Avg lat: {}".format(tgt_meter.avg))

    # Normalize tgt
    if params.lat_truth_max is not None:
        lat_truth_max = min(params.lat_truth_max, tgt_meter.max)
    else:
        lat_truth_max = tgt_meter.max
    book_keeper.log("Target normalization constant: {}".format(lat_truth_max))
    for t in train_g_data:
        t[-1] = min(t[-1] / lat_truth_max, 1.0)
        assert 0 <= t[-1] <= 1
    for t in dev_g_data:
        t[-1] = min(t[-1] / lat_truth_max, 1.0)
        assert 0 <= t[-1] <= 1
    for t in test_g_data:
        t[-1] = min(t[-1] / lat_truth_max, 1.0)
        assert 0 <= t[-1] <= 1

    train_loader = RNNShapeRegressDataLoader(params.batch_size, train_g_data)
    dev_loader = RNNShapeRegressDataLoader(params.batch_size, dev_g_data)
    test_loader = RNNShapeRegressDataLoader(params.batch_size, test_g_data)
    book_keeper.log(
        "{} overlap(s) between train/dev loaders".format(train_loader.get_overlapping_data_count(dev_loader)))
    book_keeper.log(
        "{} overlap(s) between train/test loaders".format(train_loader.get_overlapping_data_count(test_loader)))
    book_keeper.log(
        "{} overlap(s) between dev/test loaders".format(dev_loader.get_overlapping_data_count(test_loader)))

    book_keeper.log("Initializing {}".format(params.model_name))
    model = make_rnn_shape_regressor(n_unique_labels=len(op2idx),
                                     out_embed_size=params.in_channels,
                                     shape_embed_size=16, n_shape_vals=6,
                                     hidden_size=params.hidden_size,
                                     activ=torch.nn.Sigmoid() if params.add_output_activ else None,
                                     n_layers=params.num_layers,
                                     dropout_prob=params.dropout_prob)
    perf_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.initial_lr)

    book_keeper.log(model)
    book_keeper.log("Model name: {}".format(params.model_name))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    book_keeper.log("Number of trainable parameters: {}".format(n_params))

    def _batch_fwd_func(_model, _batch):
        # Define how a batch is handled by the model
        node_feature_tsr = _batch[DK_BATCH_NODE_FEATURE_TSR]
        shape_tsr = _batch[DK_BATCH_NODE_SHAPE_TSR]
        return _model(node_feature_tsr, shape_tsr)

    book_keeper.log("Training for {} epochs".format(params.epochs))
    try:
        train_predictor(_batch_fwd_func, model, train_loader, perf_criterion, optimizer,
                        book_keeper, num_epochs=params.epochs,
                        max_gradient_norm=params.max_gradient_norm,
                        dev_loader=dev_loader if params.dev_data_size > 0 else None)
    except KeyboardInterrupt:
        book_keeper.log("Training interrupted")

    book_keeper.report_curr_best()
    book_keeper.load_model_checkpoint(model, allow_silent_fail=True, skip_eval_perfs=True,
                                      checkpoint_file=P_SEP.join([book_keeper.saved_models_dir,
                                                                  params.model_name + "_best.pt"]))

    with torch.no_grad():
        model.eval()
        run_predictor_epoch(_batch_fwd_func, model, test_loader, perf_criterion, None, book_keeper,
                            desc="Test", report_metrics=True)
        run_predictor_demo(_batch_fwd_func, model, test_loader, log_f=book_keeper.log,
                           normalize_constant=lat_truth_max, n_batches=5)


if __name__ == "__main__":
    _parser = prepare_global_params()
    _args = prepare_local_params(_parser)
    m_util.DEVICE_STR_OVERRIDE = _args.device_str
    main(_args)
    print("done")
