from params import *
import argparse
import torch as t
import pickle
from ge_utils.data_loader import get_regress_train_test_data, make_dataloader, boost_train_data
from ge_utils.data_loader import graph_regressor_batch_fwd
from ge_utils.measure_feat_importance import measure_feat_importance
from onnx_ir.trainer import train_regressor
from ge_utils.model import make_predictor
from ge_utils.diff_srcc import MixedLoss as LTRLoss
from ge_utils.misc_utils import set_random_seed
from ge_utils.metrics import compute_predictor_test_stats_ltr, compute_predictor_test_stats
from ge_utils.label_eq import format_target
from ge_utils.subgraph_moments import calc_moments_training_data
import os
from joblib import delayed, Parallel
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


def train_predictor(params, fold):
    set_random_seed(params.seed)
    train_data, test_data, _ = get_regress_train_test_data(caches=params.families, format=params.format, train_ratio=0.8, seed=params.seed, label=params.target, fold=fold)

    if params.format == "custom":
        moment_tsr = calc_moments_training_data(train_data, params.families)
    else:
        moment_tsr = None

    from ge_utils.acenas_loss import RelevanceCalculator
    rel_cal = RelevanceCalculator.from_data([x[-1] for x in train_data])
    train_data = [[t[0], rel_cal(t[-1])] for t in train_data]
    test_data = [[t[0], rel_cal(t[-1])] for t in test_data]
    xmu, xsig = 0, 1

    if params.boost:
        train_loader = boost_train_data(train_data)

    train_loader = make_dataloader(train_data, format=params.format, batch_size=params.batch_size, shuffle=True)
    test_loader = make_dataloader(test_data, format=params.format, batch_size=params.batch_size, shuffle=False)

    predictor = make_predictor(gnn_dim=params.gnn_dim,
                            gnn_type=params.gnn_type,
                            format=params.format,
                            families=params.families,
                            gnn_activ=params.gnn_activ,
                            reg_activ=params.reg_activ,
                            num_layers=params.num_layers,
                            aggr_method=params.aggr_method,
                            residual=params.residual,
                            fe_mlp=params.fe_mlp)
    
    if moment_tsr is not None:
        predictor.assign_new_moment_tsr(moment_tsr)
    predictor.mlp_moments[0] = xmu
    predictor.mlp_moments[1] = xsig

    pred_params = predictor.parameters()
    optim = t.optim.AdamW(pred_params, lr=1e-3, weight_decay=1e-6)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=params.epochs_predictor)

    fwd_func = graph_regressor_batch_fwd
    criterion = LTRLoss(t.nn.MSELoss(), l_lambda=params.corr_lambda, p=params.norm, srcc_list=srcc_list)
    train_func = train_regressor

    train_func(fwd_func, predictor, train_loader, criterion, optim, params.epochs_predictor, scheduler=scheduler)
    predictor.eval()

    test_tar, test_preds = [], []
    test_embed_list = [[] for _ in range(params.num_layers + 1)]
    with t.no_grad():
        for batch in test_loader:
            test_tar.append(batch.y.unsqueeze(1))
            pred, embed_list = fwd_func(predictor, batch)
            test_preds.append(pred.unsqueeze(1))
            for i in range(params.num_layers + 1):
                test_embed_list[i].append(t.linalg.norm(embed_list[i], ord=params.norm, dim=-1, keepdim=True))
    test_tar = t.cat(test_tar).squeeze().tolist()
    test_preds = t.cat(test_preds).squeeze().tolist()
    test_embed_list = [t.cat(test_embeds).squeeze().tolist() for test_embeds in test_embed_list]
    hop_srcc = compute_predictor_test_stats(test_tar, test_preds, test_embed_list, predictor)
    hop_ndcg = compute_predictor_test_stats_ltr(test_tar, test_preds, test_embed_list, predictor)

    for i, srcc in enumerate(hop_srcc):
        predictor.hop_srcc[i] = srcc * hop_ndcg[i]

    if params.fe_mlp:
        measure_feat_importance(test_loader, predictor)
        
    predictor.generate_moments(train_loader, ord=params.norm)

    save_f = "saved_models/" + params.chkpt + f"/fold_{fold}_predictor.pt"
    print(f"Saving predictor state dict at {save_f}")
    t.save(predictor.state_dict(),  save_f)
    config_f = params.chkpt + "_config.pkl"
    with open(f"configs/{config_f}", "wb") as f:
        pickle.dump(vars(params), f, protocol=4)
    return


if __name__ == "__main__":

    # This predictor uses an 80/20 train/test split and leverages the product of test SRCC and test NDCG@10 to weight subgraph scores.
    params = argparse.ArgumentParser(description="")
    params.add_argument("-gnn_type", default="GATv2Conv", choices=["GATv2Conv"])
    params.add_argument("-families", type=str, default="dit")
    params.add_argument("-target", required=False, type=str, default="FID-GT-1k*-1")
    params.add_argument("-format", type=str, choices=["sdm"], default="sdm")
    params.add_argument("-seed", required=False, type=int, default=12345, help="Random seed")
    params.add_argument("-fe_mlp", default=False, action="store_true") # We never used this
    params.add_argument("-gnn_dim", required=False, type=int, default=64)
    params.add_argument("-num_layers", required=False, type=int, default=4)
    params.add_argument("-aggr_method", required=False, type=str, default="mean")
    params.add_argument("-gnn_activ", required=False, type=str, default="ReLU")
    params.add_argument("-reg_activ", required=False, type=str, default="ReLU")
    params.add_argument("-corr_lambda", type=float, default=1.)
    params.add_argument("-norm", type=int, default=1)
    params.add_argument("-epochs_predictor", required=False, type=int, default=10000)
    params.add_argument("-batch_size", required=False, type=int, default=128)
    params.add_argument("-boost", default=True)
    params.add_argument("-srcc_rel", default=True, action="store_true")
    params.add_argument("-tag", type=str, default="hybrid")

    params = params.parse_args()

    params.residual = True

    srcc_list = list(range(5))
    if params.srcc_rel:
        if "dit" in params.families:
            srcc_list = [0, 1, 4]
        elif "sdv15" in params.families or "sdxl" in params.families:
            srcc_list = [0, 1, 2, 3]

    params.chkpt = "_".join([params.families, format_target(params.target)])
    
    if params.tag is not None:
        params.chkpt = params.tag + "_" + params.chkpt

    predictor_dir = "saved_models/" + params.chkpt + os.sep
    os.makedirs(predictor_dir, exist_ok=False)

    Parallel(n_jobs=5)(delayed(train_predictor)(params, i) for i in range(5))

    if params.format == "sdm":
        import subprocess
        spstr = f"python label_units_sdm_fold.py -chkpt {predictor_dir} -node"
        subprocess.run(spstr.split(" "))
        spstr = f"python label_units_sdm_fold.py -chkpt {predictor_dir}"
        subprocess.run(spstr.split(" "))
