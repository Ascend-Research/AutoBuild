from params import *
import argparse
import torch as t
import pickle
from ge_utils.data_loader import get_regress_train_test_data, make_dataloader, standardize_targets
from ge_utils.data_loader import graph_regressor_batch_fwd
from ge_utils.measure_feat_importance import measure_feat_importance
from onnx_ir.trainer import train_regressor
from ge_utils.model import make_predictor
from ge_utils.diff_srcc import CorrelLoss
from ge_utils.misc_utils import set_random_seed
from ge_utils.metrics import compute_predictor_test_stats
from ge_utils.label_eq import format_target
from ge_utils.subgraph_moments import calc_moments_training_data


if __name__ == "__main__":

    params = argparse.ArgumentParser(description="")
    params.add_argument("-gnn_type", default="GATv2Conv", choices=["GATv2Conv"])
    params.add_argument("-families", type=str, default="ofa_mbv3")
    params.add_argument("-target", required=False, type=str, default="acc")
    params.add_argument("-format", type=str, choices=["custom", "onnx_ir"], default="custom")
    params.add_argument("-seed", required=False, type=int, default=12345, help="Random seed")
    params.add_argument("-gnn_dim", required=False, type=int, default=32)
    params.add_argument("-num_layers", required=False, type=int, default=4)
    params.add_argument("-aggr_method", required=False, type=str, default="mean")
    params.add_argument("-gnn_activ", required=False, type=str, default="ReLU")
    params.add_argument("-reg_activ", required=False, type=str, default="ReLU")
    params.add_argument("-corr_lambda", type=float, default=1.)
    params.add_argument("-norm", type=int, default=1)
    params.add_argument("-epochs_predictor", required=False, type=int, default=200)
    params.add_argument("-batch_size", required=False, type=int, default=128)
    params.add_argument("-tag", type=str, default=None)
    params.add_argument("-plots", action="store_true", default=False)

    params = params.parse_args()

    params.residual = True

    if params.format == "custom":
        assert len(params.families.split("+")) == 1, "Custom format is family-specific"

    set_random_seed(params.seed)

    params.chkpt = "_".join([params.families, params.format, format_target(params.target), params.gnn_type.replace(" ", ""), params.aggr_method, "directed", 
                             f"srcc{str(int(params.corr_lambda))}"])
    
    if params.tag is not None:
        params.chkpt = params.tag + "_" + params.chkpt
    
    train_data, test_data, best_entry = get_regress_train_test_data(caches=params.families, format=params.format, train_ratio=0.9, seed=params.seed, label=params.target)

    if params.format == "custom":
        moment_tsr = calc_moments_training_data(train_data, params.families)
    else:
        moment_tsr = None

    train_data, xmu, xsig = standardize_targets(train_data)
    test_data, _, _ = standardize_targets(test_data, xmu, xsig)
    
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
                               residual=params.residual)
    
    if moment_tsr is not None:
        predictor.assign_new_moment_tsr(moment_tsr)
    predictor.mlp_moments[0] = xmu
    predictor.mlp_moments[1] = xsig

    pred_params = predictor.parameters()
    optim = t.optim.AdamW(pred_params, lr=1e-4, weight_decay=1e-5)
 
    fwd_func = graph_regressor_batch_fwd
    criterion = CorrelLoss(t.nn.MSELoss(), l_lambda=params.corr_lambda, p=params.norm)
    train_func = train_regressor

    train_func(fwd_func, predictor, train_loader, criterion, optim, params.epochs_predictor)
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
    compute_predictor_test_stats(test_tar, test_preds, test_embed_list, predictor)

    measure_feat_importance(test_loader, predictor)

    if params.plots:
        try:
            if params.format == "onnx_ir":
                from ge_utils.visualize_nembed import gen_onnx_embed_plot
                gen_onnx_embed_plot(predictor, best_entry, fname=f"plots/{params.chkpt}", 
                                    norm=params.norm, gen_all=True, undirected=params.undirected)
            else:
                from ge_utils.visualize_nembed import gen_custom_embed_plot
                gen_custom_embed_plot(predictor, best_entry, fname=f"plots/{params.chkpt}", 
                                    norm=params.norm, gen_all=True)
        except:
            print("[CAUGHT ERROR] '-plots' specified, but likely graphviz was not installed")
        
    predictor.generate_moments(train_loader, ord=params.norm)

    save_f = "saved_models/" + params.chkpt + "_predictor.pt"
    print(f"Saving predictor state dict at {save_f}")
    t.save(predictor.state_dict(),  save_f)
    config_f = params.chkpt + "_config.pkl"
    with open(f"configs/{config_f}", "wb") as f:
        pickle.dump(vars(params), f, protocol=4)
