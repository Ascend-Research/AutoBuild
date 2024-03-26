import argparse
from ge_utils.misc_utils import get_train_config_from_chkpt, save_units
from ge_utils.model import make_predictor
import torch as t
from tqdm import tqdm


if __name__ == "__main__":
    params = argparse.ArgumentParser(description="")
    params.add_argument("-chkpt", type=str, required=True)

    params = params.parse_args()

    train_config = get_train_config_from_chkpt(params.chkpt)
    assert train_config['format'] == "custom"
    predictor = make_predictor(gnn_chkpt=params.chkpt, **train_config)
    predictor.eval()

    family = train_config['families']
    if "mbv3" in family: from iterators.mbv3 import gen_sgs
    elif "pn" in family: from iterators.pn import gen_sgs
    else: raise NotImplementedError

    all_sgs = gen_sgs()
    for sg_dict in tqdm(all_sgs):
        sg_embed = predictor.get_gnn_node_embeds(sg_dict['tg_subgraph'])[sg_dict['hops']][-1, :]
        biased_score = t.linalg.norm(sg_embed, ord=train_config['norm'], keepdim=False)
        sg_dict['score'] = predictor.dist_shift(biased_score, sg_dict['unit'] - 1, sg_dict['hops']).detach().item()

    save_f = save_units(all_sgs, params.chkpt)
