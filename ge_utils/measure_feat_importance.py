from constants import *
import torch as t


def _compute_dist_on_feat(x_slice):
    (std, mean) = t.std_mean(x_slice)
    results = {
        'mean': mean,
        'std': std,
        'min': t.min(x_slice),
        'max': t.max(x_slice)
    }
    return results


def measure_feat_importance(loader, predictor):
    
    embed_layer_str = predictor.embed_layer.__class__.__name__
    embed_layer_type = embed_layer_str.split(".")[-1]

    node_embedding_list = [predictor.get_gnn_node_embeds(x)[0].cpu() for x in loader]
    node_embedding_tsr = t.cat(node_embedding_list, dim=0)

    print(f"FE-MLP: Analyze Feature Importance")
    rel_dict = eval(f"{embed_layer_type}_FEAT_MAP")
    for key, val in rel_dict.items():
        result_dict = _compute_dist_on_feat(node_embedding_tsr[:, key])
        print(f"Feature {val}: N({result_dict['mean']}, {result_dict['std']}), [{result_dict['min']}, {result_dict['max']}])")
    
    predictor.embed_layer.analyze_node_feats()
