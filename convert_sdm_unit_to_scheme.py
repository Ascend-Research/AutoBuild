import pickle
import argparse


if __name__ == "__main__":
    params = argparse.ArgumentParser(description="")
    params.add_argument("-units", type=str, required=True)
    params.add_argument("-fname", default=None, required=False)
    params.add_argument("-reverse", type=int, default=None, nargs="+")

    params = params.parse_args()

    with open(params.units, "rb") as f:
        score_dict = pickle.load(f)

    top_1_dict = {}
    for k, v in score_dict.items():
        best_subgraph = v[0]
        if params.reverse is not None and best_subgraph.hops in params.reverse:
            best_subgraph = v[-1]
        for node in v[0].nodes(data=True):
            node_name = node[0]
            best_node_config = node[1]['config']
            if "quantizer" not in node_name:
                continue
            assert node_name not in top_1_dict.keys()
            top_1_dict[node_name] = best_node_config

    save_file_name = params.units.replace("units", "quant_configs").replace("labeled_sgs", "top_1_config") if params.fname is None else params.fname

    with open(save_file_name, "wb") as f:
        pickle.dump(top_1_dict, f, protocol=4)
