import argparse
import pickle


if __name__ == "__main__":

    params = argparse.ArgumentParser(description="")
    params.add_argument("-unit_corpus", type=str, required=True)
    params.add_argument("-k", type=int, nargs="+", default=5)
    params.add_argument("-n", type=int, default=-1)
    
    params = params.parse_args()

    with open(params.unit_corpus, "rb") as f:
        unit_list = pickle.load(f)

    if "mbv3" in params.unit_corpus:
        num_units = 5
    elif "pn" in params.unit_corpus:
        num_units = 5
    else:
        raise NotImplementedError
     
    if type(params.k) == int:
        params.k = [params.k] * num_units
    elif len(params.k) == 1:
        params.k = [params.k[0]] * num_units

    k_seq = "-".join([str(x) for x in params.k])
    save_file_name = params.unit_corpus.replace("units", "evals").replace("labeled_sgs", f"{k_seq}_search_space")

    top_k_units = []
    total_combs = 1
    for u in range(1, num_units + 1):
        relevant_units = [unit for unit in unit_list if unit['unit'] == u]
        relevant_units.sort(reverse=True, key=lambda x: x['score'])
        relevant_units = [u['config'] for u in relevant_units]
        top_k_units.append(relevant_units[:params.k[u - 1]])
        total_combs *= len(top_k_units[u - 1]) 
        print(f"Unit {u}: {len(top_k_units[u - 1])} units")
    print(f"Total search space size: {total_combs}")

    with open(save_file_name, "wb") as f:
        pickle.dump(top_k_units, f, protocol=4)