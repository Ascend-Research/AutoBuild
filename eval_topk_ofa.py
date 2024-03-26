import argparse
import pickle
import itertools
import numpy as np
import fcntl
import os


class Locker:
    def __init__(self, lockname):
        self.lockname = lockname
        if not os.path.exists(lockname):
            with open(lockname, "wb") as f:
                pickle.dump("I am a lock", f, protocol=4)
    def __enter__ (self):
        self.fp = open(self.lockname)
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()


def merge_dicts(new, existing):
    for k, v in new.items():
        if k in existing.keys():
            existing[k] = {**v, **existing[k]}
    return existing


def save_file(contents, fname, lock):
    with lock:
        if os.path.isfile(fname):
            with open(fname, "rb") as f:
                data = pickle.load(f)
            data = merge_dicts(contents, data)
        else:
            data = contents
        with open(fname, "wb") as f:
            pickle.dump(data, f, protocol=4)


if __name__ == "__main__":

    params = argparse.ArgumentParser(description="")
    params.add_argument("-unit_corpus", type=str, required=True)
    params.add_argument("-k", type=int, nargs="+", default=5)
    params.add_argument("-n", type=int, default=-1)
    params.add_argument("-m", type=str, default="acc")
    params.add_argument("-seed", type=int, default=-1)
    params.add_argument("-start_idx", type=int, default=0)
    params.add_argument("-res", type=int, default=224)
    params.add_argument("-no_eval", action="store_true", default=False)

    params = params.parse_args()
    
    if params.m == "acc":
        assert params.res in [192, 208, 224]

    with open(params.unit_corpus, "rb") as f:
        unit_list = pickle.load(f)

    if "mbv3" in params.unit_corpus:
        num_units = 5
        from iterators.mbv3 import assemble_whole_net_cfg
        from ge_utils.torch_geo_data import _mobilenet_custom_as_torch_geo as torch_geo_func
        from iterators.ofa_manager import MBv3Manager as AccManager
    elif "pn" in params.unit_corpus:
        num_units = 5
        from iterators.pn import assemble_whole_net_cfg
        from ge_utils.torch_geo_data import _mobilenet_custom_as_torch_geo as torch_geo_func
        from iterators.ofa_manager import PNManager as AccManager
    else:
        raise NotImplementedError
    
    acc_manager = AccManager(metric=params.m)
    
    if type(params.k) == int:
        params.k = [params.k] * num_units
    elif len(params.k) == 1:
        params.k = [params.k[0]] * num_units

    k_seq = "-".join([str(x) for x in params.k])
    save_file_name = params.unit_corpus.replace("units", "evals").replace("labeled_sgs", f"{k_seq}_s{params.seed}")
    lock = Locker(lockname=save_file_name.replace(".pkl", ".lock"))

    top_k_units = []
    total_combs = 1
    for u in range(1, num_units + 1):
        relevant_units = [unit for unit in unit_list if unit['unit'] == u]
        relevant_units.sort(reverse=True, key=lambda x: x['score'])
        top_k_units.append(relevant_units[:params.k[u - 1]])
        total_combs *= len(top_k_units[u - 1]) 
        print(f"Unit {u}: {len(top_k_units[u - 1])} units")
    print(f"Total search space size: {total_combs}")

    all_archs = list(itertools.product(*top_k_units))
    all_archs.sort(reverse=True, key=lambda x: sum([cfg['score'] for cfg in x]))
    
    if params.seed >= 0:
        import random
        random.shuffle(all_archs)

    if params.n > 0:
        arch_subset = all_archs[params.start_idx:params.start_idx+params.n]
    else:
        arch_subset = all_archs[params.start_idx:]

    arch_dict = {}
    try:
        for idx, arch in enumerate(arch_subset):
            i = idx + params.start_idx
            unit_cfgs = [a['config'] for a in arch]
            whole_arch_cfg = assemble_whole_net_cfg(*unit_cfgs)
            print("===================================")
            print(f"Architecture {i}: {whole_arch_cfg}")
            arch_dict[i] = {'arch': whole_arch_cfg}
            arch_geo = torch_geo_func(whole_arch_cfg)

            if params.no_eval:
                continue

            arch_met = acc_manager.eval(whole_arch_cfg)
            print(f"Arch {i}: {arch_met}")
            arch_dict[i] = {**arch_dict[i], **arch_met}

    except FileNotFoundError:
        print("INTERRUPTED! SAVING!")
    finally:
        save_file(arch_dict, save_file_name, lock)
        if not params.no_eval:
            print(f"Metric distribution amongst {len(arch_dict.keys())} archs")
            metrics = [x[params.m] for x in arch_dict.values()]
            mu, sig, mi, ma = np.mean(metrics), np.std(metrics), np.min(metrics), np.max(metrics)
            print(f"N({mu}, {sig}); [{mi}, {ma}]")

