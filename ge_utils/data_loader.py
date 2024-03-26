import os
import pickle
import random
from params import CACHE_DIR
from torch_geometric.loader import DataLoader
from ge_utils.torch_geo_data import get_entry_as_torch_geo
from ge_utils.label_eq import process_label
import numpy as np
import torch as t


def graph_regressor_batch_fwd(net, batch, index=None, ext_feat=None):
    return net(batch)


def get_regress_train_test_data(caches="ofa_mbv3", format="custom", train_ratio = 0.9, seed=None, label='acc'):

    data = []
    file_prefixes = caches.split("+")
    if format != "onnx_ir":
        assert len(file_prefixes) == 1, "Custom families are specific"
    for prefix in file_prefixes:
        file_path = os.path.join(CACHE_DIR, f"{prefix}_{format}_cache.pkl")
        print("Loading", file_path)
        with open(file_path, "rb") as f:
            data.extend(pickle.load(f))
    print("Loading done")

    instances = []
    valid_instances = 0
    best_label, best_entry = float('-inf'), None
    for d in data:
        target = process_label(d, label)
        if target is not None:
            valid_instances += 1
            instances.append((d, target))
            if target > best_label:
                best_label = target
                best_entry = d

    print(f"Collected {valid_instances}/{len(data)} ({(100*valid_instances) / len(data)}%) data entries for label string '{label}'")

    if seed is not None and type(seed) is int:
        random.seed(seed)
        random.shuffle(instances)

    train_idx = int(len(instances) * train_ratio)
    train_instances = instances[:train_idx]
    test_instances = instances[train_idx:]

    return train_instances, test_instances, best_entry

        
def standardize_targets(instances, xmu=None, xsig=None):
    if xmu is None:
        assert xsig is None
        xmu, xsig = _calc_normal_stats(instances)
        print("Normalizing targets to N(0, 1)")
        instances = _apply_normal_fit(instances, xmu, xsig)
    else:
        print("Normalizing using existing mean/s.dev")
        instances = _apply_normal_fit(instances, xmu, xsig)
    return instances, xmu, xsig


def make_dataloader(instances, format="onnx_ir", batch_size=32, shuffle=True, undirected=False):
    data_list = [get_entry_as_torch_geo(x[0], undirected=undirected, format=format, y=t.FloatTensor([x[1]])) for x in instances]
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def _calc_normal_stats(train_instance_list):
    labels = [l[-1] for l in train_instance_list]
    mu, sig, mi, ma = np.mean(labels), np.std(labels), np.min(labels), np.max(labels)
    print(f"Training data distribution: N({mu}, {sig}), [{mi}, {ma}]")
    return mu, sig


def _apply_normal_fit(instance_list, xmu, xsig):
    def _transform(x, xmu, xsig):
        return (x - xmu) / xsig
    return [(l[0], _transform(l[1], xmu, xsig)) for l in instance_list]

def reverse_normal_fit(value, xmu, xsig):
    return (value + xmu) * xsig
