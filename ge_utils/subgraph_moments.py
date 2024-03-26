import torch as t
import numpy as np


def _get_config_or_net(entry):
    if "original config" in entry.keys():
        return entry['original config'][1]
    else:
        return entry['net']


def calc_moments_training_data(train_data, family):

    if "mbv3" in family:
        S = 5
        H = 4
        target_dict = {}
        for entry in train_data:
            config = _get_config_or_net(entry[0])
            for s, sublist in enumerate(config):
                key = f"{s}_{len(sublist) - 1}"
                if key in target_dict.keys():
                    target_dict[key].append(entry[1])
                else:
                    target_dict[key] = [entry[1]]
    elif "pn" in family:
        S = 5
        H = 5
        target_dict = {}
        for entry in train_data:
            config = _get_config_or_net(entry[0])
            for s, sublist in enumerate(config[:-1]):
                if s == 4:
                    key = f"{s}_{len(sublist)}"
                else:
                    key = f"{s}_{len(sublist) - 1}"
                if key in target_dict.keys():
                    target_dict[key].append(entry[1])
                else:
                    target_dict[key] = [entry[1]]
    else:
        return None
    moment_tensor = t.cat([t.zeros(S, H, 1), t.ones(S, H, 1)], dim=-1)
    for k, v in target_dict.items():
        s, h = k.split("_")
        s = int(s)
        h = int(h)
        moment_tensor[s, h, 0], moment_tensor[s, h, 1] = np.mean(v), np.std(v)
    print(moment_tensor)
    return moment_tensor