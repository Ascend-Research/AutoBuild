import copy
import torch
import random
import numpy as np
import pickle
from torch.distributions import Categorical
from sklearn.metrics import mean_squared_error, mean_absolute_error


DEVICE_STR_OVERRIDE = None


def device(device_id="cuda:0", ref_tensor=None):
    if ref_tensor is not None:
        return ref_tensor.get_device()
    if DEVICE_STR_OVERRIDE is None:
        return torch.device(device_id if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(DEVICE_STR_OVERRIDE)


def sample_categorical_idx(weights):
    if len(weights) == 1: return 0
    idx = Categorical(probs=torch.FloatTensor(weights)).sample()
    return idx.item()


def set_random_seed(seed, log_f=print):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    log_f('My seed is {}'.format(torch.initial_seed()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        log_f('My cuda seed is {}'.format(torch.cuda.initial_seed()))


class UniqueList:

    def __init__(self, key_func=str):
        self._i = 0
        self._list = []
        self._set = set()
        self._key_func = key_func

    def tolist(self):
        return [v for v in self._list]

    def keys(self):
        return copy.deepcopy(self._set)

    def clear(self):
        self._i = 0
        self._list = []
        self._set = set()

    def append(self, val):
        key = self._key_func(val)
        if key not in self._set:
            self._set.add(key)
            self._list.append(val)
            return True
        return False

    def extend(self, vals):
        n_added = 0
        for item in vals:
            if self.append(item):
                n_added += 1
        return n_added

    def next(self):
        if self._i >= len(self._list):
            self._i  = 0
            raise StopIteration()
        item = self._list[self._i]
        self._i += 1
        return item

    def __getitem__(self, idx):
        return self._list[idx]

    def __contains__(self, item):
        key = self._key_func(item)
        return key in self._set

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self._list)

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return str(self)


class RunningStatMeter(object):

    def __init__(self):
        self.avg = 0.
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0.
        self.cnt = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        return self

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.max = max(self.max, val)
        self.min = min(self.min, val)


def reverse_dict(dictionary):
    return {v: k for k, v in dictionary.items()}


def get_train_config_from_chkpt(chkpt):
    config_file = chkpt.replace("saved_models", "configs").replace("predictor.pt", "config.pkl")
    with open(config_file, "rb") as f:
        train_config = pickle.load(f)
    return train_config


def get_train_config_from_chkpt_folder(chkpt):
    config_file = chkpt.replace("saved_models", "configs")
    config_file = config_file[:-1]
    config_file += "_config.pkl"
    with open(config_file, "rb") as f:
        train_config = pickle.load(f)
    return train_config


def save_units(sg_list, chkpt):
    save_file = chkpt.replace("saved_models", "units").replace("predictor.pt", "labeled_sgs.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(sg_list, f, protocol=4)
    return save_file


def save_units_folder(sg_list, chkpt):
    save_file = chkpt.replace("saved_models", "units") #.replace("predictor.pt", "labeled_sgs.pkl")
    save_file = save_file[:-1]
    save_file += "_labeled.pkl"
    print(save_file)
    with open(save_file, "wb") as f:
        pickle.dump(sg_list, f, protocol=4)
    return save_file


def mean(list_val, fallback_val=None):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ZeroDivisionError()
    return sum(list_val) / len(list_val)


def variance(list_val, fallback_val=0):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ValueError()
    v = np.var(np.asarray(list_val))
    return v


def get_regression_metrics(pred_list, target_list):
    assert len(pred_list) == len(target_list), \
        "pred len: {}, target len: {}".format(len(pred_list), len(target_list))
    n = len(pred_list)
    mean_sq_error = mean_squared_error(target_list, pred_list)
    mean_abs_error = mean_absolute_error(target_list, pred_list)
    max_err = max([abs(target_list[i] - pred_list[i]) for i in range(n)])
    if any(abs(v) < 1e-9 for v in target_list):
        mape_pred_list, mape_target_list = [], []
        for i, v in enumerate(target_list):
            if abs(v) < 1e-9: continue
            mape_pred_list.append(pred_list[i])
            mape_target_list.append(v)
    else:
        mape_pred_list, mape_target_list = pred_list, target_list
    mape = 100 * mean([abs(mape_pred_list[i] - truth) / abs(truth) for i, truth in enumerate(mape_target_list)],
                      fallback_val=0)
    pred_mean = mean(pred_list)
    pred_variance = variance(pred_list)
    truth_mean = mean(target_list)
    truth_variance = variance(target_list)
    rv = {
        "mean_square_error": mean_sq_error,
        "mean_absolute_error": mean_abs_error,
        #"max_error": max_err,
        #"mean_absolute_percent_error": mape,
        #"mape_effective_sample_size_diff": abs(len(mape_pred_list) - len(pred_list)),
        #"pred_mean": pred_mean,
        #"pred_variance": pred_variance,
        #"truth_mean": truth_mean,
        #"truth_variance": truth_variance,
    }
    
    return rv