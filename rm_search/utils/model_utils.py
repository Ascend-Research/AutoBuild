import math
import torch
import numpy as np
import torch.nn as nn


DEVICE_STR_OVERRIDE = None


def device(device_id="cuda", ref_tensor=None):
    if ref_tensor is not None:
        return ref_tensor.get_device()
    if DEVICE_STR_OVERRIDE is None:
        return torch.device(device_id if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(DEVICE_STR_OVERRIDE)


def can_parallel(device_str=None):
    if not torch.cuda.is_available():
        return False
    if torch.cuda.device_count() <= 1:
        return False
    return device_str is None or device_str.lower() == "cuda"


def measure_gpu_latency(inf_func, m_ignore_runs=10, n_reps=100):
    from search.rm_search.utils.math_utils import mean, variance
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    # GPU-WARM-UP
    for _ in range(m_ignore_runs):
        inf_func()
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(n_reps):
            starter.record()
            inf_func()
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # in milliseconds
            timings.append(curr_time)
    return mean(timings), variance(timings)


def measure_cpu_latency(inf_func, m_ignore_runs=1, n_reps=5):
    import time
    from search.rm_search.utils.math_utils import mean, variance
    timings = []
    for _ in range(m_ignore_runs):
        inf_func()
    with torch.no_grad():
        for rep in range(n_reps):
            start = time.time()
            inf_func()
            curr_time = time.time() - start # In seconds
            timings.append(curr_time)
    return mean(timings), variance(timings)


def model_save(file_path, data):
    torch.save(data, file_path)


def model_load(file_path):
    return torch.load(file_path, map_location=lambda storage,loc:storage)


def set_random_seed(seed, log_f=print):
    torch.manual_seed(seed)
    log_f('My seed is {}'.format(torch.initial_seed()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        log_f('My cuda seed is {}'.format(torch.cuda.initial_seed()))
    np.random.seed(seed)


def to_sorted_tensor(tensor, lengths, sort_dim):
    """
    Sort tensor according to sequence lengths.
    This is used before applying pack_padded_sequence.
    :param tensor: input tensor
    :param lengths: 1D tensor of sequence lengths
    :param sort_dim: the dimension to sort in input tensor
    :param device: calculation device
    :return: sorted tensor, sorted lengths, and sorted index.
    """
    # rng = list(range(lengths.size(0)))
    # original_idx = torch.LongTensor(rng).unsqueeze(1).to(device())
    # lengths = lengths.unsqueeze(1)
    # lengths = torch.cat([original_idx, lengths], dim=1)
    sorted_lengths, sorted_idx = torch.sort(lengths.long(), dim=0, descending=True)
    sorted_idx = sorted_idx.to(device())
    sorted_tensor = tensor.index_select(dim=sort_dim, index=sorted_idx)
    return sorted_tensor, sorted_lengths, sorted_idx


def to_original_tensor(sorted_tensor, sorted_idx, sort_dim):
    """
    Restore tensor to its original order.
    This is used after applying pad_packed_sequence.
    :param sorted_tensor: a sorted tensor
    :param sorted_idx: sorted index of the sorted_tensor
    :param sort_dim: the dimension of sorted_tensor where it is sorted
    :device: calculation device
    :return: the original unsorted tensor
    """
    rng = range(sorted_idx.size(0))
    tmp = sorted(rng, key=lambda x: sorted_idx[x])
    original_idx = torch.LongTensor(tmp).to(device())
    tensor = sorted_tensor.index_select(dim=sort_dim, index=original_idx)
    return tensor


def init_weights(model, base_model_type=None):
    if base_model_type == "rnn":
        for p in filter(lambda pa: pa.requires_grad, model.parameters()):
            if p.dim() == 1:
                p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
            else:
                nn.init.xavier_normal_(p, math.sqrt(3))
    elif base_model_type == "transformer":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def parallel(model):
    if hasattr(model, "avoid_parallel") and model.avoid_parallel:
        return model
    return nn.DataParallel(model)
