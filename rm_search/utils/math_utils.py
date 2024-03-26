import torch
import numpy as np
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax_onehot(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_discrete = (y_hard - y).detach() + y
    return y_discrete


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


def median(list_val, fallback_val=0):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ValueError()
    v = np.median(np.asarray(list_val))
    return v


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


if __name__ == '__main__':
    import math
    print(
        gumbel_softmax_onehot(
            logits=torch.FloatTensor([[math.log(0.1), math.log(0.4), math.log(0.3), math.log(0.2)]] * 20000),
            temperature=0.8
        ).sum(dim=0)
    )
