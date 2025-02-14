import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


RELEVANCE_SCALE = 50.


class RelevanceCalculator:
    def __init__(self, lower, upper, scale=RELEVANCE_SCALE):
        print(f'Creating relevance calculator: lower = {lower:.6f}, upper = {upper:.6f}, scale = {scale:.2f}', __name__)
        self.lower = lower
        self.upper = upper
        self.scale = scale

    @classmethod
    def from_data(cls, x, scale=RELEVANCE_SCALE):
        # TODO use 20% for now
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        lower = np.quantile(x, 0.2)
        upper = np.max(x)
        return cls(lower, upper, scale)

    def __call__(self, x):
        if torch.is_tensor(x):
            return torch.clamp((x - self.lower) / (self.upper - self.lower), 0, 1) * self.scale
        else:
            return np.clip((x - self.lower) / (self.upper - self.lower), 0, 1) * self.scale


class RankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        split_size = input.size(0) // 2
        pred_diff = input[:split_size] - input[-split_size:]
        targ_diff = (target[:split_size] - target[-split_size:] > 0).float()
        return self.bce_loss(pred_diff, targ_diff)


class BRPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, input1, input2, target1, target2):
        input = F.log_softmax(torch.stack((input1, input2), 1), 1)
        target = F.softmax(torch.stack((target1, target2), 1), 1)
        return self.kl_div_loss(input, target)


class Normalizer:
    def __init__(self, mean, std):
        print(f'Creating normalizer: mean = {mean:.6f}, std = {std:.6f}', __name__)
        self.mean = mean
        self.std = std

    @classmethod
    def from_data(cls, x):
        if torch.is_tensor(x):
            return cls(torch.mean(x).item(), torch.std(x).item())
        return cls(np.mean(x).item(), np.std(x).item())

    def __call__(self, x, denormalize=False):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


class DCG(object):

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int DCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        self.k = k
        self.discount = self._make_discount(256)
        if gain_type in ['exp2', 'identity']:
            self.gain_type = gain_type
        else:
            raise ValueError('gain type not equal to exp2 or identity')

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        return np.sum(np.divide(gain, discount))

    def _get_gain(self, targets):
        t = targets[:self.k]
        if self.gain_type == 'exp2':
            return np.power(2.0, t) - 1.0
        else:
            return t

    def _get_discount(self, k):
        if k > len(self.discount):
            self.discount = self._make_discount(2 * len(self.discount))
        return self.discount[:k]

    @staticmethod
    def _make_discount(n):
        x = np.arange(1, n+1, 1)
        discount = np.log2(x + 1)
        return discount


class NDCG(DCG):
    """
    NDCG:
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
    """

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int NDCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        super(NDCG, self).__init__(k, gain_type)

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        dcg = super(NDCG, self).evaluate(targets)
        ideal = np.sort(targets)[::-1]
        idcg = super(NDCG, self).evaluate(ideal)
        return dcg / idcg

    def maxDCG(self, targets):
        """
        :param targets: ranked list with relevance
        :return:
        """
        ideal = np.sort(targets)[::-1]
        return super(NDCG, self).evaluate(ideal)


class LambdaRankLoss(nn.Module):
    def __init__(self):
        super(LambdaRankLoss, self).__init__()
        self.ndcg_gain_in_train = 'exp2'
        self.ideal_dcg = NDCG(2 ** 9, self.ndcg_gain_in_train)
        self.sigma = 1.0

    def forward(self, prediction, target):
        # target should have been relevance-computed
        target_npy = target.cpu().numpy()
        prediction = prediction.view(-1, 1)
        target = target.view(-1, 1)

        N = 1.0 / self.ideal_dcg.maxDCG(target_npy)

        # compute the rank order of each document
        rank_df = pd.DataFrame({'Y': target_npy, 'doc': np.arange(target_npy.shape[0])})
        rank_df = rank_df.sort_values('Y').reset_index(drop=True)
        rank_order = rank_df.sort_values('doc').index.values + 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp(self.sigma * (prediction - prediction.t()))

            rel_diff = target - target.t()
            pos_pairs = (rel_diff > 0).float()
            neg_pairs = (rel_diff < 0).float()
            Sij = pos_pairs - neg_pairs
            if self.ndcg_gain_in_train == 'exp2':
                gain_diff = torch.pow(2.0, target) - torch.pow(2.0, target.t())
            elif self.ndcg_gain_in_train == 'identity':
                gain_diff = target - target.t()

            rank_order_tensor = torch.tensor(rank_order, dtype=torch.float, device=prediction.device).view(-1, 1)
            decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            lambda_update = self.sigma * (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, 1, keepdim=True)

            assert lambda_update.shape == prediction.shape

        prediction.backward(lambda_update, retain_graph=True)


def get_criterion(loss_fn):
    if loss_fn == 'ranknet':
        criterion = RankNetLoss()
    elif loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn == 'lambdarank':
        criterion = LambdaRankLoss()
    elif loss_fn == 'brp':
        criterion = BRPLoss()
    else:
        raise ValueError(f'Criterion type {loss_fn} not found.')
    return criterion


def dcg_score(y_true, y_score, k=10, gains='exponential'):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == 'exponential':
        gains = 2 ** y_true - 1
    elif gains == 'linear':
        gains = y_true
    else:
        raise ValueError('Invalid gains option.')

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains='exponential', thresh=0., printfunc=print):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    thresh : float
        Threshold for filtering values
    printfunc  : printer
        print function
    Returns
    -------
    NDCG @k : float
    """
    if thresh > 0:
        order = np.argsort(y_true)[::-1]
        y_true = y_true[order]
        y_score = y_score[order]
        filt_indices = [0]
        i, j = 0, 1
        while j < y_true.shape[0]:
            if np.abs(y_true[i] - y_true[j]) > thresh:
                filt_indices.append(j)
                i = j
            j += 1
        printfunc("NDCG threshold %f; %d/%d data samples (%.3f%%) kept" % (thresh, len(filt_indices), y_true.shape[0],
                                                                           (100 *
                                                                            (len(filt_indices) / y_true.shape[0]))))
        y_true = y_true[filt_indices]
        y_score = y_score[filt_indices]

    from sklearn.metrics import ndcg_score as new_ndcg_score
    y_true = np.expand_dims(y_true, axis=0)
    y_score = np.expand_dims(y_score, axis=0)
    return new_ndcg_score(y_true, y_score, k=len(y_true), ignore_ties=False)
