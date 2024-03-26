import torchsort
import torch


# https://forum.numer.ai/t/differentiable-spearman-in-pytorch-optimize-for-corr-directly/2287
def t_corrcoef(target, pred):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()


def t_spearman( target, pred, regularization="l2", regularization_strength=1.0):
    pred = torchsort.soft_rank( pred.cpu(),
                                regularization=regularization,
                                regularization_strength=regularization_strength )
    return t_corrcoef(target, pred.to( target.device) / pred.to( target.device).shape[-1])


class CorrelLoss(torch.nn.Module):
    def __init__(self, origloss, l_lambda=1., crit="srcc", p=1):
        super(CorrelLoss, self).__init__()
        self.l = origloss
        self.l_lambda = l_lambda
        assert crit in ['srcc']  
        self.crit = crit
        self.p = p

    def forward(self, input, target, embed_list):
        original_loss = self.l(input, target)
        new_loss = 0.
        if self.l_lambda > 0.:
            target = target.unsqueeze(0)
            embed_list = [torch.linalg.norm(embedding, ord=self.p, dim=-1).unsqueeze(0) for embedding in embed_list]
            for embedding in embed_list:
                new_loss = new_loss + (1. - t_spearman(target, embedding))
            new_loss = new_loss / len(embed_list)
        return original_loss + (self.l_lambda * new_loss)
