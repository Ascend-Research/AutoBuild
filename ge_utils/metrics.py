from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import spearmanr
from ge_utils.acenas_loss import ndcg_score


def compute_predictor_test_stats(targets, preds, embeds=None, predictor=None):

    mae = mean_absolute_error(targets, preds)
    print(f"Test MAE: {mae}")
    mape = mean_absolute_percentage_error(targets, preds)
    print(f"Test MAPE: {mape}")
    t_srcc, t_p = spearmanr(targets, preds)
    print(f"Test SRCC: {t_srcc}, p: {t_p}")
    if predictor is not None:
        predictor.test_metrics[0] = mae
        predictor.test_metrics[1] = mape
        predictor.test_metrics[2] = t_srcc
    if embeds is not None:
        hop_srccs = []
        for i in range(len(embeds)):
            e_srcc, e_p = spearmanr(targets, embeds[i])
            print(f"Embed {i}-hop SRCC: {e_srcc}, p: {e_p}")
            hop_srccs.append(e_srcc)
        return hop_srccs


def compute_predictor_test_stats_ltr(targets, preds, embeds=None, predictor=None):

    mae = mean_absolute_error(targets, preds)
    print(f"Test MAE: {mae}")
    mape = mean_absolute_percentage_error(targets, preds)
    print(f"Test MAPE: {mape}")
    t_ndcg = ndcg_score(targets, preds)
    print(f"Test NDCG: {t_ndcg}")
    if predictor is not None:
        predictor.test_metrics[0] = mae
        predictor.test_metrics[1] = mape
        predictor.test_metrics[2] = t_ndcg
    if embeds is not None:
        hop_srccs = []
        for i in range(len(embeds)):
            e_ndcg = ndcg_score(targets, embeds[i])
            print(f"Embed {i}-hop NDCG: {e_ndcg}")
            hop_srccs.append(e_ndcg)
        return hop_srccs
