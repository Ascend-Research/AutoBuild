from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import spearmanr


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
        for i in range(len(embeds)):
            e_srcc, e_p = spearmanr(targets, embeds[i])
            print(f"Embed {i}-hop SRCC: {e_srcc}, p: {e_p}")
