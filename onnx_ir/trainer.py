from ge_utils.misc_utils import device, get_regression_metrics
from torch_geometric.data.batch import Batch
import torch
import collections
from scipy.stats import spearmanr


def train_regressor(batch_fwd_func, model, train_loader, criterion, optimizer, num_epochs,
                    log_f=print,
                    max_gradient_norm=5.0,
                    eval_start_epoch=1, eval_every_epoch=1,
                    rv_metric_name="mean_absolute_percent_error",
                    completed_epochs=0,
                    dev_loader=None,
                    dev_eval_func=lambda new, old: new < old,
                    best_dev_score=float("inf"),
                    best_dev_epoch=0,
                    checkpoint_func=None):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(num_epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_score = run_regressor_epoch(batch_fwd_func, model, train_loader, criterion, optimizer,
                                          rv_metric_name=rv_metric_name, max_grad_norm=max_gradient_norm,
                                          curr_epoch=report_epoch, log_f=log_f)
        log_f(f"Train score at epoch {report_epoch}: {train_score}")
        if checkpoint_func is not None:
            checkpoint_func("latest", report_epoch, model, optimizer, train_score)

        if dev_loader is not None:
            with torch.no_grad():
                model.eval()
                if report_epoch >= eval_start_epoch and report_epoch % eval_every_epoch == 0:
                    dev_score = run_regressor_epoch(batch_fwd_func, model, dev_loader, criterion, None,
                                                    rv_metric_name=rv_metric_name, desc="Dev",
                                                    max_grad_norm=max_gradient_norm,
                                                    curr_epoch=report_epoch,
                                                    log_f=log_f)
                    log_f(f"Dev score at epoch {report_epoch}: {dev_score}")
                    if dev_eval_func(dev_score, best_dev_score):
                        best_dev_score = dev_score
                        best_dev_epoch = report_epoch
                        if checkpoint_func is not None:
                            checkpoint_func("best", report_epoch, model, optimizer, dev_score)
                    log_f(f"Current best dev score: {best_dev_score}, epoch: {best_dev_epoch}")
        log_f("")


def run_regressor_epoch(batch_fwd_func,
                        model, data, criterion, optimizer,
                        desc="Train",
                        curr_epoch=0,
                        max_grad_norm=5.0,
                        report_metrics=True,
                        rv_metric_name="mean_absolute_percent_error",
                        log_f=print):
    total_loss, n_instances = 0., 0
    metrics_dict = collections.defaultdict(float)
    preds, targets = [], []
    for batch in data:
        truth = batch.y.to(device())
        
        pred = batch_fwd_func(model, batch)
        if type(pred) is tuple:
            pred, embeds = pred[0], pred[1]
            loss = criterion(pred, truth, embeds)
        else:
            loss = criterion(pred, truth)
        total_loss += loss.item() * truth.shape[0]
        preds.extend(pred.detach().tolist())
        targets.extend(truth.detach().tolist())
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        n_instances += truth.shape[0]
    rv_loss = total_loss / n_instances
    msg = desc + f" epoch: {curr_epoch}, loss: {rv_loss}"
    log_f(msg)
    if report_metrics:
        metrics_dict = get_regression_metrics(preds, targets)
        spearman_rho, spearman_p = spearmanr(preds, targets)
        metrics_dict["spearman_rho"] = spearman_rho
        metrics_dict["spearman_p"] = spearman_p
        log_f(f"{desc} performance: {str(metrics_dict)}")
    return rv_loss if not report_metrics else metrics_dict[rv_metric_name]