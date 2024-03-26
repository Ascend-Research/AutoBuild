import os
import torch
from tqdm import tqdm
from os.path import sep as P_SEP
from search.rm_search.constants import *
from search.rm_search.utils.model_utils import model_load, model_save



def write_line_to_file(s, f_path="progress.txt", new_file=False, verbose=False):
    code = "w" if new_file else "a"
    if verbose: print(s)
    with open(f_path, code, encoding='utf-8') as f:
        f.write(s)
        f.write("\n")


class BookKeeper:

    def __init__(self, log_file_name, model_name,
                 saved_models_dir="./saved_models/",
                 verbose=True,
                 init_eval_perf=0., init_eval_iter=0,
                 eval_perf_comp_func=lambda old, new: new > old,  # how to judge which eval perf is better
                 saved_model_file=None,
                 logs_dir="./logs/"):
        self.saved_models_dir = saved_models_dir
        self.logs_dir = logs_dir
        self.model_name = model_name
        self.log_file = P_SEP.join([self.logs_dir, log_file_name])
        self.verbose = verbose
        self.curr_best_eval_perf = init_eval_perf
        self.curr_best_eval_iter = init_eval_iter
        self.eval_perf_comp_func = eval_perf_comp_func
        self._init_eval_perf = init_eval_perf
        self._init_eval_iter = init_eval_iter
        self.saved_model_file = saved_model_file
        if not os.path.isdir(self.logs_dir):
            os.mkdir(self.logs_dir)
        if not os.path.isdir(self.saved_models_dir):
            os.mkdir(self.saved_models_dir)
        self.log("Model name: {}".format(self.model_name))
        self.log("Saved models dir: {}".format(self.saved_models_dir))
        self.log("Log dir: {}".format(self.logs_dir))

    def reset_eval_perfs(self, eval_perf=None, eval_iter=None):
        self.curr_best_eval_perf = self._init_eval_perf if eval_perf is None else eval_perf
        self.curr_best_eval_iter = self._init_eval_iter if eval_iter is None else eval_iter

    def log(self, msg, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if not isinstance(msg, str):
            msg = str(msg)
        write_line_to_file(msg, self.log_file, verbose=verbose)

    def report_curr_best(self):
        self.log("Model name: {}".format(self.model_name))
        self.log("curr_best_eval_perf: {}, curr_best_eval_iter: {}".format(self.curr_best_eval_perf,
                                                                           self.curr_best_eval_iter))

    def load_model_checkpoint_w_suffix(self, model, suffix, model_key=CHKPT_MODEL, skip_eval_perfs=False,
                                       allow_silent_fail=True):
        checkpoint_file = P_SEP.join([self.saved_models_dir, self.model_name + suffix])
        return self.load_model_checkpoint(model, model_key=model_key, checkpoint_file=checkpoint_file,
                                          skip_eval_perfs=skip_eval_perfs, allow_silent_fail=allow_silent_fail)

    def load_model_checkpoint(self, model, model_key=CHKPT_MODEL, checkpoint_file=None,
                              skip_eval_perfs=False, allow_silent_fail=True):
        if checkpoint_file is None:
            checkpoint_file = self.saved_model_file
        if os.path.isfile(checkpoint_file):
            self.log("Found checkpoint: {}, loading".format(checkpoint_file))
            sd = model_load(checkpoint_file)
            model.load_state_dict(sd[model_key])
            self.log("Found best_eval_perf: {}, best_eval_iter: {}".format(sd[CHKPT_BEST_EVAL_RESULT],
                                                                           sd[CHKPT_BEST_EVAL_ITERATION]))
            if not skip_eval_perfs:
                self.curr_best_eval_perf = sd[CHKPT_BEST_EVAL_RESULT]
                self.curr_best_eval_iter = sd[CHKPT_BEST_EVAL_ITERATION]
                self.log("Loaded curr_best_eval_perf: {}, curr_best_eval_iter: {}".format(self.curr_best_eval_perf,
                                                                                          self.curr_best_eval_iter))
            if CHKPT_ITERATION in sd:
                completed_iterations = sd[CHKPT_ITERATION]
                self.log("Completed iterations: {}".format(completed_iterations))
                return completed_iterations
            elif CHKPT_COMPLETED_EPOCHS in sd:
                completed_epochs = sd[CHKPT_COMPLETED_EPOCHS]
                self.log("Completed epochs: {}".format(completed_epochs))
                return completed_epochs
        elif not allow_silent_fail:
            raise FileNotFoundError("checkpoint_file: {} not found".format(checkpoint_file))
        return 0

    def load_model_optim_checkpoint(self, model, optimizer, model_key=CHKPT_MODEL, optimizer_key=CHKPT_OPTIMIZER,
                                    checkpoint_file=None, skip_eval_perfs=False, allow_silent_fail=True):
        if checkpoint_file is None:
            checkpoint_file = self.saved_model_file
        if os.path.isfile(checkpoint_file):
            self.log("Found checkpoint: {}, loading".format(checkpoint_file))
            sd = model_load(checkpoint_file)
            model.load_state_dict(sd[model_key])
            optimizer.load_state_dict(sd[optimizer_key])
            self.log("Found best_eval_perf: {}, best_eval_iter: {}".format(sd[CHKPT_BEST_EVAL_RESULT],
                                                                           sd[CHKPT_BEST_EVAL_ITERATION]))
            if not skip_eval_perfs:
                self.curr_best_eval_perf = sd[CHKPT_BEST_EVAL_RESULT]
                self.curr_best_eval_iter = sd[CHKPT_BEST_EVAL_ITERATION]
                self.log("Loaded curr_best_eval_perf: {}, curr_best_eval_iter: {}".format(self.curr_best_eval_perf,
                                                                                          self.curr_best_eval_iter))
            if CHKPT_ITERATION in sd:
                completed_iterations = sd[CHKPT_ITERATION]
                self.log("Completed iterations: {}".format(completed_iterations))
                return completed_iterations
            elif CHKPT_COMPLETED_EPOCHS in sd:
                completed_epochs = sd[CHKPT_COMPLETED_EPOCHS]
                self.log("Completed epochs: {}".format(completed_epochs))
                return completed_epochs
        elif not allow_silent_fail:
            raise FileNotFoundError("checkpoint_file: {} not found".format(checkpoint_file))
        return 0

    def load_state_dict_checkpoint(self, obj, checkpoint_file, allow_silent_fail=False):
        assert hasattr(obj, "load_state_dict")
        import pickle
        checkpoint_file = P_SEP.join([self.saved_models_dir, checkpoint_file])
        if os.path.isfile(checkpoint_file):
            self.log("Found state dict checkpoint: {}, loading".format(checkpoint_file))
            with open(checkpoint_file, "rb") as f:
                sd = pickle.load(f)
            obj.load_state_dict(sd)
        elif not allow_silent_fail:
            raise FileNotFoundError("checkpoint_file: {} not found".format(checkpoint_file))
        return None

    def load_object_checkpoint(self, checkpoint_file, allow_silent_fail=False):
        import pickle
        checkpoint_file = P_SEP.join([self.saved_models_dir, checkpoint_file])
        if os.path.isfile(checkpoint_file):
            self.log("Found object checkpoint: {}, loading".format(checkpoint_file))
            with open(checkpoint_file, "rb") as f:
                obj = pickle.load(f)
            return obj
        elif not allow_silent_fail:
            raise FileNotFoundError("checkpoint_file: {} not found".format(checkpoint_file))
        return None

    def checkpoint_model(self, suffix, iteration, model, optimizer, eval_perf=None,
                         model_key=CHKPT_MODEL, optim_key=CHKPT_OPTIMIZER,
                         update_eval_perf=True):
        if eval_perf is not None:
            if self.eval_perf_comp_func(self.curr_best_eval_perf, eval_perf):
                if update_eval_perf:
                    self.curr_best_eval_perf = eval_perf
                    self.curr_best_eval_iter = iteration
                self._checkpoint_model(suffix, iteration, model, optimizer,
                                       model_key, optim_key)
        else:
            self._checkpoint_model(suffix, iteration, model, optimizer,
                                   model_key, optim_key)

    def _checkpoint_model(self, suffix, iteration, model, optimizer,
                          model_key=CHKPT_MODEL, optim_key=CHKPT_OPTIMIZER):
        file_path = P_SEP.join([self.saved_models_dir, self.model_name + suffix])
        sv = {
            CHKPT_ITERATION: iteration,
            model_key: model.state_dict(),
            optim_key: optimizer.state_dict() if optimizer is not None else None,
            CHKPT_BEST_EVAL_RESULT: self.curr_best_eval_perf,
            CHKPT_BEST_EVAL_ITERATION: self.curr_best_eval_iter,
        }
        self.log("Saving model to {}, please do not terminate".format(file_path))
        model_save(file_path, sv)
        self.log("Checkpoint complete")

    def checkpoint_object(self, obj, checkpoint_file):
        import pickle
        checkpoint_file = P_SEP.join([self.saved_models_dir, checkpoint_file])
        self.log("Saving object to {}, please do not terminate".format(checkpoint_file))
        with open(checkpoint_file, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        self.log("Checkpoint complete")

    def checkpoint_state_dict(self, sd, checkpoint_file):
        import pickle
        checkpoint_file = P_SEP.join([self.saved_models_dir, checkpoint_file])
        self.log("Saving state dict to {}, please do not terminate".format(checkpoint_file))
        with open(checkpoint_file, "wb") as f:
            pickle.dump(sd, f, protocol=4)
        self.log("Checkpoint complete")


class ArchEvaluator:

    def __init__(self, enable_perf_memo=True):
        self.enable_perf_memo = enable_perf_memo
        self.perf_memo = {}
        self.num_eval = 0

    def get_perf_values(self, state):
        # Assuming this is the expensive part
        raise NotImplementedError("Should override in child classes")


class SimpleCachedData(torch.utils.data.Dataset):

    def __init__(self, feature_tsr_list, target_tsr_list):
        assert len(feature_tsr_list) == len(target_tsr_list)
        self.feature_tsr_list = feature_tsr_list
        self.target_tsr_list = target_tsr_list

    def __getitem__(self, index):
        x = self.feature_tsr_list[index]
        y = self.target_tsr_list[index]
        return x, y

    def __len__(self):
        return len(self.feature_tsr_list)


def get_simple_cached_class_loader(ext_loader, batch_size,
                                   num_workers=6, shuffle=True,
                                   pin_memory=True, drop_last=False):
    feature_tsr_list = []
    target_tsr_list = []
    for feature, target in tqdm(ext_loader,
                                desc="Extracting instances", ascii=True):
        batch_size = feature.shape[0]
        assert batch_size == target.shape[0]
        for bi in range(batch_size):
            feature_tsr_list.append(feature[bi, :])
            target_tsr_list.append(target[bi])
    dset = SimpleCachedData(feature_tsr_list, target_tsr_list)
    loader = torch.utils.data.DataLoader(dset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=shuffle,
                                         pin_memory=pin_memory,
                                         drop_last=drop_last)
    return loader
