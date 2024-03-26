import time
import copy
import random
import numpy as np
from tqdm import tqdm
from search.rm_search.model_helpers import ArchEvaluator, BookKeeper
import pickle
import os


"""
Random mutation that outsources the mutation part to potentially support any structure
To use this we need to write our own version of the BaseMutator
"""


class BaseMutator:

    def __init__(self, is_group_mutator=False):
        self.is_group_mutator = is_group_mutator

    def mutate(self, *args, **kwargs):
        raise NotImplementedError


class TopKAccSet:

    def __init__(self, init_arch2acc, top_k,
                 ignore_init_update=False,
                 log_f=print):
        self.top_k = top_k
        if ignore_init_update:
            self._arch2acc = init_arch2acc
            self._curr_top_archs = [v for _, (v, _) in self._arch2acc.items()]
            self._curr_top_arch_keys = set(str(v) for v in self._curr_top_archs)
        else:
            self._curr_top_archs = []
            self._curr_top_arch_keys = set()
            self._arch2acc = {}
            self.update(init_arch2acc, log_f=log_f)

    def set_arch2acc(self, new_arch2acc, log_f=print):
        log_f("Setting new global arch2acc")
        self._arch2acc = new_arch2acc
        self._curr_top_archs = [v for _, (v, _) in self._arch2acc.items()]
        self._curr_top_arch_keys = set(str(v) for v in self._curr_top_archs)
        new_top_acc = [v[1] for k, v in self._arch2acc.items()]
        new_top_acc.sort(reverse=True)
        aggr_acc = mean(new_top_acc)
        log_f("New avg acc: {} ".format(aggr_acc))
        log_f("New top-3 acc: {} ".format(new_top_acc[:3]))
        log_f("New bottom-3 acc: {} ".format(new_top_acc[-3:]))

    def is_visited_arch(self, arch):
        return str(arch) in self._arch2acc

    def top_arch2acc(self):
        return {str(v):self._arch2acc[str(v)] for v in self._curr_top_archs}

    def get_complete_arch_with_acc(self):
        rv = []
        for k, (v, acc) in self._arch2acc.items():
            rv.append((v, acc))
        rv.sort(key=lambda t: t[1], reverse=True)
        return rv

    def state_dict(self):
        return {"top_k": self.top_k,
                "_curr_top_archs": self._curr_top_archs,
                "_curr_top_arch_keys": self._curr_top_arch_keys,
                "_arch2acc": self._arch2acc}

    def load_state_dict(self, sd):
        self.top_k = sd["top_k"]
        self._curr_top_archs = sd["_curr_top_archs"]
        self._curr_top_arch_keys = sd["_curr_top_arch_keys"]
        self._arch2acc = sd["_arch2acc"]

    def get_top_arch_with_acc(self):
        rv = []
        for arch in self._curr_top_archs:
            rv.append((arch, self._arch2acc[str(arch)][1]))
        rv.sort(key=lambda t: t[1], reverse=True)
        return rv

    def get_top_archs(self):
        return [v for v in self._curr_top_archs]

    def get_top_acc_values(self):
        rv = []
        for v in self._curr_top_archs:
            rv.append(self._arch2acc[str(v)][1])
        return rv

    def update(self, new_arch2acc, log_f=print):
        # new_arch2acc expected format: {arch_key: (arch_configs, acc)}
        log_f("Updating acc set")
        old_top_keys = self._curr_top_arch_keys
        # Merge new archs with current set, overwrite existing arch's perf if duplicated
        self._arch2acc = {**self._arch2acc, **new_arch2acc}
        _archs = [v for _, v in self._arch2acc.items()]
        _archs.sort(key=lambda t: t[1], reverse=True)
        _archs = _archs[:self.top_k]
        new_top_archs = [v[0] for v in _archs]
        new_top_acc = [v[1] for v in _archs]
        assert len(new_top_archs) > 0
        new_top_keys = set(str(v) for v in new_top_archs)
        self._curr_top_archs = new_top_archs
        self._curr_top_arch_keys = set(str(v) for v in new_top_archs)
        log_f("New number of top acc archs: {}".format(len(self._curr_top_archs)))
        log_f("{} archs from previous set are kept".format(len(new_top_keys.intersection(old_top_keys))))
        log_f("{} new archs added".format(len(new_top_keys - old_top_keys)))
        aggr_acc = mean(new_top_acc)
        log_f("New avg acc: {} ".format(aggr_acc))
        log_f("New top-3 acc: {} ".format(new_top_acc[:3]))
        log_f("New bottom-3 acc: {} ".format(new_top_acc[-3:]))
        log_f("Total number of visited archs: {}".format(len(self._arch2acc)))
        return aggr_acc

    def re_map_performance(self, perf_dict, log_f=print):
        log_f("Re-mapping acc")
        missing_archs = set()
        new_arch2acc = {}
        _curr_arch2acc = self.top_arch2acc()
        for key, (arch, acc) in _curr_arch2acc.items():
            if key not in perf_dict:
                missing_archs.add(key)
            else:
                acc, lat = perf_dict[key]
                new_arch2acc[key] = (arch, acc)
        self._curr_top_archs = []
        self._curr_top_arch_keys = set()
        self._arch2acc = {}
        self.update(new_arch2acc, log_f=log_f)
        log_f("Number of missing archs: {}".format(len(missing_archs)))

    def __contains__(self, arch):
        return str(arch) in self._curr_top_arch_keys

    def __str__(self):
        return "TopKAccSet[size: {}]".format(len(self._curr_top_archs))

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._curr_top_archs)


class ArchParetoFront:

    def __init__(self, init_arch2perfs, perf_smaller_better_list,
                 num_pareto_runs=1, ignore_init_update=False, log_f=print):
        # Whether smaller is better, must match the length and the order of the input arch2perfs dict
        self.perf_smaller_is_better = tuple(perf_smaller_better_list)
        self.num_pareto_runs = num_pareto_runs
        self._arch2perfs = {}
        if not ignore_init_update:
            self.update(init_arch2perfs, log_f=log_f)

    def state_dict(self):
        return {
            "perf_smaller_is_better": self.perf_smaller_is_better,
            "num_pareto_runs": self.num_pareto_runs,
            "_arch2perfs": self._arch2perfs,
        }

    def load_state_dict(self, sd):
        self.perf_smaller_is_better = sd["perf_smaller_is_better"]
        self.num_pareto_runs = sd["num_pareto_runs"]
        self._arch2perfs = sd["_arch2perfs"]

    @property
    def complete_arch2perfs(self):
        return self._arch2perfs

    @property
    def pareto_arch2perfs(self):
        return self.get_pareto_neighborhood_arch2perfs()

    def get_custom_pareto_arch2perfs(self, num_runs=None):
        return self.get_pareto_neighborhood_arch2perfs(num_pareto_runs=num_runs)

    def is_dominated(self, perfs):
        for p_perfs in self.get_pareto_arch_perf_values():
            dominated = True
            pi = 0
            for p_perf, perf in zip(p_perfs, perfs):
                if self.perf_smaller_is_better[pi]:
                    p_perf, perf = -p_perf, -perf
                if p_perf < perf:
                    dominated = False
                    break
                pi += 1
            if dominated: return True
        return False

    @staticmethod
    def get_pareto_for_archs(arch2perfs, perf_smaller_is_better, past_arch_perfs):
        # Expected arch2perfs format: {arch_key: (arch_object, perf_values)}
        idx2key, costs = [], []
        for arch_key, v in arch2perfs.items():
            idx2key.append(arch_key)
            _, perf_vals = v
            past_arch_perfs[arch_key] = v
            arch_cost = []
            for i, perf in enumerate(perf_vals):
                # Convert every performance to smaller is better
                cost = perf
                if not perf_smaller_is_better[i]:
                    cost = - perf
                arch_cost.append(cost)
            costs.append(arch_cost)
        costs = np.asarray(costs)
        pareto_masks = ArchParetoFront.is_pareto_efficient(costs, return_mask=True)
        pareto_indices = np.argwhere(pareto_masks).squeeze(1).tolist()
        kept_arch_keys = [idx2key[idx] for idx in pareto_indices]
        pareto_arch2perfs = {key: arch2perfs[key] for key in kept_arch_keys}
        return pareto_arch2perfs

    def get_pareto_neighborhood_arch2perfs(self, num_pareto_runs=None):
        if num_pareto_runs is None:
            num_pareto_runs = self.num_pareto_runs
        cand_arch2perfs = {k: v for k, v in self._arch2perfs.items()}
        completed_runs = 0
        rv = {}
        while len(cand_arch2perfs) > 0 and completed_runs < num_pareto_runs:
            new_arch2perfs = self.get_pareto_for_archs(cand_arch2perfs, self.perf_smaller_is_better, {})
            for key, (a, perf) in new_arch2perfs.items():
                if key not in rv:
                    rv[key] = (a, perf)
                    del cand_arch2perfs[key]
            completed_runs += 1
        return rv

    def get_pareto_archs_with_perf(self, num_runs=None):
        rv = []
        for key, (arch, perfs) in \
                self.get_custom_pareto_arch2perfs(num_runs=num_runs).items():
            rv.append((arch, perfs))
        rv.sort(key=lambda t: t[1], reverse=True)
        return rv

    def get_pareto_archs(self, num_runs=None):
        rv = []
        for key, (arch, perfs) in \
                self.get_custom_pareto_arch2perfs(num_runs=num_runs).items():
            rv.append(arch)
        return rv

    def get_pareto_arch_perf_values(self, num_runs=None):
        rv = []
        for key, (arch, perfs) in \
                self.get_custom_pareto_arch2perfs(num_runs=num_runs).items():
            rv.append(perfs)
        rv.sort(reverse=True)
        return rv

    @staticmethod
    def is_pareto_efficient(costs, return_mask=True):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    def update(self, new_arch2perfs, log_f=print):
        # new_arch2perfs expected format: {arch_key: (arch_object, perf_values)}
        log_f("Updating Pareto front")
        old_pareto_keys = set(self.pareto_arch2perfs.keys())
        new_arch_keys = set(new_arch2perfs.keys())
        # Merge new archs with current front, overwrite existing arch's perf if duplicated
        self._arch2perfs = {**self._arch2perfs, **new_arch2perfs}
        new_pareto_arch2perfs = self.pareto_arch2perfs
        new_pareto_keys = set(new_pareto_arch2perfs.keys())
        log_f("New full Pareto front size: {}".format(len(new_pareto_arch2perfs)))
        log_f("{} archs from previous full front are kept".format(len(new_pareto_keys.intersection(old_pareto_keys))))
        log_f("{} new archs added".format(len(new_pareto_keys.intersection(new_arch_keys - old_pareto_keys))))
        log_f("{} total unique archs recorded".format(len(self._arch2perfs)))
        new_pareto_values = self.get_pareto_arch_perf_values(num_runs=1)
        log_f("Showing up to 5 top-most Pareto perf values: {}".format(new_pareto_values[:5]))

    def __str__(self):
        return "ArchParetoFront[total size: {}]".format(len(self._arch2perfs))

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._arch2perfs)


def mean(list_val, fallback_val=None):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ZeroDivisionError()
    return sum(list_val) / len(list_val)


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


def mutate_archs(mutator:BaseMutator, top_archs, budget,
                 max_n_per_arch_attempts=100, mutation_depth=1,
                 max_n_total_attempts=100, verbose=True):
    assert len(top_archs) > 0, "Empty top arch detected"
    if mutator.is_group_mutator:
        # This means the mutator is able to take in all the top archs and takes care of the mutation internally
        return mutator.mutate(top_archs, budget)
    else:
        bar = None
        if verbose:
            bar = tqdm(total=budget, desc="Per-arch mutation", ascii=True)
        # This means the mutator takes in one arch at a time and output one mutated arch
        rv = UniqueList(key_func=lambda a: str(a))
        n_total_attempts = 0
        while len(rv) < budget and \
            n_total_attempts < max_n_total_attempts:
            added = False
            for arch in top_archs:
                # For each arch in the top set, we try to get a unique d-edit mutation chain
                curr_arch = arch
                for d in range(mutation_depth):
                    n_block_attempts = 0
                    new_arch = mutator.mutate(curr_arch)
                    added = rv.append(new_arch)
                    while not added and \
                            n_block_attempts < max_n_per_arch_attempts:
                        new_arch = mutator.mutate(curr_arch)
                        added = rv.append(new_arch)
                        n_block_attempts += 1
                    if not added:
                        # If after many attempts but cannot find a single new arch for the current top arch, give up
                        # And move on to the next top arch
                        break
                    if bar is not None:
                        bar.update(1)
                    curr_arch = new_arch
            if added:
                # If we are able to add at least 1 new arch after trying to mutate all top archs, reset counter
                n_total_attempts = 0
        if bar is not None:
            bar.close()
        rv = rv.tolist()
        random.shuffle(rv)
        return rv[:budget]


def rm_pareto_search(mutator:BaseMutator, num_iterations, budget_per_iter,
                     pareto_front:ArchParetoFront, evaluator:ArchEvaluator,
                     book_keeper:BookKeeper, mutation_depth=1, completed_iter=0,
                     pareto_update_callback=None):
    g_start = time.time()
    pareto_save_dir = f"pareto_fronts/{book_keeper.model_name}/"
    os.makedirs(pareto_save_dir, exist_ok=True)
    for i in range(num_iterations - completed_iter):
        report_iteration = i + completed_iter + 1
        book_keeper.log("Running RM Pareto search iteration {}...".format(report_iteration))
        book_keeper.log("Budget per iter: {}".format(budget_per_iter))
        curr_pareto_archs = pareto_front.get_pareto_archs()
        new_archs = mutate_archs(mutator, curr_pareto_archs, budget_per_iter,
                                 mutation_depth=mutation_depth)
        mutate_arch_keys = {str(a) for a in new_archs} # Assume str() gives the key
        book_keeper.log("Num mutated unique archs: {}".format(len(mutate_arch_keys)))
        new_arch2perfs = {}
        book_keeper.log("Evaluating {} new archs".format(len(new_archs)))
        for arch in tqdm(new_archs, desc="Getting arch perfs", ascii=True):
            key = str(arch)
            if key in new_arch2perfs:
                continue
            perf_values = evaluator.get_perf_values(arch)
            new_arch2perfs[key] = (arch, perf_values)
        if len(new_arch2perfs) < budget_per_iter:
            book_keeper.log("Budget goal {} not reached: {}".format(budget_per_iter, len(new_arch2perfs)))
        gen_acc_values = [v[0] for _, (_, v) in new_arch2perfs.items()]
        gen_lat_values = [v[1] for _, (_, v) in new_arch2perfs.items()]
        book_keeper.log("Max acc in generated archs: {}".format(max(gen_acc_values)))
        book_keeper.log("Min acc in generated archs: {}".format(min(gen_acc_values)))
        book_keeper.log("Max lat in generated archs: {}".format(max(gen_lat_values)))
        book_keeper.log("Min lat in generated archs: {}".format(min(gen_lat_values)))
        book_keeper.log("Num archs evaluated so far: {}".format(evaluator.num_eval))
        pareto_front.update(new_arch2perfs, log_f=book_keeper.log)
        if pareto_update_callback is not None:
            pareto_update_callback(pareto_front, report_iteration, book_keeper.model_name)
        book_keeper.checkpoint_state_dict(pareto_front.state_dict(),
                                          "{}_pareto_front.pkl".format(book_keeper.model_name))
        pareto_perf_values = pareto_front.get_pareto_arch_perf_values(num_runs=1)
        acc_values = [t[0] for t in pareto_perf_values]
        lat_values = [t[1] for t in pareto_perf_values]
        book_keeper.log("Max acc in top-most Pareto: {}".format(max(acc_values)))
        book_keeper.log("Min acc in top-most Pareto: {}".format(min(acc_values)))
        book_keeper.log("Max lat in top-most Pareto: {}".format(max(lat_values)))
        book_keeper.log("Min lat in top-most Pareto: {}".format(min(lat_values)))
        book_keeper.log("New full Pareto front size: {}, "
                        "new_arch2perfs len: {}".format(len(pareto_front.pareto_arch2perfs),
                                                        len(new_arch2perfs)))
        example_archs = [a for _, (a, _) in new_arch2perfs.items()][:3]
        book_keeper.log("Example generated archs:")
        for a in example_archs:
            book_keeper.log(str(a))
        #if i > (num_iterations - completed_iter - 1):
        with open(f"{pareto_save_dir}iter_{i}.pkl", "wb") as f:
            pickle.dump(pareto_front.get_pareto_archs_with_perf(num_runs=1), f, protocol=4)
        archs_w_perf = pareto_front.get_pareto_archs_with_perf(num_runs=1)
        for arch, perf in archs_w_perf:
            acc, lat = perf
            book_keeper.log("{}|{}|{}".format(str(arch), acc, lat))
    #with open(f"{pareto_save_dir}iter_{i}.pkl", "wb") as f:
    #    pickle.dump(pareto_front.get_pareto_archs_with_perf(num_runs=1), f, protocol=4)
    book_keeper.log("Num archs evaluated so far: {}".format(evaluator.num_eval))
    book_keeper.log("Search seconds: {}".format(time.time() - g_start))


def rm_cons_acc_search(mutator:BaseMutator, num_iterations, budget_per_iter,
                       acc_set:TopKAccSet, evaluator:ArchEvaluator,
                       max_cons_score, cons_predictor,
                       book_keeper:BookKeeper,
                       mutation_depth=1, completed_iter=0,
                       update_callback=None):
    g_start = time.time()
    for i in range(num_iterations - completed_iter):
        report_iteration = i + completed_iter + 1
        book_keeper.log("Running RM cons acc search iteration {}...".format(report_iteration))
        book_keeper.log("Budget per iter: {}".format(budget_per_iter))
        curr_top_archs = acc_set.get_top_archs()
        new_archs = mutate_archs(mutator, curr_top_archs, budget_per_iter,
                                 mutation_depth=mutation_depth)

        if report_iteration == 1:
            # Remove any init arch in top set that do not satisfy the constraints
            filtered_arch2acc = {}
            arch_w_acc = acc_set.get_complete_arch_with_acc()
            book_keeper.log("arch2acc size before filtering: {}".format(len(arch_w_acc)))
            for arch, acc in tqdm(arch_w_acc, desc="Filtering curr top acc by constraints", ascii=True):
                cons_score = cons_predictor(arch)
                if cons_score - max_cons_score > 1e-5:
                    continue
                filtered_arch2acc[str(arch)] = (arch, acc)
            acc_set.set_arch2acc(filtered_arch2acc)
            arch_w_acc = acc_set.get_complete_arch_with_acc()
            book_keeper.log("arch2acc size after filtering: {}".format(len(arch_w_acc)))

        mutate_arch_keys = {str(a) for a in new_archs}
        book_keeper.log("Mutated unique valid archs: {}".format(len(mutate_arch_keys)))
        new_arch2acc = {}
        book_keeper.log("Evaluating {} new archs".format(len(new_archs)))
        n_not_match = 0
        for arch in tqdm(new_archs, desc="Getting arch perfs", ascii=True):
            key = str(arch)
            if key in new_arch2acc:
                continue
            cons_score = cons_predictor(arch)
            if cons_score - max_cons_score > 1e-5:
                n_not_match += 1
                continue
            # By convention, lat returned by the evaluator will be ignored
            acc, _ = evaluator.get_perf_values(arch)
            new_arch2acc[key] = (arch, acc)
        book_keeper.log("{} out of {} generated archs did not match constraint".format(n_not_match,
                                                                                       len(new_archs)))
        if len(new_arch2acc) < budget_per_iter:
            book_keeper.log("Budget goal {} not reached: {}".format(budget_per_iter, len(new_arch2acc)))
        acc_set.update(new_arch2acc, log_f=book_keeper.log)
        if update_callback is not None:
            update_callback(acc_set, report_iteration, book_keeper.model_name)
        book_keeper.checkpoint_state_dict(acc_set.state_dict(),
                                          "{}_top_acc_set.pkl".format(book_keeper.model_name))
        gen_acc_values = [acc for _, (a, acc) in new_arch2acc.items()]
        book_keeper.log("Max acc in generated archs: {}".format(max(gen_acc_values)))
        book_keeper.log("Min acc in generated archs: {}".format(min(gen_acc_values)))
        acc_vals = acc_set.get_top_acc_values()
        book_keeper.log("Best acc so far: {}".format(max(acc_vals)))
        book_keeper.log("New top acc set size: {}, new_arch2acc len: {}".format(len(acc_set), len(new_arch2acc)))
        example_archs = [a for _, (a, _) in new_arch2acc.items()][:3]
        book_keeper.log("Example generated archs:")
        for a in example_archs:
            book_keeper.log(str(a))
        book_keeper.log("")
    book_keeper.log("Num archs evaluated so far: {}".format(evaluator.num_eval))
    book_keeper.log("Search seconds: {}".format(time.time() - g_start))
