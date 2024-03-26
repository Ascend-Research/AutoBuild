import copy
import random
import sys
from search.rm_search.ea.model_rm_custom import BaseMutator
from search.rm_search.ofa_profile.arch_gen import get_ofa_resnet_arch_id
from search.rm_search.ofa_profile.constants import OFA_RES_STAGE_MAX_N_BLOCKS, OFA_RES_STAGE_MIN_N_BLOCKS
from search.rm_search.constants import DIR_TO_APPEND

sys.path.append(DIR_TO_APPEND)


"""
Contains different types of arch mutator for OFA
"""


def get_mutate_stage_inds(probs):
    # For each stage in a network, there's some chance for mutation
    # Must mutate at least 1 stage
    sel_inds = []
    while len(sel_inds) == 0:
        for si in range(len(probs)):
            if random.random() < probs[si]:
                sel_inds.append(si)
    return sel_inds


def get_valid_stage_mutations(blocks, min_n_blocks, max_n_blocks, mutations):
    mutations = set(mutations)
    assert min_n_blocks <= len(blocks) <= max_n_blocks
    if len(blocks) == max_n_blocks and \
        "add" in mutations:
        mutations.remove("add")
    if len(blocks) == min_n_blocks and \
        "remove" in mutations:
        mutations.remove("remove")
    assert len(mutations) > 0
    return list(mutations)


def apply_mutation_to_stage(blocks, mutation, candidates):
    new_blocks = [copy.deepcopy(b) for b in blocks]
    if mutation == "add":
        new_block = random.choice(candidates)
        insert_idx = random.choice(list(range(len(new_blocks) + 1)))
        new_blocks.insert(insert_idx, new_block)
    elif mutation == "remove":
        remove_idx = random.choice(list(range(len(new_blocks))))
        new_blocks.pop(remove_idx)
    elif mutation == "swap":
        # Swap only 1 block in a stage
        swap_idx = random.choice(list(range(len(new_blocks))))
        stage_level_candidates = set(candidates)
        if new_blocks[swap_idx] in stage_level_candidates:
            stage_level_candidates.remove(new_blocks[swap_idx])
        new_block = random.choice(list(stage_level_candidates))
        new_blocks[swap_idx] = new_block
    else:
        raise ValueError("Unknown mutation: {}".format(mutation))
    return new_blocks


class OFAStageLevelMutator(BaseMutator):
    """
    Performs mutation on 1 OFA-PN/MBV3 arch
    Supports stage-wise candidate spaces
    """
    def __init__(self, min_stage_block_counts,
                 max_stage_block_counts,
                 stage_level_candidates,
                 stage_mutate_probs,
                 stage_mutate_types=("add", "remove", "swap")):
        super(OFAStageLevelMutator, self).__init__(is_group_mutator=False)
        assert len(min_stage_block_counts) == len(max_stage_block_counts)
        assert len(stage_level_candidates) == len(stage_mutate_probs)
        assert len(stage_level_candidates) == len(max_stage_block_counts)
        self.min_stage_block_counts = min_stage_block_counts
        self.max_stage_block_counts = max_stage_block_counts
        self.stage_level_candidates = stage_level_candidates
        self.stage_mutate_probs = stage_mutate_probs
        self.stage_mutate_types = stage_mutate_types

    def mutate(self, net_configs):
        """
        :param net_configs: Should match the definition of the search space
        """
        assert len(net_configs) == len(self.stage_level_candidates) == len(self.stage_mutate_probs)
        mutate_stage_inds = get_mutate_stage_inds(self.stage_mutate_probs)
        mutate_stage_inds = set(mutate_stage_inds)
        new_net_configs = []
        for si, blocks in enumerate(net_configs):
            if si not in mutate_stage_inds:
                new_net_configs.append([copy.deepcopy(b) for b in blocks])
                continue
            mutations = get_valid_stage_mutations(blocks,
                                                  self.min_stage_block_counts[si],
                                                  self.max_stage_block_counts[si],
                                                  self.stage_mutate_types)
            mutation = random.choice(mutations)
            new_blocks = apply_mutation_to_stage(blocks, mutation, self.stage_level_candidates[si])
            assert self.min_stage_block_counts[si] <= len(new_blocks) <= self.max_stage_block_counts[si], \
                "Invalid new_blocks: {}".format(str(new_blocks))
            new_net_configs.append(new_blocks)
        return new_net_configs
    

class OFAStageWholeMutator(BaseMutator):
    """
    Performs mutation on 1 OFA-PN/MBV3 arch
    Supports stage-wise candidate spaces
    """
    def __init__(self, stage_level_candidates,
                 stage_mutate_probs): 
#                 stage_mutate_types=("add", "remove", "swap")):
        super(OFAStageWholeMutator, self).__init__(is_group_mutator=False)
        #assert len(stage_level_candidates) == len(stage_mutate_probs)
        self.stage_level_candidates = stage_level_candidates
        self.stage_mutate_probs = stage_mutate_probs
        #self.stage_mutate_types = stage_mutate_types

    def mutate(self, net_configs):
        """
        :param net_configs: Should match the definition of the search space
        """
        #assert len(net_configs) == len(self.stage_level_candidates) == len(self.stage_mutate_probs)
        mutate_stage_inds = get_mutate_stage_inds(self.stage_mutate_probs)
        mutate_stage_inds = set(mutate_stage_inds)
        new_net_configs = []
        for si, blocks in enumerate(net_configs):
            #print(si)
            if si not in mutate_stage_inds:
                new_net_configs.append([copy.deepcopy(b) for b in blocks])
                if si == 4 and len(self.stage_mutate_probs) == 6:
                    new_net_configs.append(net_configs[-1])
                    break
                continue
            elif si == 4 and len(self.stage_mutate_probs) == 6:
                stages = random.choice(self.stage_level_candidates[si])
                new_net_configs.append(stages[0])
                new_net_configs.append(stages[1])
                break
            else:
                new_net_configs.append(random.choice(self.stage_level_candidates[si]))
        return new_net_configs


def get_resnet_stage_args(d_list, e_list, w_list):
    """
    Turns a OFA dict-based net_configs into a stage-wise net setup
    This makes per-stage mutation easier
    In the end we'll need to turn it back into a dict-based config
    """
    stage_args = []
    max_n_stage_blocks = [1] + list(OFA_RES_STAGE_MAX_N_BLOCKS)
    for si in range(len(max_n_stage_blocks)):
        # First stage is special, it's essentially the stem
        if si == 0:
            stage_dict = {
                "d": d_list[0],
                "w": w_list[:2],
            }
            d_list = d_list[1:]
            w_list = w_list[2:]
        else:
            stage_w = w_list[0]
            stage_e_list = e_list[:max_n_stage_blocks[si]]
            stage_dict = {
                "d": d_list[0],
                "e": stage_e_list,
                "w": stage_w,
            }
            e_list = e_list[max_n_stage_blocks[si]:]
            del w_list[0]
            del d_list[0]
        stage_args.append(stage_dict)
    return stage_args


def recover_ofa_resnet_args(stage_args):
    """
    Performs the reverse process of get_resnet_stage_args(...)
    """
    d_list, e_list, w_list = [], [], []
    for si, sd in enumerate(stage_args):
        if si == 0:
            # First stage is the stem
            d = sd["d"]
            stem_w_list = sd["w"]
            d_list.append(d)
            w_list.extend(stem_w_list)
        else:
            d = sd["d"]
            stage_e_list = sd["e"]
            stage_w = sd["w"]
            d_list.append(d)
            e_list.extend(stage_e_list)
            w_list.append(stage_w)
    return d_list, e_list, w_list


class OFAResNetArchWrapper:
    """
    The purpose of this class is to make duplicate checking easier for OFA-ResNet
    """
    def __init__(self, d_list, e_list, w_list,
                 max_n_stage_blocks):
        self.d_list = d_list
        self.e_list = e_list
        self.w_list = w_list
        # Stem considered a special stage, must be included
        self.max_n_stage_blocks = max_n_stage_blocks

    def __deepcopy__(self, memodict={}):
        rv = OFAResNetArchWrapper(copy.deepcopy(self.d_list),
                                  copy.deepcopy(self.e_list),
                                  copy.deepcopy(self.w_list),
                                  self.max_n_stage_blocks)
        return rv

    def __str__(self):
        # Here we must chop the stem val in self.max_n_stage_blocks to match the id func requirements
        return get_ofa_resnet_arch_id(self.d_list, self.e_list, self.w_list,
                                      self.max_n_stage_blocks[1:])


def get_valid_resnet_stage_mutations(stage_idx, stage_dict,
                                     min_n_blocks, max_n_blocks,
                                     e_cands, w_cands,
                                     mutations):
    mutations = set(mutations)
    if stage_idx == 0:
        # Means we are mutating the stem, which requires special handling
        if "add" in mutations:
            mutations.remove("add")
        if "remove" in mutations:
            mutations.remove("remove")
        if "swap_e" in mutations:
            mutations.remove("swap_e")
        mutations.add("swap_stem")
    else:
        d = stage_dict["d"] + OFA_RES_STAGE_MIN_N_BLOCKS[stage_idx - 1]
        assert d <= max_n_blocks
        assert d >= min_n_blocks
        if d == max_n_blocks and \
            "add" in mutations:
            mutations.remove("add")
        if d == min_n_blocks and \
            "remove" in mutations:
            mutations.remove("remove")
        if len(e_cands) <= 1 and \
            "swap_e" in mutations:
            mutations.remove("swap_e")
        if len(w_cands) <= 1 and \
            "swap_w" in mutations:
            mutations.remove("swap_w")
    assert len(mutations) > 0
    return list(mutations)


def apply_mutation_to_resnet_stage(stage_idx, stage_dict, mutation,
                                   stem_d_choices, e_choices, w_choices):
    if stage_idx == 0:
        # Means we are mutating the stem, which requires special handling
        if mutation == "swap_stem":
            prev_d = stage_dict["d"]
            # For stem d there are essentially two choices, skip or no skip
            if prev_d == max(stem_d_choices):
                new_d = 0
            else:
                new_d = max(stem_d_choices)
            new_stage_dict = {
                "d": new_d,
                "w": copy.deepcopy(stage_dict["w"])
            }
        elif mutation == "swap_w":
            # Re-sample a new set of w values
            new_w = random.choices(w_choices, k=len(stage_dict["w"]))
            n_attempts = 0
            while str(new_w) == str(stage_dict["w"]) and n_attempts < 1000:
                new_w = random.choices(w_choices, k=len(stage_dict["w"]))
                n_attempts += 1
            assert n_attempts < 1000, "Weird w for stem: {}".format(str(stage_dict["w"]))
            new_stage_dict = {
                "d": stage_dict["d"],
                "w": new_w
            }
        else:
            raise ValueError("Unknown mutation: {}".format(mutation))
    else:
        d, e_list, w = stage_dict["d"], stage_dict["e"], stage_dict["w"]
        new_e_list = copy.deepcopy(e_list)
        new_d, new_w = d, w
        if mutation == "add":
            n_blocks = d + OFA_RES_STAGE_MIN_N_BLOCKS[stage_idx - 1]
            compact_e_list = new_e_list[:n_blocks]
            insert_idx = random.choice(list(range(n_blocks + 1)))
            compact_e_list.insert(insert_idx, random.choice(e_choices))
            while len(compact_e_list) < len(new_e_list):
                compact_e_list.append(e_choices[0])
            assert len(compact_e_list) == len(new_e_list)
            new_e_list = compact_e_list
            new_d = d + 1
        elif mutation == "remove":
            n_blocks = d + OFA_RES_STAGE_MIN_N_BLOCKS[stage_idx - 1]
            compact_e_list = new_e_list[:n_blocks]
            remove_idx = random.choice(list(range(n_blocks)))
            del compact_e_list[remove_idx]
            while len(compact_e_list) < len(new_e_list):
                compact_e_list.append(e_choices[0])
            assert len(compact_e_list) == len(new_e_list)
            new_e_list = compact_e_list
            new_d = d - 1
            assert new_d >= 0
        elif mutation == "swap_e":
            # Swap for only 1 block in a stage
            n_blocks = d + OFA_RES_STAGE_MIN_N_BLOCKS[stage_idx - 1]
            swap_idx = random.choice(list(range(n_blocks)))
            curr_e = str(new_e_list[swap_idx])
            e_choices = [e for e in e_choices if str(e) != curr_e]
            new_e = random.choice(e_choices)
            new_e_list[swap_idx] = new_e
        elif mutation == "swap_w":
            # Swap for a stage
            curr_w = str(w)
            w_choices = [w for w in w_choices if str(w) != curr_w]
            new_w = random.choice(w_choices)
        else:
            raise ValueError("Unknown mutation: {}".format(mutation))
        new_stage_dict = {
            "d": new_d,
            "e": new_e_list,
            "w": new_w
        }
    return new_stage_dict


class OFAResNetStageLevelMutator(BaseMutator):
    """
    Performs mutation on 1 OFA-ResNet arch
    Supports stage-wise candidate spaces
    """
    def __init__(self, min_stage_block_counts, # Stem considered a special stage, must be included
                 max_stage_block_counts, # Stem considered a special stage, must be included
                 stage_mutate_probs,
                 stage_level_w_choices,
                 stage_level_e_choices,
                 stem_d_choices,
                 stage_mutate_types=("add", "remove", "swap_e", "swap_w")):
        super(OFAResNetStageLevelMutator, self).__init__(is_group_mutator=False)
        assert len(min_stage_block_counts) == len(max_stage_block_counts)
        assert len(min_stage_block_counts) == len(stage_mutate_probs)
        self.min_stage_block_counts = min_stage_block_counts
        self.max_stage_block_counts = max_stage_block_counts
        self.stage_mutate_probs = stage_mutate_probs
        self.stage_level_w_choices = stage_level_w_choices
        self.stage_level_e_choices = stage_level_e_choices
        self.stem_d_choices = stem_d_choices
        self.stage_mutate_types = stage_mutate_types

    @staticmethod
    def _check_new_net_configs(new:OFAResNetArchWrapper,
                               old:OFAResNetArchWrapper):
        assert len(new.d_list) == len(old.d_list), " {} vs. {}".format(new.d_list, old.d_list)
        assert len(new.e_list) == len(old.e_list), " {} vs. {}".format(new.e_list, old.e_list)
        assert len(new.w_list) == len(old.w_list), " {} vs. {}".format(new.w_list, old.w_list)

    def mutate(self, net_configs:OFAResNetArchWrapper):
        assert all(v1 == v2 for v1, v2 in zip(self.max_stage_block_counts,
                                              net_configs.max_n_stage_blocks))
        mutate_stage_inds = get_mutate_stage_inds(self.stage_mutate_probs)
        mutate_stage_inds = set(mutate_stage_inds)
        new_net_configs = copy.deepcopy(net_configs)
        stage_args = get_resnet_stage_args(new_net_configs.d_list,
                                           new_net_configs.e_list,
                                           new_net_configs.w_list)
        new_stage_args = []
        for si, stage_dict in enumerate(stage_args):
            if si not in mutate_stage_inds:
                new_stage_args.append(stage_dict)
                continue
            mutations = get_valid_resnet_stage_mutations(si, stage_dict,
                                                         self.min_stage_block_counts[si],
                                                         self.max_stage_block_counts[si],
                                                         self.stage_level_e_choices[si],
                                                         self.stage_level_w_choices[si],
                                                         self.stage_mutate_types)
            mutation = random.choice(mutations)
            # Mutation is performed directly on stage-wise dicts
            new_stage_dict = apply_mutation_to_resnet_stage(si, stage_dict, mutation,
                                                            self.stem_d_choices,
                                                            self.stage_level_e_choices[si],
                                                            self.stage_level_w_choices[si])
            new_stage_args.append(new_stage_dict)
        assert len(stage_args) == len(new_stage_args), \
            "{} vs. {}".format(len(stage_args), len(new_stage_args))
        new_d_list, new_e_list, new_w_list = recover_ofa_resnet_args(new_stage_args)
        new_net_configs.d_list = new_d_list
        new_net_configs.e_list = new_e_list
        new_net_configs.w_list = new_w_list
        self._check_new_net_configs(new_net_configs, net_configs)
        return new_net_configs


class OFAResNetStageWholeMutator(BaseMutator):
    """
    Performs mutation on 1 OFA-ResNet arch
    Supports stage-wise candidate spaces
    """
    def __init__(self, stage_mutate_probs,
                 stage_units):
        super(OFAResNetStageWholeMutator, self).__init__(is_group_mutator=False)
        self.stage_mutate_probs = stage_mutate_probs
        self.stage_units = stage_units

    @staticmethod
    def _merge_cfg(original_cfg, new_dict):
        new_cfg_list = []
        for i in range(4):
            if i in new_dict.keys():
                new_cfg_list.append(new_dict[i])
            else:
                new_cfg_list.append(original_cfg)
        return assemble_whole_net_cfg(*new_cfg_list)


    def mutate(self, net_configs):
        mutate_stage_inds = get_mutate_stage_inds(self.stage_mutate_probs)
        mutate_stage_inds = set(mutate_stage_inds)
        new_stage_args = {}
        for si, stage_dict in enumerate(self.stage_units):
            if si not in mutate_stage_inds:
                new_stage_args[si] = net_configs #.append(stage_dict)
                continue
            mutation = random.choice(self.stage_units[si])
            new_stage_args[si] = mutation
        return self._merge_cfg(net_configs, new_stage_args)
