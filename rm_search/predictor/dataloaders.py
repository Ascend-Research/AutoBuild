import math
import torch
import random
import collections
from tqdm import tqdm
from search.rm_search.constants import *
from search.rm_search.utils.misc_utils import UniqueDict


class RNNShapeRegressDataLoader:

    def __init__(self, batch_size, data, verbose=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.curr_batch_idx = 0
        self.batches = []
        self._build_batches(data)
        self.n_batches = len(self.batches)

    def _build_batches(self, data):
        # Data format: (node_inds, shape_vals, tgt_val)
        self.batches = []
        # Partition data by the number of nodes in graph
        bins = collections.defaultdict(list)
        for node_inds, shape_vals, tgt_val in data:
            bins[len(node_inds)].append((node_inds, shape_vals, tgt_val))
        n_batches = 0
        # Compute the number of batches in total
        for _, instances in bins.items():
            n_batches += math.ceil(len(instances) / self.batch_size)
        # Build actual batches
        bar = None
        if self.verbose:
            bar = tqdm(total=n_batches, desc="Building batches", ascii=True)
        for k, data_list in bins.items():
            idx = 0
            while idx < len(data_list):
                batch_list = data_list[idx:idx + self.batch_size]
                batch_node_feature_list = []
                batch_shape_list = []
                batch_unique_str_id_set = set()
                batch_tgt = []
                for inst in batch_list:
                    node_idx_list, shape_vals, tgt = inst # Expected data format
                    node_feature_tsr = torch.LongTensor(node_idx_list)
                    batch_node_feature_list.append(node_feature_tsr.unsqueeze(0))
                    shape_tsr = torch.FloatTensor(shape_vals)
                    batch_shape_list.append(shape_tsr.unsqueeze(0))
                    unique_str_id = "|".join([str(node_idx_list), str(shape_vals)])
                    batch_unique_str_id_set.add(unique_str_id)
                    batch_tgt.append(tgt)
                batch_tgt = torch.FloatTensor(batch_tgt)
                batch_node_tsr = torch.cat(batch_node_feature_list, dim=0)
                batch_shape_tsr = torch.cat(batch_shape_list, dim=0)
                batch = UniqueDict([
                    (DK_BATCH_SIZE, len(batch_node_feature_list)),
                    (DK_BATCH_NODE_FEATURE_TSR, batch_node_tsr),
                    (DK_BATCH_NODE_SHAPE_TSR, batch_shape_tsr),
                    (DK_BATCH_UNIQUE_STR_ID_SET, batch_unique_str_id_set),
                    (DK_BATCH_TARGET_TSR, batch_tgt),
                ])
                if len(batch_unique_str_id_set) < len(batch_list):
                    print("Collected {} unique features but batch size is {}".format(len(batch_unique_str_id_set),
                                                                                     len(batch_list)))
                idx += self.batch_size
                self.batches.append(batch)
                if bar is not None: bar.update(1)
        if bar is not None: bar.close()
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.batches)

    def __iter__(self):
        self.shuffle()
        self.curr_batch_idx = 0
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        if self.curr_batch_idx >= len(self.batches):
            self.shuffle()
            self.curr_batch_idx = 0
            raise StopIteration()
        next_batch = self.batches[self.curr_batch_idx]
        self.curr_batch_idx += 1
        return next_batch

    def get_overlapping_data_count(self, loader):
        if not isinstance(loader, RNNShapeRegressDataLoader):
            print("Type mismatch, no overlaps by default")
            return 0
        n_unique_overlaps = 0
        my_data = set()
        for batch in self:
            batch_unique_str_id_set = batch[DK_BATCH_UNIQUE_STR_ID_SET]
            for str_id in batch_unique_str_id_set:
                my_data.add(str_id)
        for batch in loader:
            batch_unique_str_id_set = batch[DK_BATCH_UNIQUE_STR_ID_SET]
            for str_id in batch_unique_str_id_set:
                if str_id in my_data:
                    n_unique_overlaps += 1
        return n_unique_overlaps
