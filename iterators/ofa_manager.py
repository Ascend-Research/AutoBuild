#import tensorflow  # Required or else ofa imports will cause some error
import torch as t
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.model_zoo import ofa_net
from ofa.tutorial import FLOPsTable
from abc import ABC, abstractmethod
from iterators.ofa_utils import ofa_str_configs_to_subnet_args
from iterators.latency_utils import measure_lat
import os
import pickle
from ge_utils.model import make_predictor
from ge_utils.torch_geo_data import _mobilenet_custom_as_torch_geo
from thop import profile
import time


PREDICTOR_MAP = {
    '2080Ti': 'gpu',
    '2990WX': 'cpu',
    'Kirin9000': 'npu'
}


class GeneralOFAManager(ABC):
    def __init__(self, metric="acc", inet_path="~/ImageNet", batch_size=100, n_workers=5, res=224):
        self.metric = metric
        if self.metric == "acc":
            ImagenetDataProvider.DEFAULT_PATH = inet_path
            self.run_config = ImagenetRunConfig(test_batch_size=batch_size, n_worker=n_workers)
            self.run_config.data_provider.assign_active_img_size(res)
        self.supernet = None
        self.res = res


    @abstractmethod
    def eval(self, cfg):
        # Obviously, this does not work and is just a placeholder.
        return self._evaluate_subnet(cfg)

    def _evaluate_acc(self, subnet):
        run_manager = RunManager(".torch/ofa_nets", subnet, self.run_config, init=False)
        run_manager.reset_running_statistics(net=subnet)

        loss, (top1, top5) = run_manager.validate(net=subnet)
        return top1
    
    def _evaluate_gpu(self, subnet):
        time.sleep(1)
        return measure_lat(subnet.cuda(), "gpu", h=self.res, w=self.res)
    
    def _evaluate_cpu(self, subnet):
        time.sleep(1)
        return measure_lat(subnet.cpu(), "cpu", h=self.res, w=self.res)
    
    def load_predictors(self, family, metrics=['2080Ti', '2990WX', 'Kirin9000']):
        predictors = {}
        for metric in metrics:
            cfg_fname = f"l1_models/minMAE_ofa_{family}_latency_{PREDICTOR_MAP[metric]}_custom_latency_{PREDICTOR_MAP[metric]}_GraphConv_sum_directed_srcc1_config.pkl"
            if os.path.exists(cfg_fname):
                with open(cfg_fname, "rb") as f:
                    config = pickle.load(f)
                predictor_fname = f"l1_models/minMAE_ofa_{family}_latency_{PREDICTOR_MAP[metric]}_custom_latency_{PREDICTOR_MAP[metric]}_GraphConv_sum_directed_srcc1_predictor.pt"
                predictor = make_predictor(gnn_chkpt=predictor_fname, **config)
                predictor.eval()
                predictors[f'{metric}_predicted'] = predictor
        self.predictors = predictors

    def execute_aux_predictors(self, config):
        convert_func = _mobilenet_custom_as_torch_geo
        results = {}
        for p_label, predictor in self.predictors.items():
            geo_data = convert_func(config)
            results[p_label] = predictor.forward_denorm(geo_data)[0].cpu().item()
        return results

    
class MBv3Manager(GeneralOFAManager):
    def __init__(self, metric="acc", supernet=True):
        super().__init__(metric=metric)
        if supernet:
            self.supernet = ofa_net("ofa_mbv3_d234_e346_k357_w1.2", pretrained=True)
            self.flops_p = FLOPsTable(device="cuda:0", batch_size=1)
        self.load_predictors("mbv3")

    def eval(self, cfg):
        ks, e, d = ofa_str_configs_to_subnet_args(cfg, max_n_net_blocks=20, expected_prefix="mbconv3")
        self.supernet.set_active_subnet(ks=ks, e=e, d=d)
        subnet = self.supernet.get_active_subnet(preserve_weight=True)
        if self.metric.endswith("_predicted"):
            results = {}
        else:
            results = {self.metric: eval(f"self._evaluate_{self.metric}(subnet)")}
        arch_dict = {'ks': ks, 'e': e, 'd': d}
        results['flops'] = self.flops_p.predict_efficiency(arch_dict)
        results = {**results, **self.execute_aux_predictors(cfg)} 
        return results
    

class PNManager(GeneralOFAManager):
    def __init__(self, metric="acc", supernet=True):
        super().__init__(metric=metric)
        if supernet:
            self.supernet = ofa_net("ofa_proxyless_d234_e346_k357_w1.3", pretrained=True)
        self.load_predictors("pn")
        self.input_tensor = t.randn(1, 3, self.res, self.res).cuda()

    def eval(self, cfg):
        ks, e, d = ofa_str_configs_to_subnet_args(cfg, max_n_net_blocks=21, expected_prefix="mbconv2")
        self.supernet.set_active_subnet(ks=ks, e=e, d=d)
        subnet = self.supernet.get_active_subnet(preserve_weight=True)
        if self.metric.endswith("_predicted"):
            results = {}
        else:
            results = {self.metric: eval(f"self._evaluate_{self.metric}(subnet)")}
        results['flops'] = profile(subnet.cuda(), inputs=(self.input_tensor.cuda(),), verbose=False)[0] / 500000
        results = {**results, **self.execute_aux_predictors(cfg)} 
        return results
