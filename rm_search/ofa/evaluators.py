from search.rm_search.utils.model_utils import device
from search.rm_search.ofa.model_zoo import ofa_net
from search.rm_search.model_helpers import ArchEvaluator
from search.rm_search.ofa.my_utils import ofa_str_configs_to_subnet_args
from search.rm_search.model_helpers import get_simple_cached_class_loader
from search.rm_search.ofa.imagenet_classification.run_manager import RunManager
from search.rm_search.ofa.imagenet_classification.run_manager import ImagenetRunConfig
from search.rm_search.ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider


"""
Different types of evaluators for the OFA
"""


class OFANetAccLatEvaluator(ArchEvaluator):

    def __init__(self, data_path, runs_dir,
                 net_name, model_dir,
                 max_n_net_blocks, resolution,
                 block_prefix=None,
                 ext_lat_predictor=None, n_worker=32,
                 dummy_lat=1.0, valid_size=None, batch_size=128,
                 max_n_blocks_per_stage=4, is_test=False,
                 use_cached_loader=False,
                 metric="2080Ti_predicted"):
        super(OFANetAccLatEvaluator, self).__init__()
        self.max_n_net_blocks = max_n_net_blocks
        self.max_n_blocks_per_stage = max_n_blocks_per_stage
        self.runs_dir = runs_dir
        self.resolution = resolution
        self.dummy_lat = dummy_lat
        self.ext_lat_predictor = ext_lat_predictor
        self.block_prefix = block_prefix
        ImagenetDataProvider.DEFAULT_PATH = data_path
        self.network = ofa_net(net_id=net_name, pretrained=True, model_dir=model_dir)
        self.run_config = ImagenetRunConfig(test_batch_size=batch_size, n_worker=n_worker,
                                            valid_size=valid_size)
        self.is_test = is_test
        self.use_cached_loader = use_cached_loader
        self.network.sample_active_subnet()
        subnet = self.network.get_active_subnet(preserve_weight=True).to(device())
        self.run_manager = RunManager(self.runs_dir, subnet, self.run_config,
                                      init=False, verbose=False)
        self.run_config.data_provider.assign_active_img_size(self.resolution)
        self.cached_loader = None
        if self.use_cached_loader:
            print("Using cached loader!")
            ext_loader = self.run_manager.run_config.test_loader if is_test \
                else self.run_manager.run_config.valid_loader
            self.cached_loader = get_simple_cached_class_loader(ext_loader, batch_size)
        self.metric = metric

    def get_perf_values(self, net_configs):
        if str(net_configs) in self.perf_memo:
            return self.perf_memo[str(net_configs)]
        else:
            k_list, e_list, d_list = \
                ofa_str_configs_to_subnet_args(net_configs, self.max_n_net_blocks,
                                               max_n_blocks_per_stage=self.max_n_blocks_per_stage,
                                               expected_prefix=self.block_prefix)
            self.network.set_active_subnet(wid=None, ks=k_list, e=e_list, d=d_list)
            subnet = self.network.get_active_subnet(preserve_weight=True)
            subnet = subnet.to(device())
            self.run_manager.reset_running_statistics(net=subnet)
            _, (top1, _) = self.run_manager.validate(net=subnet,
                                                     data_loader=self.cached_loader,
                                                     is_test=self.is_test,
                                                     no_logs=True, verbose=False)
            self.num_eval += 1
            if self.ext_lat_predictor is not None:
                if self.metric.endswith("_predicted"):
                    lat = self.ext_lat_predictor.execute_aux_predictors(net_configs)[self.metric]
                else:
                    lat = eval(f"self.ext_lat_predictor._evaluate_{self.metric}(subnet)")
                #lat = self.ext_lat_predictor.eval(net_configs)[self.metric]
            else:
                lat = self.dummy_lat
            if self.enable_perf_memo:
                self.perf_memo[str(net_configs)] = (top1, lat)
            return top1, lat


class OFAResNetAccLatEvaluator(ArchEvaluator):

    def __init__(self, data_path, runs_dir,
                 net_name, model_dir,
                 resolution,
                 ext_lat_predictor=None, n_worker=32,
                 dummy_lat=1.0, valid_size=None, batch_size=128,
                 is_test=False,
                 use_cached_loader=False,
                 metric="2080Ti_predicted"):
        super(OFAResNetAccLatEvaluator, self).__init__()
        self.runs_dir = runs_dir
        self.resolution = resolution
        self.dummy_lat = dummy_lat
        self.ext_lat_predictor = ext_lat_predictor
        ImagenetDataProvider.DEFAULT_PATH = data_path
        self.network = ofa_net(net_id=net_name, pretrained=True, model_dir=model_dir)
        self.run_config = ImagenetRunConfig(test_batch_size=batch_size, n_worker=n_worker,
                                            valid_size=valid_size)
        self.is_test = is_test
        self.use_cached_loader = use_cached_loader
        self.cached_loader = None
        self.network.sample_active_subnet()
        subnet = self.network.get_active_subnet(preserve_weight=True).to(device())
        self.run_manager = RunManager(self.runs_dir, subnet, self.run_config,
                                      init=False, verbose=False)
        self.run_config.data_provider.assign_active_img_size(self.resolution)
        if self.use_cached_loader:
            print("Using cached loader!")
            ext_loader = self.run_manager.run_config.test_loader if is_test \
                else self.run_manager.run_config.valid_loader
            self.cached_loader = get_simple_cached_class_loader(ext_loader, batch_size)
        self.metric = metric

    def get_perf_values(self, net_configs):
        if str(net_configs) in self.perf_memo:
            return self.perf_memo[str(net_configs)]
        else:
            if type(net_configs) == dict:
                d_list, e_list, w_list = net_configs['d'], net_configs['e'], net_configs['w']
            else:
                d_list = net_configs.d_list
                e_list = net_configs.e_list
                w_list = net_configs.w_list
            self.network.set_active_subnet(d=d_list, e=e_list, w=w_list)
            subnet = self.network.get_active_subnet(preserve_weight=True)
            subnet = subnet.to(device())
            self.run_manager.reset_running_statistics(net=subnet)
            _, (top1, _) = self.run_manager.validate(net=subnet,
                                                     data_loader=self.cached_loader,
                                                     is_test=self.is_test,
                                                     no_logs=True, verbose=False)
            self.num_eval += 1
            if self.ext_lat_predictor is not None:
                if self.metric.endswith("_predicted"):
                    if type(net_configs) == dict:
                        lat = self.ext_lat_predictor.execute_aux_predictors(net_configs)[self.metric]
                    else:
                        net_config_as_dict = {'d': net_configs.d_list,
                                            'e': net_configs.e_list,
                                            'w': net_configs.w_list}
                        lat = self.ext_lat_predictor.execute_aux_predictors(net_config_as_dict)[self.metric]
                else:
                    lat = eval(f"self.ext_lat_predictor._evaluate_{self.metric}(subnet)")
            else:
                lat = self.dummy_lat
            if self.enable_perf_memo:
                self.perf_memo[str(net_configs)] = (top1, lat)
            return top1, lat
