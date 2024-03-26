import torch
from tqdm import tqdm
from thop import profile
from search.rm_search.ofa_profile.networks import OFAMbv3Net
from search.rm_search.utils.misc_utils import RunningStatMeter
from search.rm_search.utils.model_utils import measure_gpu_latency, measure_cpu_latency


class InputShapeHook:

    def __init__(self, module:torch.nn.Module):
        self.input_shape = None
        self.handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input_tsr, output_tsr):
        if isinstance(input_tsr, tuple):
            self.input_shape = tuple(input_tsr[0].shape)
        else:
            self.input_shape = tuple(input_tsr.shape)

    def close(self):
        self.handle.remove()


class InputOutputShapeHook:

    def __init__(self, module:torch.nn.Module):
        self.input_shape = None
        self.output_shape = None
        self.handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input_tsr, output_tsr):
        if isinstance(input_tsr, tuple):
            self.input_shape = tuple(input_tsr[0].shape)
        else:
            self.input_shape = tuple(input_tsr.shape)
        if isinstance(output_tsr, tuple):
            self.output_shape = tuple(output_tsr[0].shape)
        else:
            self.output_shape = tuple(output_tsr.shape)

    def close(self):
        self.handle.remove()


def _add_hooks(net:OFAMbv3Net):
    hooks = []
    for bi, block in enumerate(net.blocks):
        if bi == 0: continue # Ignore first fixed block
        hook = InputShapeHook(block)
        hooks.append(hook)
    return hooks


def _close_hooks(hooks):
    for hook in hooks:
        hook.close()


def measure_lats(nets, start_idx=0, end_idx=5000,
                 net_func=OFAMbv3Net, dev="cuda",
                 unit="ms", C=3, H=224, W=224):
    assert H == W
    if dev == "cuda":
        measure_func = measure_gpu_latency
    else:
        measure_func = measure_cpu_latency
    data = []
    lat_meter = RunningStatMeter()
    summed_lat_meter = RunningStatMeter()
    n_weight_meter = RunningStatMeter()
    net_batch = torch.ones([1, C, H, W]).float().to(torch.device(dev))
    bar = tqdm(total=end_idx - start_idx, desc="Measuring lat, device={}".format(dev), ascii=True)
    for net_id, configs in enumerate(nets):
        if net_id < start_idx or net_id >= end_idx:
            continue
        block_stage_inds = []
        for si, stage_b in enumerate(configs):
            for _ in stage_b:
                block_stage_inds.append(si)
        # End-to-end lat
        net = net_func(configs).to(torch.device(dev))
        n_net_weights = sum(p.numel() for p in net.parameters() if p.requires_grad)
        with torch.no_grad():
            net.eval()
            macs, _ = profile(net, inputs=(net_batch,))
            net_flops = 2 * macs
            net_avg, net_var = measure_func(lambda: net(net_batch))
        lat_meter.update(net_avg)
        n_weight_meter.update(n_net_weights)
        # Per-layer lat
        hooks = _add_hooks(net)
        with torch.no_grad():
            net.eval()
            net(net_batch)
        _close_hooks(hooks)
        block_input_shapes = [hook.input_shape for hook in hooks]
        per_block_lats = []
        per_block_weights = []
        per_block_flops = []
        net = net_func(configs).to(torch.device(dev))
        with torch.no_grad():
            net.eval()
            for bi, shape in enumerate(block_input_shapes):
                b_batch = torch.ones(shape).float().to(torch.device(dev))
                block = net.blocks[bi + 1] # Ignore first fixed block
                n_block_weights = sum(p.numel() for p in block.parameters() if p.requires_grad)
                b_macs, _ = profile(block, inputs=(b_batch,))
                b_flops = b_macs * 2
                block_avg, _ = measure_func(lambda: block(b_batch))
                per_block_lats.append(block_avg)
                per_block_weights.append(n_block_weights)
                per_block_flops.append(b_flops)
        assert len(per_block_lats) == len(block_stage_inds)
        data.append({"net": configs,
                     "net_id": net_id,
                     "device": dev,
                     "unit": unit,
                     "overall {} lat".format(dev): net_avg,
                     "overall {} lat variance".format(dev): net_var,
                     "num trainable weights": n_net_weights,
                     "flops": net_flops,
                     "per block {} lats".format(dev): per_block_lats,
                     "block input shapes": block_input_shapes,
                     "block stage inds": block_stage_inds,
                     "input HWC": (H, W, C),
                     "resolution": H,
                     "per block weights": per_block_weights,
                     "per block flops": per_block_flops})
        sum_lat = sum(per_block_lats)
        summed_lat_meter.update(sum_lat)
        bar.desc = "Device={}, lat avg: {:.2f}, max: {:.2f}, min: {:.2f}, " \
                   "summed avg: {:.2f}, weights: {:.1f}".format(dev, lat_meter.avg, lat_meter.max, lat_meter.min,
                                                                summed_lat_meter.avg, n_weight_meter.avg)
        bar.update(1)
    bar.close()
    return data
