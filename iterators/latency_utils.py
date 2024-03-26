import numpy as np
import torch


def measure_lat(net, device, h=224, w=224, c=3):
    mock_tensor = torch.rand(1, c, h, w)
    def inf_func():
        net(mock_tensor)
    
    if device == "gpu":
        mock_tensor = mock_tensor.cuda()
        return measure_gpu_latency(inf_func)[0]
    elif device == "cpu":
        return measure_cpu_latency(inf_func)[0]
    else:
        raise NotImplementedError



def measure_gpu_latency(inf_func, m_ignore_runs=10, n_reps=100):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(m_ignore_runs):
        inf_func()
    with torch.no_grad():
        for rep in range(n_reps):
            starter.record()
            inf_func()
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # in milliseconds
            timings.append(curr_time)
    return np.mean(timings), np.var(timings)


def measure_cpu_latency(inf_func, m_ignore_runs=10, n_reps=20):
    import time
    timings = []
    for _ in range(m_ignore_runs):
        inf_func()
    with torch.no_grad():
        for rep in range(n_reps):
            start = time.time()
            inf_func()
            curr_time = time.time() - start # In seconds
            timings.append(curr_time * 1000)  # Now in milliseconds
    return np.mean(timings), np.var(timings)
