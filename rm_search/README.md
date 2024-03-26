## Random mutation search related files

Before starting any search you'll need to make sure:

* The OFA supernet checkpoint files are placed under `<your path to the code>/BlockProfile/.torch/ofa_nets/`.
* Latency predictors are pre-trained and the `.pt` checkpoint file placed under `<your path to the code>/BlockProfile/models/Latency/`.
* ImageNet validation data placed in some folder you know.

#### FAQ
* If you are getting an error like this when running a search: `ModuleNotFoundError: No module named 'search'`. 
Add `export PYTHONPATH=/your_path/to/BlockProfile/ && ` before the `CUDA_VISIBLE_DEVICES=...` command.

#### To train NPU/GPU/CPU latency predictors

Process the latency data for random networks into a pickle file. Make it a list of dicts. 
Each dict contains the net configs data, the end-to-end truth latency, and resolution. 
Then look at line `77-88` of `run_ofa_op_graph_lat_predictor.py` to see how each dict is processed. 
When ready, simply run `run_ofa_op_graph_lat_predictor.py` to train a latency predictor.

#### To run Pareto front search for the ProxylessNAS design space:

With insight, using **NPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_npu_insight_pareto -mutate_prob_type npu -space_id npu -resolution 224 -lat_predictor_checkpoint saved_models/ofa_pn_op_graph_npu_lat_predictor_best.pt

No insight, using **NPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_npu_full_space_pareto -resolution 224 -lat_predictor_checkpoint saved_models/ofa_pn_op_graph_npu_lat_predictor_best.pt

With insight, using **GPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_gpu_insight_pareto -mutate_prob_type gpu -space_id gpu -stage_block_count_type gpu -resolution 224 -lat_predictor_checkpoint saved_models/ofa_pn_op_graph_gpu_lat_predictor_best.pt

No insight, using **GPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_gpu_full_space_pareto -resolution 224 -lat_predictor_checkpoint saved_models/ofa_pn_op_graph_gpu_lat_predictor_best.pt

With insight, using **CPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_cpu_insight_pareto -mutate_prob_type cpu -space_id cpu -stage_block_count_type cpu -resolution 224 -lat_predictor_checkpoint saved_models/ofa_pn_op_graph_cpu_lat_predictor_best.pt

No insight, using **CPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_cpu_full_space_pareto -resolution 224 -lat_predictor_checkpoint saved_models/ofa_pn_op_graph_cpu_lat_predictor_best.pt

#### To run Pareto front search for the OFA(MobileNetV3) design space:

With insight, using **NPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_npu_insight_pareto -mutate_prob_type npu -space_id npu -resolution 224 -lat_predictor_checkpoint saved_models/ofa_mbv3_op_graph_npu_lat_predictor_best.pt

No insight, using **NPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_npu_full_space_pareto -resolution 224 -lat_predictor_checkpoint saved_models/ofa_mbv3_op_graph_npu_lat_predictor_best.pt

With insight, using **GPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_gpu_insight_pareto -mutate_prob_type gpu -space_id gpu -stage_block_count_type gpu -resolution 224 -lat_predictor_checkpoint saved_models/ofa_mbv3_op_graph_gpu_lat_predictor_best.pt

No insight, using **GPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_gpu_full_space_pareto -resolution 224 -lat_predictor_checkpoint saved_models/ofa_mbv3_op_graph_gpu_lat_predictor_best.pt

With insight, using **CPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_cpu_insight_pareto -mutate_prob_type cpu -space_id cpu -stage_block_count_type cpu -resolution 224 -lat_predictor_checkpoint saved_models/ofa_mbv3_op_graph_cpu_lat_predictor_best.pt

No insight, using **CPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_cpu_full_space_pareto -resolution 224 -lat_predictor_checkpoint saved_models/ofa_mbv3_op_graph_cpu_lat_predictor_best.pt

With insight, using **Note10** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_n10_insight_pareto -mutate_prob_type n10 -space_id n10 -stage_block_count_type n10 -resolution 224 -lat_predictor_type n10

No insight, using **Note10** latency predictor:

    CUDA_VISIBLE_DEVICES=3 python3 -u run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_n10_full_space_pareto -resolution 224 -lat_predictor_type n10

Once finished, the Pareto front architectures and their values will be printed in the console.

#### To run Max acc search for the ResNet50 design space:

With insight:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_resnet_rm_cons_acc_search.py -batch_size 200 -seed 1 -model_name ofa_resnet_rm_insight_max_acc -space_id max_acc -max_stage_block_count_type max_acc -min_stage_block_count_type max_acc -resolution 224

No insight:

    CUDA_VISIBLE_DEVICES=0 python3 -u run_ofa_resnet_rm_cons_acc_search.py -batch_size 200 -seed 1 -model_name ofa_resnet_rm_full_space_max_acc -resolution 224
    
Once finished, the top-10 architectures and their accuracy values will be printed in the console.
