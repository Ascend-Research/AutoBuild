# Building Optimal Neural Architectures using Interpretable Knowledge
# Qua<sup>2</sup>SeDiMo Branch

<p align="center">
    <a href="https://aaai.org/Conferences/AAAI-23/" alt="Conference">
        <img src="https://img.shields.io/badge/AAAI'25-blue" /></a>
    <a href="https://github.com/Ascend-Research/AIO-P/blob/master/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-MIT-purple" /></a>
    <a href="https://www.python.org/" alt="Python">
        <img src="https://img.shields.io/badge/Python-3.10-yellow" /></a>
    <a href="https://pytorch.org/" alt="PyTorch">
        <img src="https://img.shields.io/badge/PyTorch-2.1-orange" /></a>
<p/>

Repository for the Qua<sup>2</sup>SeDiMo [AAAI-25] code branch of AutoBuild [CVPR'24]. This repository branch deals with training interpretable predictors and examining the quantifiable sensitivity insights. It relies on datasets of sampled quantization configurations from the main code repository. It then builds optimal quantization configurations using an interpretable GNN predictor.

**Setting up the environment**
This code uses a different virtual environment than the main code repo, primarily that it does not rely on Diffusers or any other packages related to images, but *does* rely on [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and  [torchsort](https://github.com/teddykoker/torchsort).

```
conda create -n "autobuild" python=3.10
conda activate autobuild
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torchsort-0.1.9+pt21cu118-cp310-cp310-linux_x86_64.whl  (Downloaded from https://github.com/teddykoker/torchsort/releases/tag/v0.1.9)
conda install pyg -c pyg
conda install -c conda-forge onnx
pip install numpy==1.26.4 pandas
```

You will also need to install [pytorchltr](https://github.com/rjagerman/pytorchltr). Trying to just use `pip` caused issues in our case. Instead, do the following:
```
pip install cython
git clone https://github.com/rjagerman/pytorchltr.git
cd pytorchltr
python setup.py install
```

## Setup Datasets
Make two new directories: `/cache/`. `/cache/` should be populated using content from the [Google Drive](https://drive.google.com/drive/folders/19gTl00BfDaQSQMlOC_aM5MMPo6grHC_1?usp=sharing).

## Running the predictor
We consider three hop-level losses: SRCC, LambdaRank (NDCG) and a hybrid loss that combines them both. Corresponding with this, we have three top-level predictor files: `train_predictor_fold_{srcc, ndcg, hybrid}.py`. Example usage:
```
python train_predictor_fold_ndcg.py -families alpha -tag "alpha_ndcg_code_submission" -target "-1*FID-GT-1k - avg_bits*150" > example_alpha_fid_bits.txt;
```

**Flags**
* `-families` is the denoiser neural network. Choose from {alpha, sigma, hunyuan, sdxl, sdv15, dit}
* `-target` is the target equation to optimize, e.g., `"-1*FID-GT-1k - avg_bits*150"` in the paper. Default is `"FID-GT-1k*-1"`. Should be an arithmetric equation involving dataset keys from `/cache/`, namely `FID-GT-1k`, `avg_bits` and `bops`. Please see `/ge_utils/label_eq.py` for more details on how `-target` syntax is processed.


This script uses joblib to the `K=5` fold splits simultaneously. We strongly recommend piping output to a text file as shown in the example. The script will report SRCC and/or NDCG@10 for each fold. When each fold is complete, it will then call `label_units_sdm_fold.py` twice to perform the subgraph labeling, once for the node-level optimization, and again for the subgraph-level optimization. These are the subgraph scores, stored in `units/`. We provide several examples for the node and subgraph optimization already (subgraph-level optimization files are large, so there's only a few). You can use ipython and pickle to open the file and enumerate the subgraphs to look at how they are setup and their scores. 

Additionally, `label_units_sdm_fold.py` calls `convert_sdm_unit_to_scheme` to generate the optimal quantization configuration and store it in `quant_configs/`. We provide some sample quantization configurations from the paper and they can be paired with the forked `q-diffusion` code to evaluate (requires generating a 'supernetwork' first due to size limit; see `README.md` in the `q-diffusion` fork).

## Analyzing Quantization Sensitivity Insights and Generating Plots
*Coming Soon...*