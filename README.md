# DRCP
Basically everything of DRCP for cooperative perception.

This repo is a realization of DRCP paper on DAIR-V2X dataset and OPV2V dataset, which refined and improved **efficient cross-modal fusion** between **LiDAR and Camera** sensory data for **cooperative perception** tasks based on RG-Attn [GitHub](https:github.com/LantaoLi/RG-Attn/tree/main). 

DRCP is built on opencood and HEAL, most of the conda environment and dataset configurations are the same except introducing diffusers module requiring Python 3.8 environment (as diffuser module is not available for Python 3.7).

## Data Preparation
- DAIR-V2X-C: Download the data from [this page](https://thudair.baai.ac.cn/index). We use complemented annotation, so please also follow the instruction of [this page](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/).
- [Optional] OPV2V: Please refer to [this repo](https://github.com/DerrickXuNu/OpenCOOD).

Create a `dataset` folder under any folder path you like and put your data there. Make the naming and structure consistent with the following and change the dataset paths accordingly in the config.yaml (the first few lines) for training or testing purpose.
```
/any_path_U_like

.
├── dair_v2x
│   ├── v2x_c
│   ├── v2x_i
│   └── v2x_v
├── OPV2V [Optional]
│   ├── additional
│   ├── test
│   ├── train
│   └── validate
```


## Installation
### Step 1: Conda Env
If opencood or HEAL conda environment is already setup on your machine, then trying those environments with our project is totally fine. There might be some modules missing or conflicting, but not going to be too messy. Basically, following [HEAL](https://github.com/yifanlu0227/HEAL) to configure the environment is enough for running this project.

```bash
conda create -n drcp python=3.8
#python 3.7 might not support diffusers for diffusion process
conda activate drcp

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cudatoolkit=11.7 -c pytorch -c nvidia

pip install -r requirements.txt

python setup.py develop
```

### Step 2: Spconv (1.2.1 or 2.x)
For generating voxel features:

To install spconv 1.2.1, please follow the guide in https://github.com/traveller59/spconv/tree/v1.2.1.

To install spconv 2.x, please run the following commands (if you are using cuda 11.3):
```python
pip install spconv-cu113
```
#### Tips for installing spconv 1.2.1:
1. make sure your cmake version >= 3.13.2
2. CUDNN and CUDA runtime library (use `nvcc --version` to check) needs to be installed on your machine.

**Note that** spconv 2.x are much easier to install, but our experiments and checkpoints follow spconv 1.2.1. If you do not mind training from scratch, spconv 2.x is recommended.

### Step 3: Bbx IoU cuda version compile
Install bbx nms calculation cuda version

```bash
python opencood/utils/setup.py build_ext --inplace
```

### Step 4: Install pypcd by hand for DAIR-V2X LiDAR loader.

``` bash
pip install git+https://github.com/klintan/pypcd.git
#if RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED, change the numpy_pc2.py line 301 dtype=float
```

---
### Train by yourself
```python
python opencood/tools/train.py --model_dir ${CHECKPOINT_FOLDER} -y config.yaml
```

The corresponding CHECKPOINT_FOLDER is already configured as /RGAttn_root/opencood/logs/CP_Dair_Final/Dair_Clean_PTP and /RGAttn_root/opencood/logs/CP_Dair_Final/Dair_Clean_CoSCoCo.

### Test
```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER}
```

## Benchmark Checkpoints
We also provide checkpoint files at [RG-Attn's Huggingface Hub](https://huggingface.co/LLT007/RG-Attn/tree/main).
Please note that 21.pth is for PTP and 23.pth is for CoS-CoCo, put them in their corresponding folders for direct evaluations.

## Thanks
We appreciate the great efforts and foundation works from UCLA, SJTU, Tsinghua and all other research facilities on cooperative perception.

## Citation
```
@article{li2025rg,
  title={RG-Attn: Radian Glue Attention for Multi-modality Multi-agent Cooperative Perception},
  author={Li, Lantao and Yang, Kang and Zhang, Wenqi and Wang, Xiaoxue and Sun, Chen},
  journal={arXiv preprint arXiv:2501.16803},
  year={2025}
}
```
