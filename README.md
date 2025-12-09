# DRCP
Basically everything of DRCP for cooperative perception.

This repo is a realization of DRCP paper on DAIR-V2X dataset and OPV2V dataset, which refined and improved **efficient cross-modal fusion** between **LiDAR and Camera** sensory data for **cooperative perception** tasks based on [RG-Attn](https://github.com/LantaoLi/RG-Attn). 

DRCP is built upon **OpenCood** and **HEAL**, while adopting a diffusion-enhanced architecture. Compared to HEAL-style environments, DRCP requires **Python ≥ 3.8**, as the diffusers module does not support Python 3.7.

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
Since Python 3.7 is not suitable for running drcp, please follow the following commands strictly to configure the environment, or upgrade existing conda environment (from HEAL project or similar existing opencood environments configurations) to **Python 3.8** with `diffusers==0.30.3` module installed and recommended PyTorch stack as `pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cudatoolkit=11.8`.

```bash
conda create -n drcp python=3.8
#python 3.7 might not support diffusers for diffusion process
conda activate drcp

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cudatoolkit=11.8 -c pytorch -c nvidia

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
#you might need to change the numpy_pc2.py line 301 from dtype=np.float to dtype=np.float32
```

---
## Training
### Stage 1 (Non-diffusion model training)
To train a non-diffusion model (non_dif or prgaf), please use following command:
```python
python opencood/tools/train.py --model_dir ${CHECKPOINT_FOLDER} -y config.yaml
```

The corresponding CHECKPOINT_FOLDER are already configured as `/DRCP_root/opencood/logs/dairv2x/non_dif`,  `/DRCP_root/opencood/logs/dairv2x/prgaf` and `/DRCP_root/opencood/logs/opv2v/nodif`.

### Stage 1 (Train with diffusion)
After the non-diffusion model trained, move the best model checkpoints to the diffusion training directories such as `/DRCP_root/opencood/logs/dairv2x/dif` or `/DRCP_root/opencood/logs/opv2v/dif`. The second stage training for diffusion-based models can then be continued as:

```python
# For dair-v2x
python opencood/tools/diffusion_partial_train.py --model_dir ${CHECKPOINT_FOLDER} -y config_dif2e-4.yaml
# For opv2v
python opencood/tools/diffusion_partial_train.py --model_dir ${CHECKPOINT_FOLDER} -y dif_config.yaml
```

## Testing
For non-diffusion model
```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER}
```

For diffusion-enabled model
```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --diffuse True -y <Diffusion-enabled .yaml files>
```

## Benchmark Checkpoints
We also provide checkpoint files at [DRCP's Huggingface Hub](https://huggingface.co/LLT007/DRCP).

## Thanks
We appreciate the great efforts and foundation works from UCLA, SJTU, Tsinghua, TUM and all other research facilities on cooperative perception.

## Citation
```
@article{li2025drcp,
  title={DRCP: Diffusion on Reinforced Cooperative Perception for Perceiving Beyond Limits},
  author={Li, Lantao and Yang, Kang and Song, Rui and Sun, Chen},
  journal={arXiv preprint arXiv:2509.24903},
  year={2025}
}
```
