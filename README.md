# Transductive Semantic Decoupling Double Variational Inference for Few-shot Classification
 
## Install the Requirement
```bash
###################################
###  Step by Step Installation   ##
###################################

# 1. create and activate conda environment
conda env create -f tsdvi.yaml
# if don't run
pip install -r tsdvi.txt

# 2. download this project
git clone https://github.com/zjh1015/tsdvi
```

## Prepare GLT Datasets
We propose two datasets for the Generalized Long-Tailed (GLT) classification tasks: ImageNet-GLT and MSCOCO-GLT. 
- For **ImageNet-GLT** [(link)](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch/tree/main/_ImageNetGeneration), like most of the other datasets, we don't have attribute annotations, so we use feature clusters within each class to represent K ''pretext attributes''. In other words, each cluster represents a meta attribute layout for this class.
- For **MSCOCO-GLT** [(link)](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch/tree/main/_COCOGeneration), we directly adopt attribute annotations from [MSCOCO-Attribute](https://github.com/genp/cocottributes) to construct our dataset.

Please follow the above links to prepare the datasets.

## Conduct Training and Testing

### Train Models
Run the following command to train a TRAIN_TYPE model:
```
python -m src.tsdvi_train --cnfg CONFIGS_PATH
#example
python -m src.tsdvi_train --cnfg configs/fc100-5,1/train_conf.json

```

### Test Models
Run the following command to test a baseline model:
```
python -m src.tsdvi_test --cnfg CONFIGS_PATH
#example
python -m src.tsdvi_test --cnfg configs/fc100-5,1/test_conf.json
```

## Program Description
- -**configs**:Configuration information for datasets running on the model
- -**data**:Initial processing of the dataset
- -**logs**:Logs of generated model training and testing
- -**model**:the model selected after training
- -**src**:Source code files
- -**analyze.ipynb**:Analyze and process logs
