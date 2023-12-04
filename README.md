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

## Prepare Datasets
1. (option)
    * miniImagenet: Use [this](https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view) link to download the dataset. Extract the .tar.gz file in the `dataset` directory.
    * tieredImagenet: Use [this](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view) link to download the dataset. Extract the .tar.gz file in the `dataset` directory.
    * cifarfs: Use [this](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u) link to download the dataset. Extract the .tar file in the `dataset` directory.
    * fc100: Use[this](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u) link to download the dataset. Extract the .tar file in the `dataset` directory.
    

2. (option) 
    * you can download the entire datasets this [Baidu Drive(c6rc)](https://pan.baidu.com/s/1V9mCYzCvNjM_7FVPBX1tcQ?pwd=c6rc) link to download the dataset. Extract the .tar.gz file in the `dataset` directory.
    
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
- -**dataset**:store datasets 
- -**logs**:Logs of generated model training and testing
- -**model**:the model selected after training
- -**src**:Source code files
- -**analyze.ipynb**:Analyze and process logs
