#!/usr/local/bash

input_root=PATH_TO_YOUR_DATASET
save_root=PATH_TO_SAVE_IMAGES
ckp_path=PATH_TO_CHECKPOINT
config_path=PATH_TO_MODEL_CONFIG
n_pred=3
python evac/main/infer_all.py -i $input_root -s $save_root --ckp_path $ckp_path --config_path $config_path --n_pred $n_pre
