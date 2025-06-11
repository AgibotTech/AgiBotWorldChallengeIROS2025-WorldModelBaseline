#!/usr/local/bash

input_root=PATH_TO_LOAD_INPUTS
save_root=PATH_TO_SAVE_IMAGES
ckp_path=PATH_TO_CHECKPOINT
config_path=PATH_TO_MODEL_CONFIG

python evac/main/infer_all.py -i $input_root -s $save_root --ckp_path $ckp_path --config_path $config_path
