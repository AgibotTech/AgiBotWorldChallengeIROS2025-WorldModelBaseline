# World Model Baseline

We adopt [EVAC](https://github.com/AgibotTech/EnerVerse-AC) as the baseline model for the [AgiBot World Challenge @ IROS 2025](https://agibot-world.com/challenge) - World Model track.

This repo provides a minial version of training codes. 

## News

- [2025.06.12] The instruction to evaluating your model locally have been released.

- [2025.05.26] 🚀🚀 The minimal version of training code for [AgiBot World Challenge @ IROS 2025](https://agibot-world.com/challenge) - World Model track have been released.

- [2025.05.26] 🔥🔥 The training and validation datasets of [AgiBot World Challenge @ IROS 2025 - World Model track](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/WorldModel) have been released.

- [2025.05.16] The minimal version of training code for AgibotWorld dataset and pretrained weights have been released.

## Getting started

### Setup
```
git clone https://github.com/AgibotTech/AgiBotWorldChallengeIROS2025-WorldModelBaseline.git
conda create -n enerverse python=3.10.4
conda activate enerverse

pip install -r requirements.txt

### install pytorch3d following https://github.com/facebookresearch/pytorch3d
### note that although the CUDA version is 11.8, we use the pytorch3d prebuilt on CUDA 12.1
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

```

### Inference
We have released the [test set](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/blob/main/WorldModel/test.tar.gz) for the competition, which adheres to the data organization requirement of [EWMBench](https://github.com/AgibotTech/EWMBench). To facilitate participants in conducting local evaluations using [EWMBench](https://github.com/AgibotTech/EWMBench) on the validation set, we also provide [reorganized validation set](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/blob/main/WorldModel/val_reorganized.tar.gz) along with the corresponding inference and evaluation scripts.

1. Download the reorganized validation set or the test dataset, or reorganize you custom dataset to the the required directory structure outlined below.
```
PATH_TO_YOUR_DATASET/
├── task_0/
│   ├── episode_0/
│   │   ├── frame.png
│   │   ├── head_intrinsic_params.json
│   │   ├── head_extrinsic_params_aligned.json
│   │   └── proprio_stats.h5
│   ├── episode_2/
│   └── ...
├── task_1/
└── ...
```
2. If you are using EVAC as the baseline model, make sure the submodule [evac](https://huggingface.co/agibot-world/EnerVerse-AC) is the latest version
```
git submodule update --remote
```
3. Modify the path variables in scripts/infer.sh
4. Run the script, which will predict ${n_pred} different generations for each input episode
```bash scripts/infer.sh```
5. The output directory will contain the following:
```
ACWM_dataset/
├── task_0/
│   ├── episode_0/
│   |   ├── 0/
|   |   |   └── video/
|   |   |       ├── frame_00000.png
|   |   |       ├── frame_00001.png
|   |   |       ├── ...
|   |   |       └── frame_*.png
│   |   ├── 1/
|   |   |   └── video/
|   |   |       ├── frame_00000.png
|   |   |       ├── frame_00001.png
|   |   |       ├── ...
|   |   |       └── frame_*.png
│   |   └── 2/
|   |       └── video/
|   |           ├── frame_00000.png
|   |           ├── frame_00001.png
|   |           ├── ...
|   |           └── frame_*.png
│   ├── episode_1/
│   └── ...
├── task_1/
└── ...
```

### Online Evaluation on test dataset

TBD


### Local Evaluation on validation dataset

1. Clone [EWMBench](https://github.com/AgibotTech/EWMBench.git) and setup the environment following the instruction in [EWMBench](https://github.com/AgibotTech/EWMBench)
2. Download the reorganized validation set or reorganize you custom dataset to the the required directory structure outlined below.

```
DIRPATH_TO_YOUR_DATASET/
  gt_dataset/
  ├── task_1/
  │   ├── episode_1/
  │   │   └── video/
  │   │       ├── frame_00000.png
  │   │       ├── ...
  │   │       └── frame_0000n.png
  │   ├── episode_2/
  │   └── ...
  ├── task_2/
  └── ...
```
3. Modify the path in PATH_TO_EWMBench/config.yaml
```
model_name: ACWM
data:
  gt_path: DIRPATH_TO_YOUR_DATASET/gt_dataset
  val_base: DIRPATH_SAVE_PREDICTION/ACWM_dataset
...
```

Note that the base name of ``gt_path`` should be ``gt_dataset`` and the base name of ``val_base`` should be ``${model_name}_dataset``

4. Run the scripts
```
# Preprocess input images and detect grippers
bash processing.sh ./config.yaml

# Calculate metrics
python evaluate.py --dimension 'semantics' 'trajectory_consistency' 'diversity' 'scene_consistency' 'psnr' 'ssim' --config ./config.yaml
```

We only use three metrics for online evaluation: **PSNR**, **scene_consistency** and **nDTW**. The evaluatoin results of EVAC on the validation dataset are tabulated bellow. More detailed results can be found in https://huggingface.co/agibot-world/EnerVerse-AC/blob/main/EVAC_validation_set_metrics.csv .

| PSNR   | Scene Consistency |  nDTW  |
|:------:|:-----------------:|:------:|
|20.9841 |     0.9013        | 0.9065 |




### Train

#### Training on [AgiBot World Challenge @ IROS 2025](https://agibot-world.com/challenge)

1. Download [🤗AgiBot World Challenge @ IROS 2025 - World Model track](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/WorldModel) dataset.

2. Download the checkpoint from [EVAC](https://huggingface.co/agibot-world/EnerVerse-AC), and modify ``model.pretrained_checkpoint`` in ``configs/agibotworld/train_config_iros_challenge_wm.yaml`` to the checkpoint file ``*.pt``

3. Download the weight of [CLIP](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K), and modify ``model.params.img_cond_stage_config.params.abspath``
in ``configs/agibotworld/train_config_iros_challenge_wm.yaml`` to the absolute path to ``open_clip_pytorch_model.bin`` inside the download directory

4. Modify the path ``data.params.train.params.data_roots`` in ``configs/agibotworld/train_config_iros_challenge_wm.yaml`` to the root of AgiBotWorld dataset

5. Run the script
```
bash scripts/train.sh configs/agibotworld/train_config_iros_challenge_wm.yaml
```

#### Training on [AgiBotWolrd](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)

1. Download [🤗AgiBotWolrd](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) dataset.

2. Download the checkpoint from [EVAC](https://huggingface.co/agibot-world/EnerVerse-AC), and modify ``model.pretrained_checkpoint`` in ``configs/agibotworld/train_configs.yaml`` to the checkpoint file ``*.pt``

3. Download the weight of [CLIP](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K), and modify ``model.params.img_cond_stage_config.params.abspath``
in ``configs/agibotworld/train_configs.yaml`` to the absolute path to ``open_clip_pytorch_model.bin`` inside the download directory

4. Modify the path ``data.params.train.params.data_roots`` in ``configs/agibotworld/train_configs.yaml`` to the root of AgiBotWorld dataset

5. Run the script
```
bash scripts/train.sh configs/agibotworld/train_config.yaml
```


## TODO
- [x] Minimal version of training code for [AgibotWorld dataset](https://github.com/OpenDriveLab/AgiBot-World) and pretrained weights.
- [x] Release train & val dataset.
- [x] Minimal version of training code for the challenge's dataset.
- [x] Release test dataset(without GT).  
- [x] Evaluation script.
- [ ] Submission instructions.



## Related Works
This project draws inspiration from the following projects:
- [EnerVerse](https://sites.google.com/view/enerverse)
- [EnerVerse-AC](https://github.com/AgibotTech/EnerVerse-AC)
- [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter)
- [LVDM](https://github.com/YingqingHe/LVDM)



## Citation
Please consider citing our paper if our codes are useful:
```bib
@article{huang2025enerverse,
  title={Enerverse: Envisioning Embodied Future Space for Robotics Manipulation},
  author={Huang, Siyuan and Chen, Liliang and Zhou, Pengfei and Chen, Shengcong and Jiang, Zhengkai and Hu, Yue and Liao, Yue and Gao, Peng and Li, Hongsheng and Yao, Maoqing and others},
  journal={arXiv preprint arXiv:2501.01895},
  year={2025}
}
@article{jiang2025enerverseac,
  title={EnerVerse-AC: Envisioning Embodied Environments with Action Condition},
  author={Jiang, Yuxin and Chen, Shengcong and Huang, Siyuan and Chen, Liliang and Zhou, Pengfei and Liao, Yue and He, Xindong and Liu, Chiming and Li, Hongsheng and Yao, Maoqing and Ren, Guanghui},
  journal={arXiv preprint arXiv:2505.09723},
  year={2025}
}
```


## License
All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

