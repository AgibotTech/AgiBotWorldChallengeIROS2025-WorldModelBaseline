# World Model Baseline

We adopt [EVAC](https://huggingface.co/agibot-world/EnerVerse-AC) as the baseline model for the [AgiBot World Challenge @ IROS 2025](https://agibot-world.com/challenge) - World Model track.

This repo provides a minial version of training codes. 


## Getting started

### Setup
```
git clone --recursive https://github.com/AgibotTech/AgiBotWorldChallengeIROS2025-WorldModelBaseline.git
conda create -n enerverse python=3.10.4
conda activate enerverse
pip install -r requirements.txt`
```

### Training

1. Download the checkpoint from [EVAC](https://huggingface.co/agibot-world/EnerVerse-AC), and modify ``model.pretrained_checkpoint`` in ``configs/agibotworld/train_configs.yaml`` to the checkpoint file ``*.pt``

2. Download the weight of [CLIP](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K), and modify ``model.params.img_cond_stage_config.params.abspath``
in ``configs/agibotworld/train_configs.yaml`` to the absolute path to ``open_clip_pytorch_model.bin`` inside the download directory

3. Modify the path ``data.params.train.params.data_roots`` in ``configs/agibotworld/train_configs.yaml`` to the root of AgiBotWorld dataset

4. Run the script
```
bash scripts/train.sh
```


## TODO
- [x] Minimal version of training code for [AgibotWorld dataset](https://github.com/OpenDriveLab/AgiBot-World) and pretrained weights.
- [ ] Minimal version of training code for the challenge's dataset. (available once the challenge dataset is ready)  
- [ ] Evaluation script.
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
```


## License
All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

