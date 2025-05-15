import os, re
from omegaconf import OmegaConf
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
from collections import OrderedDict


def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo

def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None

def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        },
        "batch_logger": {
            "target": "callbacks.ImageLogger",
            "params": {
                "save_dir": logdir,
                "batch_frequency": 1000,
                "max_images": 4,
                "clamp": True,
            }
        },    
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                                                   'params': {
                                                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                                                        "filename": "{epoch}-{step}",
                                                        "verbose": True,
                                                        'save_top_k': -1,
                                                        'every_n_train_steps': 10000,
                                                        'save_weights_only': True
                                                    }
                                                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg

def get_trainer_logger(lightning_config, logdir):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg

def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPShardedStrategy"
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg

def load_checkpoints(model, model_cfg, ignore_mismatched_sizes=True):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        
        # try:
        if 'state_dict' in pl_sd.keys():
            
            state_dict = pl_sd["state_dict"]
            model_state_dict = model.state_dict()
            loaded_keys = list(state_dict.keys())
            original_loaded_keys = loaded_keys
            def _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                ignore_mismatched_sizes,
            ):
                mismatched_keys = []
                if ignore_mismatched_sizes:
                    for checkpoint_key in loaded_keys:
                        model_key = checkpoint_key

                        if (
                            model_key in model_state_dict
                            and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                        ):
                            mismatched_keys.append(
                                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                            )
                            del state_dict[checkpoint_key]

                return mismatched_keys

            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            
            mainlogger.info(">>> mismatched_keys: %s" % mismatched_keys)
            mainlogger.info(">>> missing: %s" % missing)
            mainlogger.info(">>> unexpected: %s" % unexpected)
            mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
        else:
            def _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                ignore_mismatched_sizes,
                rm_mismatched_weights=True,
            ):
                mismatched_keys = []
                if ignore_mismatched_sizes:
                    for checkpoint_key in loaded_keys:
                        model_key = checkpoint_key

                        if (
                            model_key in model_state_dict
                            and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                        ):
                            mismatched_keys.append(
                                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                            )
                            if rm_mismatched_weights:
                                del state_dict[checkpoint_key]

                return mismatched_keys
            
            # deepspeed
            new_pl_sd = OrderedDict()

            if "module" in pl_sd:
                pl_sd = pl_sd["module"]
            for key in pl_sd.keys():
                new_pl_sd[key[16:]]=pl_sd[key]
            state_dict = new_pl_sd
            model_state_dict = model.state_dict()
            loaded_keys = list(state_dict.keys())
            original_loaded_keys = loaded_keys
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes if not getattr(model_cfg, "auto_padzero_input_block", False) else True,
                False if getattr(model_cfg, "auto_padzero_input_block", False) else True,
            )
            

            if getattr(model_cfg, "auto_padzero_input_block", False):
                for k, _, _ in mismatched_keys:
                    if k.find("input_blocks.0.0.weight")>=0:
                        sd_shape = state_dict[k].shape
                        model_sd_shape = model_state_dict[k].shape
                        if model_sd_shape[1] > sd_shape[1]:
                            padding_zeros = torch.zeros(
                                sd_shape[0], model_sd_shape[1]-sd_shape[1], 3, 3
                            ).to(dtype=state_dict[k].dtype, device=state_dict[k].device)
                            state_dict.update({k: torch.cat((state_dict[k], padding_zeros), dim=1)})
                        else:
                            state_dict.update({k: state_dict[k][:,:model_sd_shape.shape[1]]})

            if getattr(model_cfg, "auto_padrand_input_block", False):
                for k, _, _ in mismatched_keys:
                    if k.find("input_blocks.0.0.weight")>=0:
                        sd_shape = state_dict[k].shape
                        model_sd_shape = model_state_dict[k].shape
                        if model_sd_shape[1] > sd_shape[1]:
                            padding_rands = torch.randn(
                                sd_shape[0], model_sd_shape[1]-sd_shape[1], 3, 3
                            ).to(dtype=state_dict[k].dtype, device=state_dict[k].device)
                            state_dict.update({k: torch.cat((state_dict[k], padding_zeros), dim=1)})
                        else:
                            state_dict.update({k: state_dict[k][:,:model_sd_shape.shape[1]]})

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            mainlogger.info(">>> mismatched_keys: %s" % mismatched_keys)
            mainlogger.info(">>> missing: %s" % missing)
            mainlogger.info(">>> unexpected: %s" % unexpected)
            mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)

        # except:
            # model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")

    return model

def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger