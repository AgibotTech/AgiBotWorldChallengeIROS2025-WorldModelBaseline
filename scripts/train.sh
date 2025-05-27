#!/usr/local/bash

name="agibotworld"
tag="_expv0"
config_file=${1:-configs/${name}/train_config.yaml}

save_root="./log"

seed=123

NGPU=`nvidia-smi --list-gpus | wc -l`
export OMP_NUM_THREADS=4

echo "Training on 1 Node, $NGPU GPUs"
echo $config_file

exit


torchrun --nnodes=1 \
    --nproc_per_node=$NGPU \
    --node_rank=0 \
    trainer/trainer.py \
    --base $config_file \
    --train \
    --seed $seed \
    --name ${name}${tag} \
    --logdir $save_root \
    --devices $NGPU \
    lightning.trainer.num_nodes=1

# echo $?