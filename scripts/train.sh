#!/usr/bin/env sh

ln -s /home/users/shengqin.tang/datasets/coco data
export PYTHONPATH="./:$PYTHONPATH"

# train faster-rcnn-r50-fpn 1x baseline
CUDA_VISIBLE_DEVICES=1,2,3 \
python -m torch.distributed.launch --nproc_per_node=3 ./tools/train.py \
    ./my_configs/faster_rcnn_r50_fpn_1x.py \
    --launcher pytorch \
    --work_dir ./work_dirs \
    --seed 1    \

# train faster-rcnn-r50-fpn 1x with dynamic-rcnn
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py \
#    ./configs/retinanet_r50_fpn_atss_1x.py \
#    --launcher pytorch \
#    --work_dir ./work_dirs \
#    --seed 2    \

