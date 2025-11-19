#!/bin/bash


python PriMAT-tracking/src/train.py mot --exp_id macaques_rocamadour\
                    --load_tracking_model '../.project/dir.project/workshop-models/hrnetv2_w32_imagenet_pretrained.pth'\
                    --num_epochs 50\
                    --lr_step 30\
                    --lr '1e-5'\
                    --data_cfg 'PriMAT-tracking/src/lib/cfg/experiment_paths.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --gpus 0\
                    --batch_size 4\
                    --data_dir '../.project/dir.project/workshop-data'\
                    --seed 1\
                    --reid_cls_names monkey,snake,hand,cube,popcorn,peanut\
                    --val_intervals 10\
                    --save_all
cd ..
