#!/bin/bash

cd src

#module spider miniforge3
#module spider cuda/11.8.0

python train.py mot --exp_id macaques_rocamadour\
                    --load_tracking_model '../models/macaque_copypaste.pth'\
                    --num_epochs 50\
                    --lr_step 30\
                    --lr '5e-5'\
                    --data_cfg 'lib/cfg/rocamadour.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --gpus 0\
                    --data_dir '/user/vogg/u24033/.project/dir.project/Richard_Vogg/data/Rocamadour'\
                    --batch_size 4\
                    --seed 1\
                    --reid_cls_names monkey,snake,hand,cube,popcorn,peanut\
                    --val_intervals 10\
                    --save_all\
                    --primate_only
cd ..
