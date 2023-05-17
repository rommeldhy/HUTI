#!/bin/bash
gpu_id='0'
seed=202300
dataset=Amazon_Beauty

i_list=(1 2 3 4)

for i in ${i_list[@]}
do        
    echo "i: ${i} LightSANs"
    python run_training_with_score.py --model=LightSANs --dataset=${dataset} --gpu_id=${gpu_id} --seed=$((seed+i)) --config_files=config_maxlen50_LightSANs.yaml --save_dataset=False --save_dataloaders=False
done