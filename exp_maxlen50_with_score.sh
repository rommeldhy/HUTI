#!/bin/bash
gpu_id='0'
seed=202300
dataset=Amazon_Beauty
K=5
B=5
T=0.4
P=0.3

i_list=(1 2 3 4)

for i in ${i_list[@]}
do        

    echo "i: ${i} LightSANs ${P} ${T}"
    python run_training_with_score.py --model=LightSANs --dataset=${dataset} --gpu_id=${gpu_id} --seed=$((seed+i)) --config_files=config_maxlen50_LightSANs.yaml --save_dataset=False --save_dataloaders=False --K=${K} --B=${B} --P=${P} --T=${T}

done