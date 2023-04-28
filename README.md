# Handling Unpredictable Target Items in Sequential Recommendation: A Free Lunch Approach

This is our implementation for the paper under review.

The experiment is based on *[Recbole](https://github.com/RUCAIBox/RecBole)*.

## Usage

**Requirements**

* Python 3.8
* PyTorc 1.10
* Recbole v1.1.1

**Dataset**

The datasets used in our experiment can be downloaded from the *[dataset](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj)* page of Recbole.

**Run**

An example where LightSANs is the base model and Amazon_Beauty is the dataset:

* Step 1: pre-train for detecting unpredictable items.
``` bash
nohup python run_pred_training.py --model=LightSANs --dataset=Amazon_Beauty --gpu_id='0' --seed=202301 --config_files=config_maxlen50_LightSANs.yaml --K=5 --B=5 &
```

* Step 2: detecting unpredictable items and get a dataset with rankings, which is stored in `./dataset_with_rank`.
``` bash
nohup python get_ranked_dataset_new.py --model=LightSANs --dataset=Amazon_Beauty --K=5 --B=5 --gpu_id=0 &
```

* Step 3: Train the model with predictability score.
``` bash
nohup python run_training_with_score.py --model=LightSANs --dataset=Amazon_Beauty --gpu_id=0 --seed=202301 --config_files=config_maxlen50_LightSANs.yaml --save_dataset=False --save_dataloaders=False --K=5 --B=5 --P=0.3 --T=0.4 &
```

To reproduce the results in table 2, you can firstly run the Step 1 and Step 2, and then run the following script.
``` bash
nohup bash exp_maxlen50_with_score.sh &
```

To reproduce the results of baselines, you can run the following script.
``` bash
nohup bash exp_maxlen50_baseline.sh &
```