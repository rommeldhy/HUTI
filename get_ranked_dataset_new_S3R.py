import argparse
import os

import pandas as pd

from recbole.quick_start import run_recbole, load_data_and_model
from recbole.data.utils import *
from recbole.utils.case_study import *

from recbole.pred_training.pred_data_utils import (
    load_data_and_model_from_files
)


def get_rank_dataset_from_files(folder_name, model_file_name, dataset_file_name, dataloader_file_name, gpu_id="0"):
    cp_path = os.path.join('saved',folder_name)
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model_from_files(
        model_file=os.path.join(cp_path,model_file_name),
        dataset_file=os.path.join('saved', dataset_file_name),  # 'saved/' + 'ml-100k-dataset.pth',
        
        dataloader_file=os.path.join(cp_path, dataloader_file_name)  # cp_path + 'ml-100k-for-GRU4Rec-dataloader.pth'
    )
    
    uid_field = test_data.dataset.uid_field
    iid_field = test_data.dataset.iid_field
    
    dataset_test = test_data.dataset
    
    user_list = list(set(dataset_test.inter_feat[uid_field].tolist()))
    
    train_dataset_ori, valid_dataset_ori, test_dataset_ori = dataset.build()
    user_list_new = []
    item_list_new = []
    timestamp_list_new = []
    rank_list_new = []
    
    for u in user_list:
    
        first_item = dataset_test[dataset_test.inter_feat[uid_field] == u]['item_id_list'][0][0].item()
        target_item_list = dataset_test[dataset_test.inter_feat[uid_field] == u]['item_id'].tolist()
        last_two_item = valid_dataset_ori[valid_dataset_ori.inter_feat[uid_field]==u][iid_field][0].item()
        last_item = test_dataset_ori[test_dataset_ori.inter_feat[uid_field]==u][iid_field][0].item()
        first_timestamp = dataset_test[dataset_test.inter_feat[uid_field] == u]['timestamp_list'][0][0].item()
        timestamp_list = dataset_test[dataset_test.inter_feat[uid_field] == u]['timestamp'].tolist()
        last_two_timestamp = valid_dataset_ori[valid_dataset_ori.inter_feat[uid_field]==u]['timestamp'][0].item()
        last_timestamp = test_dataset_ori[test_dataset_ori.inter_feat[uid_field]==u]['timestamp'][0].item()

        score = full_sort_scores([u], model, test_data, device=config['device'])
        logits_p = score[torch.arange(score.shape[0]), target_item_list]
        score[torch.arange(score.shape[0]), target_item_list] = -torch.inf
        predictions = torch.cat([logits_p.unsqueeze(1), score], dim=-1)
        rank = torch.argsort(torch.argsort(predictions, descending=True))[:, 0].to('cpu').tolist()

        user_list_new.extend([u for _ in range(len(target_item_list)+3)])
        item_list_new.append(first_item)
        item_list_new.extend(target_item_list)
        item_list_new.append(last_two_item)
        item_list_new.append(last_item)
        timestamp_list_new.append(first_timestamp)
        timestamp_list_new.extend(timestamp_list)
        timestamp_list_new.append(last_two_timestamp)
        timestamp_list_new.append(last_timestamp)
        rank_list_new.append(-1)
        rank_list_new.extend(rank)
        rank_list_new.append(-2)
        rank_list_new.append(-2)
        
    rank_dataset = pd.DataFrame({"user_id": dataset_test.id2token(uid_field, user_list_new), "item_id": dataset_test.id2token(iid_field, item_list_new), "timestamp": timestamp_list_new, "rank": rank_list_new})

    return rank_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="GRU4Rec", help="name of models")
parser.add_argument(
    "--dataset", "-d", type=str, default="ml-1m", help="name of datasets"
)
parser.add_argument("--K", type=int, default=5, help="K")
parser.add_argument("--B", type=int, default=3, help="B")
parser.add_argument("--gpu_id", type=str, default="1", help="ID of GPU")

args = parser.parse_args()
model_name = args.model
dataset_name = args.dataset
K = args.K
B = args.B

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

dataset_file_name = dataset_name + '-dataset.pth'

folder_list = [name for name in os.listdir("saved") if name.startswith("%s-for-%s"%(dataset_name, model_name)) and name.endswith("K%d"%K)]

kb_dict = {}

for folder in folder_list:
    folder_kb = folder[len("%s-for-%s"%(dataset_name, model_name)):]
    k = int(folder_kb.split("_")[1][1:])
    b = int(folder_kb.split("_")[2][1:])
    kb_dict[(k, b)] = folder


assert len(kb_dict) == K * B


    
for b in range(B):
    df_list = []
    for k in range(K):
        print((k, b))
        folder = kb_dict[(k, b)]
        f0 = os.listdir(os.path.join('saved', folder))[0]
        f1 = os.listdir(os.path.join('saved', folder))[1]
        if f0.startswith(model_name):
            tmp = f0
            f0 = f1
            f1 = tmp
        dataloader_file_name = f0
        model_file_name = f1
        df = get_rank_dataset_from_files(folder, model_file_name, dataset_file_name, dataloader_file_name)
        df_list.append(df)
    df_all = pd.concat(df_list)
    df_all = df_all.sort_values(by=['user_id', 'timestamp'], ignore_index=True)
    new_path = os.path.join("dataset_with_rank", dataset_name, "k%d"%K)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    df_all.to_csv(os.path.join(new_path, "%s_%s_b%d.csv"%(dataset_name,model_file_name[:-4], b)), index=False)
