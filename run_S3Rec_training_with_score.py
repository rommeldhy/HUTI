# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse
from ast import arg

from recbole.pred_training.training_with_score import run_training_with_score
from recbole.pred_training.pred_data_utils import get_dataset_with_score

import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="K-fold predicatability",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=5,
        help="num of base models",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=0.5,
        help="quantile",
    )
    
    parser.add_argument(
        "--P",
        type=float,
        default=0.5,
        help="score",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=202301,
        help="seed",
    )
    
    
    args, _ = parser.parse_known_args()
    
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    
    get_dataset_with_score(args.model, args.dataset, K=args.K, B=args.B, T=args.T, P=args.P, seed=args.seed)
    
    dataset_p = "%s-%s-k%d-b%d-t%s-p%s" % (args.model, args.dataset, args.K, args.B, format(args.T, '.1f'), format(args.P, '.1f'))
    shutil.copyfile(os.path.join('dataset', args.dataset,'%s.item'%args.dataset), os.path.join('dataset', dataset_p, '%s.item'%dataset_p)) 
    if args.nproc == 1 and args.world_size <= 0:
        run_training_with_score(
            model=args.model+'P', dataset=dataset_p, config_file_list=config_file_list, K=args.K, B=args.B
        )
    else:
        pass
