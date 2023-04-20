# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/9, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1, 2022/7/6
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng, Jiawei Guan, Gaowei Zhang
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com, guanjw@ruc.edu.cn, zgw15630559577@163.com

"""
recbole.data.utils
########################
"""

import copy
import importlib
import os
import pickle

from recbole.data.dataloader import *
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, ensure_dir, get_local_time, set_color
from recbole.utils.argument_list import dataset_arguments

import random
import math
import torch
import pandas as pd


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module("recbole.data.dataset")
    if hasattr(dataset_module, config["model"] + "Dataset"):
        dataset_class = getattr(dataset_module, config["model"] + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            ModelType.GENERAL: "Dataset",
            ModelType.SEQUENTIAL: "SequentialDataset",
            ModelType.CONTEXT: "Dataset",
            ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
            ModelType.TRADITIONAL: "Dataset",
            ModelType.DECISIONTREE: "Dataset",
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(
        config["checkpoint_dir"], f'{config["dataset"]}-{dataset_class.__name__}.pth'
    )
    file = config["dataset_save_path"] or default_file
    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ["seed", "repeatable"]:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from", "pink") + f": [{file}]")
            return dataset

    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config["checkpoint_dir"])
    save_path = config["checkpoint_dir"]
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color("Saving split dataloaders into", "pink") + f": [{file_path}]")
    Serialization_dataloaders = []
    for dataloader in dataloaders:
        generator_state = dataloader.generator.get_state()
        dataloader.generator = None
        dataloader.sampler.generator = None
        Serialization_dataloaders += [(dataloader, generator_state)]

    with open(file_path, "wb") as f:
        pickle.dump(Serialization_dataloaders, f)


def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """

    default_file = os.path.join(
        config["checkpoint_dir"],
        f'{config["dataset"]}-for-{config["model"]}-dataloader.pth',
    )
    dataloaders_save_path = config["dataloaders_save_path"] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, "rb") as f:
        dataloaders = []
        for data_loader, generator_state in pickle.load(f):
            generator = torch.Generator()
            generator.set_state(generator_state)
            data_loader.generator = generator
            data_loader.sampler.generator = generator
            dataloaders.append(data_loader)

        train_data, valid_data, test_data = dataloaders
    for arg in dataset_arguments + ["seed", "repeatable", "eval_args"]:
        if config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    logger = getLogger()
    logger.info(
        set_color("Load split dataloaders from", "pink")
        + f": [{dataloaders_save_path}]"
    )
    return train_data, valid_data, test_data


def pred_data_preparation(config, dataset, train_t_dataset, train_v_dataset, valid_dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.
    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        K (int): K-fold's K
    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training * k.
            - valid_data (AbstractDataLoader): The dataloader for validation * k.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    
    model_type = config["MODEL_TYPE"]
    
    # train_t, valid_dataset, train_v (get pred data)
    train_sampler, valid_sampler, test_sampler = create_samplers(
        config, dataset, (train_t_dataset, valid_dataset, train_v_dataset)
    )
    
    if model_type != ModelType.KNOWLEDGE:
        train_data = get_dataloader(config, "train")(
            config, train_t_dataset, train_sampler, shuffle=config["shuffle"]
        )

    valid_data = get_dataloader(config, "evaluation")(
        config, valid_dataset, valid_sampler, shuffle=False
    )
    test_data = get_dataloader(config, "evaluation")(
        config, train_v_dataset, test_sampler, shuffle=False
    )
    
    if config["save_dataloaders"]:
            save_split_dataloaders(
                config, dataloaders=(train_data, valid_data, test_data)
            )

    return train_data, valid_data, test_data

def split_train_dataset_by_users(train_dataset, K=5, seed=2022):
    
    user_torch = train_dataset.inter_feat.user_id.to('cpu')
    user_list = list(set(user_torch.tolist()))
    user_index_dict = {}
    for u in user_list:
        user_index_dict[u] = torch.where(user_torch==u)[0]
    
    random.seed(seed)
    random.shuffle(user_list)
    user_num = len(user_list)
    
    train_t_dataset_list = []
    train_v_dataset_list = []
    for i in range(K):
        
        tv_user = user_list[math.ceil(1/K*float(user_num))*i: math.ceil(1/K*float(user_num))*(i+1)]
        tt_user = [u for u in user_list if u not in tv_user]
        tt_index_list = [user_index_dict[u] for u in tt_user]
        tt_index = torch.cat(tt_index_list, dim=-1)
        tv_index_list = [user_index_dict[u] for u in tv_user]
        tv_index = torch.cat(tv_index_list, dim=-1)

        train_dataset.inter_feat[tt_index]
        train_dataset.inter_feat[tv_index]
        train_t_dataset = train_dataset.copy(train_dataset.inter_feat[tt_index])
        train_v_dataset = train_dataset.copy(train_dataset.inter_feat[tv_index])
        train_t_dataset_list.append(train_t_dataset)
        train_v_dataset_list.append(train_v_dataset)
    
    return train_t_dataset_list, train_v_dataset_list

# def save_split_dataloaders(config, dataloaders, k, b):
#     """Save split dataloaders.
#     Args:
#         config (Config): An instance object of Config, used to record parameter information.
#         dataloaders (tuple of AbstractDataLoader): The split dataloaders.
#     """
#     ensure_dir(config["checkpoint_dir"])
#     save_path = config["checkpoint_dir"]
#     saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader_k%d_b%d.pth'%(k, b)
#     file_path = os.path.join(save_path, saved_dataloaders_file)
#     logger = getLogger()
#     logger.info(set_color("Saving split dataloaders into", "pink") + f": [{file_path}]")
#     Serialization_dataloaders = []
#     for dataloader in dataloaders:
#         generator_state = dataloader.generator.get_state()
#         dataloader.generator = None
#         dataloader.sampler.generator = None
#         Serialization_dataloaders += [(dataloader, generator_state)]

#     with open(file_path, "wb") as f:
#         pickle.dump(Serialization_dataloaders, f)

def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        "MacridVAE": _get_AE_dataloader,
        "CDAE": _get_AE_dataloader,
        "ENMF": _get_AE_dataloader,
        "RaCT": _get_AE_dataloader,
        "RecVAE": _get_AE_dataloader,
    }

    if config["model"] in register_table:
        return register_table[config["model"]](config, phase)

    model_type = config["MODEL_TYPE"]
    if phase == "train":
        if model_type != ModelType.KNOWLEDGE:
            return TrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:
        eval_mode = config["eval_args"]["mode"]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _get_AE_dataloader(config, phase):
    """Customized function for VAE models to get correct dataloader class.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase == "train":
        return UserDataLoader
    else:
        eval_mode = config["eval_args"]["mode"]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.
    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    phases = ["train", "valid", "test"]
    train_neg_sample_args = config["train_neg_sample_args"]
    eval_neg_sample_args = config["eval_neg_sample_args"]
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    if train_neg_sample_args["distribution"] != "none":
        if not config["repeatable"]:
            sampler = Sampler(
                phases,
                built_datasets,
                train_neg_sample_args["distribution"],
                train_neg_sample_args["alpha"],
            )
        else:
            sampler = RepeatableSampler(
                phases,
                dataset,
                train_neg_sample_args["distribution"],
                train_neg_sample_args["alpha"],
            )
        train_sampler = sampler.set_phase("train")

    if eval_neg_sample_args["distribution"] != "none":
        if sampler is None:
            if not config["repeatable"]:
                sampler = Sampler(
                    phases,
                    built_datasets,
                    eval_neg_sample_args["distribution"],
                )
            else:
                sampler = RepeatableSampler(
                    phases,
                    dataset,
                    eval_neg_sample_args["distribution"],
                )
        else:
            sampler.set_distribution(eval_neg_sample_args["distribution"])
        valid_sampler = sampler.set_phase("valid")
        test_sampler = sampler.set_phase("test")

    return train_sampler, valid_sampler, test_sampler


def load_data_and_model_from_files(model_file, dataset_file, dataloader_file):
    r"""Load filtered dataset, split dataloaders and saved model.
    Args:
        model_file (str): The path of saved model file.
    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch
    import pickle
    from recbole.utils import (
        init_logger,
        get_model,
        get_trainer,
        init_seed,
        set_color,
        get_flops,
    )

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)
            
#     if type(dataset.inter_feat) is type(pd.DataFrame()):
#         dataset._change_feat_format()  
        
    with open(dataloader_file, "rb") as f:
        dataloaders = []
        for data_loader, generator_state in pickle.load(f):
            generator = torch.Generator()
            generator.set_state(generator_state)
            data_loader.generator = generator
            data_loader.sampler.generator = generator
            dataloaders.append(data_loader)

        train_data, valid_data, test_data = dataloaders
            
        train_data.update_config(config)
        valid_data.update_config(config)
        test_data.update_config(config)
 
        
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data

def get_dataset_with_score(model_name, ori_dataset_name, K, B, T, P, seed):
    
    random.seed(seed)

    dataset_name = "%s-%s-k%d-b%d-t%s-p%s" % (model_name, ori_dataset_name, K, B, format(T, '.1f'), format(P, '.1f'))

    dir_path = os.path.join("dataset", dataset_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, "%s.inter"%dataset_name)

    input_dir_path = os.path.join("dataset_with_rank", ori_dataset_name, "k%d"%K)

    file_list = [file for file in os.listdir(input_dir_path) if file.startswith(ori_dataset_name+"_"+model_name) and file.endswith(".csv")]
    file_list = random.sample(file_list, B)
#     file_list = file_list[:B]
    df_list = [pd.read_csv(os.path.join(input_dir_path, file)) for file in file_list]
    df_ori = df_list[0].drop(['rank'], axis=1)
    df_rank = pd.concat([df['rank'] for df in df_list], axis=1)
    df_rank.columns = [ 'rank%d'%(i+1) for i in range(len(df_list))]
    df_rank['min'] = df_rank.min(axis=1)

    threshold = df_rank[df_rank['min'] >= 0]['min'].quantile(T)

    df_rank['score'] = (df_rank['min'] <= threshold).astype('float') * 1.0  +(df_rank['min'] > threshold).astype('float') * P
    df_final=pd.concat([df_ori, df_rank['score']], axis=1)
    df_final.to_csv(file_path, index=False, header=["user_id:token", "item_id:token", "timestamp:float", "score:float"], sep="\t")