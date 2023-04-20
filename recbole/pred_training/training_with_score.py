# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.pred_training.training_with_score
########################
"""
import logging
from logging import getLogger

import sys

import pickle
from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
    load_split_dataloaders,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
)




def run_training_with_score(
    model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, K=5, B=5
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    
    config_dict={'load_col':{'inter': ['user_id', 'item_id', 'timestamp', 'score']}}
    saved=False
    
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )

    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    # transform = construct_transform(config)
    # flops = get_flops(model, dataset, config["device"], logger, transform)
    # logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    test_result = trainer.evaluate(test_data, load_best_model=saved)

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")


