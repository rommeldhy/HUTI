# -*- coding: utf-8 -*-
# @Time   : 2020/8/28 14:32
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/10/2
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

r"""
FPMC
################################################

Reference:
    Steffen Rendle et al. "Factorizing Personalized Markov Chains for Next-Basket Recommendation." in WWW 2010.

"""
import torch
from torch import nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


from recbole.model.sequential_recommender.fpmc import FPMC
from recbole.model.loss import BPR_P_Loss

class FPMCP(FPMC):
    r"""The FPMC model is mainly used in the recommendation system to predict the possibility of
    unknown items arousing user interest, and to discharge the item recommendation list.

    Note:

        In order that the generation method we used is common to other sequential models,
        We set the size of the basket mentioned in the paper equal to 1.
        For comparison with other models, the loss function used is BPR.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(FPMCP, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.loss_type = config["loss_type"]

        # load dataset info
        self.n_users = dataset.user_num

        # define layers and loss
        # user embedding matrix
        self.UI_emb = nn.Embedding(self.n_users, self.embedding_size)
        # label embedding matrix
        self.IU_emb = nn.Embedding(self.n_items, self.embedding_size)
        # last click item embedding matrix
        self.LI_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # label embedding matrix
        self.IL_emb = nn.Embedding(self.n_items, self.embedding_size)

        if self.loss_type == "BPR":
            self.loss_fct = BPR_P_Loss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR']!")

        # parameters initialization
        self.apply(self._init_weights)


    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        p = interaction["score"]
        
        pos_score = self.forward(user, item_seq, item_seq_len, pos_items)
        neg_score = self.forward(user, item_seq, item_seq_len, neg_items)
        loss = self.loss_fct(pos_score, neg_score, p)

        return loss

