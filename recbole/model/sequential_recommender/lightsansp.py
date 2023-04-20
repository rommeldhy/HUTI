# -*- coding: utf-8 -*-
# @Time    : 2021/05/01
# @Author  : Xinyan Fan
# @Email   : xinyan.fan@ruc.edu.cn

"""
LightSANs
################################################
Reference:
    Xin-Yan Fan et al. "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." in SIGIR 2021.
Reference:
    https://github.com/BELIEVEfxy/LightSANs
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
# from recbole.model.loss import BPRLoss
from recbole.model.layers import LightTransformerEncoder

from recbole.model.sequential_recommender.lightsans import LightSANs
from recbole.model.loss import BPR_P_Loss, CE_P_Loss

class LightSANsP(LightSANs):
    def __init__(self, config, dataset):
        super(LightSANsP, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.k_interests = config["k_interests"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.seq_len = self.max_seq_length
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = LightTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            k_interests=self.k_interests,
            hidden_size=self.hidden_size,
            seq_len=self.seq_len,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPR_P_Loss()
        elif self.loss_type == "CE":
            self.loss_fct = CE_P_Loss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        p = interaction["score"]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score, p)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items, p)
            return loss
