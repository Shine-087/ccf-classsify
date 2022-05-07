"""
模型构建
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertConfig
import config

import transformers
transformers.logging.set_verbosity_error()

class Classify_Model(nn.Module):
    def __init__(self,pretrain_model):
        super(Classify_Model, self).__init__()
        bert_config = BertConfig.from_pretrained(pretrain_model)
        bert_config.attention_probs_dropout_prob = config.DROPOUT
        bert_config.hidden_dropout_prob = config.DROPOUT
        self.bert = BertModel.from_pretrained(pretrain_model,config=bert_config)

        self.linear = nn.Linear(in_features=768,out_features=10,bias=False)

    def forward(self,input_ids,attention_mask,token_type_ids):

        out = self.bert(input_ids,attention_mask,token_type_ids)
        out  = self.linear(out.pooler_output)

        return F.normalize(out,p=2,dim=1)
