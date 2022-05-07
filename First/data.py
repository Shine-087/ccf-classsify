import torch
from tqdm import tqdm
import pandas as pd
import random
import os
from torch.utils.data import Dataset
from transformers import BertTokenizer
import config

class TrainDataset(Dataset):
    def __init__(self,data,model_path=config.model_path,label=True):
        super(TrainDataset, self).__init__()

        self.data = data
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def process_tokenizer(self,content):
        tokenizer = self.tokenizer(
            [content],
            truncation=True,
            add_special_tokens=True,
            max_length = config.max_len,
            padding='max_length',
            return_tensors='pt'
        ).to(config.DEVICE)
        return tokenizer

    def __getitem__(self, idx):
        text = str(self.data.loc[idx, 'content'])
        sample = self.process_tokenizer(text)
        token_ids = sample['input_ids'].squeeze(0).to(config.DEVICE)
        attention_mask = sample['attention_mask'].squeeze(0).to(config.DEVICE)
        token_type_ids = sample['token_type_ids'].squeeze(0).to(config.DEVICE)
        if self.label:
            label = self.data.loc[idx,'class_label']
            return token_ids,attention_mask,token_type_ids,label
        else:
            return token_ids,attention_mask,token_type_ids

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self,root='../corpus/',model_path =config.model_path):
        super(TestDataset, self).__init__()

        self.root = root
        self.test_path = os.path.join(self.root,'test_data.csv')
        self.all_data = []
        self.create_data()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def create_data(self):
        data = pd.read_csv(self.test_path,sep=',',encoding='utf-8')
        num_sample = len(data['content'])
        for i in range(num_sample):
            content = data['content'][i]
            self.all_data.append(content)

    def __getitem__(self, idx):

        sample = self.tokenizer(
            self.all_data[idx],
            truncation=True,
            add_special_tokens=True,
            max_length=config.max_len,
            padding='max_length',
            return_tensors='pt'
        ).to(config.DEVICE)

        input_ids = sample['input_ids'].squeeze(0)
        attention_mask = sample['attention_mask'].squeeze(0)
        token_type_ids = sample['token_type_ids'].squeeze(0)
        return input_ids,attention_mask,token_type_ids

    def __len__(self):
        return len(self.all_data)

def process_data(root,classes2idx,label=True):
    data = pd.read_csv(root,encoding='utf-8')
    print(len(data))
    if label:
        data = data.replace({'class_label': classes2idx})
    return data