import pandas as pd
from sklearn.utils import shuffle as reset
import os
from torch.utils.data import Dataset
from transformers import BertTokenizer
import config

class TrainDataset(Dataset):
    def __init__(self,root='../corpus/',model_path=config.model_path,label=True,mode='train'):
        super(TrainDataset, self).__init__()

        self.root = root
        self.mode = mode
        self.path = os.path.join(self.root,'labeled_data.csv')
        self.label = label
        self.all_data = {}
        self.create_data()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def create_data(self,ratio_eval=0.2,random_state=None):
        data_df = pd.read_csv(self.path,sep=',',encoding='utf-8')
        data_df = data_df.replace({'class_label': config.classes2idx})      #替换标签的形式，便于训练
        data_df = reset(data_df,random_state=random_state)                  #开始训练之前打乱数据顺序
        if self.mode == 'train':
            train_data = data_df[int(len(data_df) * ratio_eval):].reset_index(drop=True)
            num_sample = len(train_data['content'])
            for i in range(num_sample):
                content = train_data['content'][i]
                label = train_data['class_label'][i]
                self.all_data[i] = content + '||' + str(label)
        if self.mode == 'eval':
            eval_data = data_df[ :int(len(data_df) * ratio_eval)].reset_index(drop=True)
            num_sample = len(eval_data['content'])
            for i in range(num_sample):
                content = eval_data['content'][i]
                label = eval_data['class_label'][i]
                self.all_data[i] = content + '||' + str(label)

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
        text,label = str(self.all_data[idx]).split('||')
        sample = self.process_tokenizer(text)
        token_ids = sample['input_ids'].squeeze(0).to(config.DEVICE)
        attention_mask = sample['attention_mask'].squeeze(0).to(config.DEVICE)
        token_type_ids = sample['token_type_ids'].squeeze(0).to(config.DEVICE)
        if self.label:
            return token_ids,attention_mask,token_type_ids,int(label)
        else:
            return token_ids,attention_mask,token_type_ids

    def __len__(self):
        return len(self.all_data)


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