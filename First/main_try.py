"""
main:训练
"""
import csv
import os
import random
import torch.nn as nn
import pandas as pd
import torch
import numpy as np
import config
from model import Classify_Model
from data_try import TrainDataset,TestDataset,process_data
from torch.utils.data import DataLoader
from sklearn.utils import shuffle as reset
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

best_score = 0.

def set_seed(seed):
    """
    设置种子
    :param seed: random number
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmack = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def flat_accuracy(preds,labels):
    """
    计算准确率
    :param preds:
    :param labels:
    :return:
    """
    pred_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat / len(labels_flat))

def save(model,optimizer):
    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    },config.SAVE_MODEL)
    print('\nThe best model has been saved.')

def train_eval_data(data_df,ratio_eval=0.2,shuffle=True,random_state=None):
    """
    分割训练集和评估集
    :param data_df: 数据集
    :param ratio_eval: 设置eval比例
    :param shuffle: 打乱数据
    :param random_state:
    :return:
    """
    if shuffle:
        data_df = reset(data_df,random_state=random_state)

    train_data = data_df[int(len(data_df)*ratio_eval):].reset_index(drop=True)
    eval_data = data_df[:int(len(data_df)*ratio_eval)].reset_index(drop=True)

    return train_data,eval_data

def train_eval(model,criterion,optimizer,train_loader,val_loader,epochs):
    # check_point = torch.load(config.SAVE_MODEL)
    # model.load_state_dict(check_point['model_state_dict'])

    model.to(config.DEVICE)
    #total_steps = len(train_loader) * epoches

    print('============training=============')

    for epoch in range(epochs):
        model.train()
        print(f'Epoch {epoch +1}')
        for i, data in enumerate(tqdm(train_loader)):
            # data = tuple(t.to(config.DEVICE) for t in data)
            output = model(data[0],data[1],data[2])
            label = torch.tensor(data[3]).to(config.DEVICE)
            loss = criterion(output,label)
            # loss = criterion(output,data[3])

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()

            if i %10 == 0:
                print(i,loss.item())

            if i %50 == 49:
                evaluate(model,optimizer,val_loader)

def evaluate(model,optimizer,val_dataloader):
    model.to(config.DEVICE)
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0.,0.,0
    for batch in val_dataloader:
        # batch = tuple(t.to(config.DEVICE) for t in batch)
        with torch.no_grad():
            output = model(batch[0],batch[1],batch[2])
            batch[3] = torch.tensor(batch[3]).to(config.DEVICE)
            output = output.detach().cpu().numpy()
            label_ids = batch[3].cpu().numpy()

            temp_eval_accuracy = flat_accuracy(output,label_ids)
            eval_accuracy += temp_eval_accuracy
            nb_eval_steps += 1

    print('\nValidation Accuracy: {}'.format(eval_accuracy/nb_eval_steps))

    global best_score

    if best_score < eval_accuracy / nb_eval_steps:
        best_score =  eval_accuracy / nb_eval_steps
        save(model,optimizer)

def test(model,test_dataloader):
    check_point = torch.load(config.SAVE_MODEL)

    model.load_state_dict(check_point['model_state_dict'])
    model.to(config.DEVICE)

    print('=============Testing==============')
    pred_label = []
    model.eval()
    for i,batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            output = model(batch[0],batch[1],batch[2])
            output = output.detach().cpu().numpy()
            preds = np.argmax(output,axis=1).flatten()
            pred_label.extend(preds)
            # pred_label.append(preds)  #error : Writing 32 cols but got 1 aliases 问题就是append() 要改成 extend()
    pd.DataFrame(data=pred_label,index=range(len(pred_label))).to_csv(config.Pred_Result,header=['class_label'],encoding='utf-8')

    with open(config.Pred_Result,encoding='utf-8') as f:
        rows = [row for row in csv.reader(f)]
        rows = np.array(rows[1:])      #array([['0', '0'],['1', '1'],['2', '2'],['3', '3']....[idx,label]
        label_list = [label for _,label in rows]
        final_col = []
        for i in label_list:
            final_col.append(config.rel_dict[config.idx2classes[int(i)]])

        data = pd.read_csv(config.Pred_Result)
        data['rank_label'] = final_col
        data = data.replace({'class_label':config.idx2classes})
        data.to_csv(config.submit_result,index = False,header=['id','class_label','rank_label'],encoding='utf-8')

    print('Test completed')

if __name__ == '__main__':
    set_seed(1234)

    print('Reading training dataset...')
    train_dataset = TrainDataset(mode='train')
    train_dataloader = DataLoader(train_dataset,batch_size=config.Batch_Size,shuffle=True,drop_last=True)

    print('Reading validation data...')
    val_dataset = TrainDataset(mode='eval')
    val_dataloader = DataLoader(val_dataset,batch_size=config.Batch_Size,shuffle=True,drop_last=True)

    model = Classify_Model(pretrain_model=config.model_path).to(config.DEVICE)
    # check_point = torch.load(config.SAVE_MODEL)
    # model.load_state_dict(check_point['model_state_dict'])
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(),lr=config.LR,weight_decay=config.weight_decay)

    train_eval(model,criterion,optimizer,train_dataloader,train_dataloader,config.EPOCHS)

    print('Reading test data...')
    test_dataset = TestDataset()
    test_dateloader = DataLoader(dataset=test_dataset,batch_size=config.Test_batch_size,shuffle=True,drop_last=True)

    test(model,test_dateloader)