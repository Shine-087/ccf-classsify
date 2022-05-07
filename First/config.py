#参数的设置
import torch

classes2idx = {'财经': 0, '房产': 1, '家居': 2, '教育': 3, '科技': 4,
               '时尚': 5, '时政': 6, '游戏': 7, '娱乐': 8, '体育': 9}
idx2classes = {0: '财经', 1: '房产', 2: '家居', 3: '教育', 4: '科技',
               5: '时尚', 6: '时政', 7: '游戏', 8: '娱乐', 9: '体育'}

rel_dict = {'财经': '高风险', '时政': '高风险', '房产': '中风险', '科技': '中风险',
            '教育': '低风险', '时尚': '低风险', '游戏': '低风险', '家居': '可公开',
            '体育': '可公开', '娱乐': '可公开'}

model_path = 'hfl/chinese-roberta-wwm-ext'
max_len = 512
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DROPOUT = 0.4
Batch_Size = 32
Test_batch_size = 128
LR = 1e-5
weight_decay = 1e-2
EPOCHS = 5

train_path = '../corpus/labeled_data.csv'
test_path = '../corpus/test_data.csv'
SAVE_MODEL = '../model_result/model_best.pth'
Pred_Result = '../model_result/pred.csv'
submit_result = '../model_result/submit.csv'
