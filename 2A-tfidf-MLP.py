import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import os
from os import join

# di = {'unrelated': 0, 'discuss': 1, 'agree': 1, 'disagree': 1}
# tfidf = pd.read_csv('/content/drive/MyDrive/tfidf_val_head.csv')
# train_df = pd.read_csv('/content/drive/MyDrive/val.csv')
# train_df.drop(columns=['Body ID'], inplace=True)
# train_df = train_df[['Headline',	'Stance']].merge(tfidf, on='Headline')
# # tfidf.replace({'Stance': di}, inplace=True)
# # tfidf.drop(['body ID', 'articleBody', 'Headline'], inplace=True) - not running but will run on ncc
# # tfidf.iloc[:,1:].values will be x
# # tfidf.iloc[:,1].values #will be y
# train_df.head()

SAVE_DIR = 'checkpoints/2A-DL'

train_df = pd.read_csv('nlp_csv2/tfidf_train.csv')
valid_df = pd.read_csv('nlp_csv2/tfidf_val.csv')

train_df.drop(['Body ID', 'articleBody', 'Headline'], inplace=True)
valid_df.drop(['Body ID', 'articleBody', 'Headline'], inplace=True)

# NEED TO look at how tdidf is definied to get values for below
# based on https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1
class binaryDataset(Dataset):
    def __init__(self, df):
        # read csv file and load row data into varialbles
        file_out = df # pd.read_csv(file_name)
        di = {'unrelated': 0, 'discuss': 1, 'agree': 1, 'disagree': 1}
        file_out.replace({'Stance': di}, inplace=True)

        # x = file_out.iloc[:,2:].values # whatever those values will be
        # y = file_out['Stance'].values

        # tfidf.iloc[:,1:].values will be x
        # # tfidf.iloc[:,1].values #will be y
        x = file_out.iloc[:,1:].values
        y = file_out.iloc[:,1].values

        # converting to torch tensors
        self.X_train = torch.tensor(x) # x #
        self.y_train = torch.tensor(y) # y # torch.tensor(y)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.y_train[idx], self.X_train[idx]

train_bin = binaryDataset(train_df)
valid_bin = binaryDataset(valid_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassificationModel(nn.Module): # based on team which came third but without dropout

    def __init__(self, vocab_size, num_class):
        super(TextClassificationModel, self).__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(vocab_size, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, num_class)
        # self.softmax = nn.Linear(embed_dim, num_class)
        self.init_weights()

        self.dropout = nn.Dropout(0.4)
        # dropout is 1 - 0.6 (where 0.6 is dropout on layer outputs ie the keep prob)

    def init_weights(self):
        initrange = 0.5
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    # def forward(self, text, offsets):
    def forward(self, x):
        # x = self.embedding(text, offsets)
        x = F.relu(self.fc1(x))
        # x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        soft_x = F.softmax(x)
        return soft_x, x

num_class = 2
vocab_size = 5000
emsize = 64
model = TextClassificationModel(vocab_size, num_class).to(device)

import time
criterion = torch.nn.CrossEntropyLoss()

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        # optimizer2.zero_grad()
        # text = text.long()
        predicted_label, unsoftmax_predicted_label = model(text.float())#, offsets)
        label # = torch.Tensor(label.float())
        # label = torch.reshape(label, (500, 1))
        label2 = torch.zeros(500, 2)

        for i, x in enumerate(label):
            if x == 0:
                label2[i] = torch.Tensor([1, 0])
            else:
                label2[i] = torch.Tensor([0, 1])
        # print(label2)
        loss = criterion(unsoftmax_predicted_label, label.long()) #* 0.00001 # * is the regularisation of l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # this should be 5
        optimizer.step()
        # optimizer2.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader): # (label, text, offsets)
            # text = text.long()
            predicted_label, unsoftmax_predicted_label = model(text.float())#, offsets)

            loss = criterion(unsoftmax_predicted_label, label.long()) * 0.00001 # use the unsoftmaxed with the loss function in orig
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
# Hyperparameters
EPOCHS = 90 # epoch
LR = 0.01  # learning rate
BATCH_SIZE = 500 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# used adam optimiser
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# optimizer2 = torch.optim.SparseAdam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
# train_iter, test_iter = AG_NEWS()
# train_dataset = to_map_style_dataset(train_iter)
# test_dataset = to_map_style_dataset(test_iter)
# num_train = int(len(train_dataset) * 0.95)
# split_train_, split_valid_ = \
#     random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# train_dataloader = DataLoader(train_bin, batch_size=BATCH_SIZE,
#                               shuffle=True, collate_fn=collate_batch)

train_dataloader = DataLoader(train_bin, batch_size=BATCH_SIZE,
                              shuffle=True)

valid_dataloader = DataLoader(valid_bin, batch_size=BATCH_SIZE,
                              shuffle=True)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(valid_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    scheduler.step()
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

# save the model_output
print('saving model')
torch.save(model.state_dict(), os.join(SAVE_DIR, 'tfidf-MLP.pth'))
print('model saved')
