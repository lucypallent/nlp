#### UNFINISHED need to make sure csv has the right comp ie rob is
# the same as tfidf

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import os
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from ast import literal_eval

SAVE_DIR = 'checkpoints/2B'

train_df = pd.read_csv('nlp_csv2/rob_train.csv')
valid_df = pd.read_csv('nlp_csv2/rob_val.csv')

train_df.drop(['Body ID', 'articleBody', 'headline'], axis=1, inplace=True)
valid_df.drop(['Body ID', 'articleBody', 'headline'], axis=1, inplace=True)

train_df.rob_articleBody = train_df.rob_articleBody.apply(literal_eval)
train_df.rob_headline = train_df.rob_headline.apply(literal_eval)

valid_df.rob_articleBody = valid_df.rob_articleBody.apply(literal_eval)
valid_df.rob_headline = valid_df.rob_headline.apply(literal_eval)

train_art = train_df.rob_articleBody.apply(pd.Series)
train_head = train_df.rob_headline.apply(pd.Series)

val_art = valid_df.rob_articleBody.apply(pd.Series)
val_head = valid_df.rob_headline.apply(pd.Series)

train_df = train_df.join(train_art)
train_df = train_df.join(train_head, lsuffix='art', rsuffix='head')
train_df.drop(['rob_articleBody', 'rob_headline'], axis=1, inplace=True)

valid_df = valid_df.join(val_art)
valid_df = valid_df.join(val_head)
valid_df.drop(['rob_articleBody', 'rob_headline'], axis=1, inplace=True)
print(val.columns)

# remove the 'unrelated' rows from the train_df and val_df
train_df = train_df[train_df['Stance'] != 'unrelated']
valid_df = valid_df[valid_df['Stance'] != 'unrelated']

# NEED TO look at how tdidf is definied to get values for below
# based on https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1
class multiDataset(Dataset):
    def __init__(self, df):
        # read csv file and load row data into varialbles
        file_out = df # pd.read_csv(file_name)
        di = {'discuss': 0, 'agree': 1, 'disagree': 2}
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

train_multi = multiDataset(train_df)
valid_multi = multiDataset(valid_df)

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

num_class = 3
vocab_size = 10001
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
        text = text.to(device)
        lable = label.to(device)
        predicted_label, unsoftmax_predicted_label = model(text.float())#, offsets)
        label # = torch.Tensor(label.float())
        # label = torch.reshape(label, (500, 1))
        # label2 = torch.zeros(500, 2)
        #
        # for i, x in enumerate(label):
        #     if x == 0:
        #         label2[i] = torch.Tensor([1, 0])
        #     else:
        #         label2[i] = torch.Tensor([0, 1])
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
            text = text.to(device)
            label = label.to(device)
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

train_dataloader = DataLoader(train_multi, batch_size=BATCH_SIZE,
                              shuffle=True)

valid_dataloader = DataLoader(valid_multi, batch_size=BATCH_SIZE,
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
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'rob-MLP.pth'))
print('model saved')
