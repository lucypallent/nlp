import xgboost as xgb
import pandas as pd
import numpy as np
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import os
from ast import literal_eval


SAVE_DIR = 'checkpoints/2A-ML'

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

train_df.rob_cos = train_df.rob_cos.apply(literal_eval)
train_df.rob_cos = train_df.rob_cos.apply(pd.Series)
train_df.rob_cos = train_df.rob_cos.apply(pd.Series)

valid_df.rob_cos = valid_df.rob_cos.apply(literal_eval)
valid_df.rob_cos = valid_df.rob_cos.apply(pd.Series)
valid_df.rob_cos = valid_df.rob_cos.apply(pd.Series)

train_df = train_df.join(train_art)
train_df = train_df.join(train_head, lsuffix='art', rsuffix='head')
train_df.drop(['rob_articleBody', 'rob_headline'], axis=1, inplace=True)

valid_df = valid_df.join(val_art)
valid_df = valid_df.join(val_head, lsuffix='art', rsuffix='head')
valid_df.drop(['rob_articleBody', 'rob_headline'], axis=1, inplace=True)

di = {'unrelated': 0, 'discuss': 1, 'agree': 1, 'disagree': 1}
train_df.replace({'Stance': di}, inplace=True)
valid_df.replace({'Stance': di}, inplace=True)

print(train_df['rob_cos'])
print(train_df['rob_cos'].dtype)

X_train = train_df.iloc[:,1:].to_numpy()
Y_train = train_df.iloc[:,1].to_numpy()
X_valid = valid_df.iloc[:,1:].to_numpy()
Y_valid = valid_df.iloc[:,1].to_numpy()

print(train_df.iloc[:,1])
print(train_df.iloc[:,1].unique())

D_train = xgb.DMatrix(X_train, label=Y_train)
D_valid = xgb.DMatrix(X_valid, label=Y_valid)

param = {
    'eta': 0.3,
    'max_depth': 3,
    'objective': 'multi:softprob',
    'num_class': 2}

steps = 20  # The number of training iterations

model = xgb.train(param, D_train, steps)

preds = model.predict(D_valid)
best_preds = np.asarray([int(np.round(np.argmax(line))) for line in preds])

print(best_preds.dtype)
print(Y_valid.dtype)
print(valid_df.columns)
Y_valid = Y_valid.astype(int)

# print(preds)
# print(len(preds))
# print(preds.shape)
# print(best_preds)
# print(len(best_preds))
# print(best_preds.shape)
# print(Y_valid)
# print(len(Y_valid))
# print(Y_valid.shape)
#
# print(np.unique(best_preds))
#
# print(np.unique(best_preds, return_counts=True))

print('results with original paramaters')
print("Precision = {}".format(precision_score(Y_valid, best_preds, average='macro')))
print("Recall = {}".format(recall_score(Y_valid, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_valid, best_preds)))

# clf = xgb.XGBClassifier()
# parameters = {
#      "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#      "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
#      "min_child_weight" : [ 1, 3, 5, 7 ],
#      "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#      "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
#      "num_class"        : [3],
#      'objective'        : ['multi:softprob']
#      }
#
# grid = GridSearchCV(clf,
#                     parameters, n_jobs=4,
#                     scoring="neg_log_loss",
#                     cv=3)
#
# grid.fit(X_train, Y_train)
#
# print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid.best_params_)
#
# model2 = xgb.train(grid.best_params_, D_train, steps)
#
# preds2 = model2.predict(D_test)
# best_preds2 = np.asarray([np.argmax(line) for line in preds2])
#
# print('best preds')
# print("Precision = {}".format(precision_score(Y_valid, best_preds2, average='macro')))
# print("Recall = {}".format(recall_score(Y_valid, best_preds2, average='macro')))
# print("Accuracy = {}".format(accuracy_score(Y_valid, best_preds2)))
#
# save the model_output
print('saving model')
model.save_model(os.path.join(SAVE_DIR, 'rob-XGB.json'))
# model2.save_model(os.path.join(SAVE_DIR, 'rob-XGB2.json'))
print('model saved')
