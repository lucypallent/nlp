# based on: https://huggingface.co/sentence-transformers/all-roberta-large-v1

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import scipy
from scipy import spatial
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import numpy as np

def create_cos_sim_column(df_pth, sv_pth):
    # val = pd.read_csv('nlp_csv2/val.csv')
    val = pd.read_csv(df_pth)
    val.rename(columns = {'Headline':'headline'}, inplace = True)

    unique = pd.read_csv('nlp_csv2/unique.csv')

    # merge on articleBody
    val = val.merge(unique[['articleBody', 'rob_articleBody']], how='inner', on='articleBody')

    val = val.merge(unique[['headline', 'rob_headline']], how='inner', on='headline')

    # print('successfully merged')
    #
    # print(type(val['rob_articleBody'][2]))
    # print(type(val['rob_headline'][2]))

    val.rob_articleBody = val.rob_articleBody.apply(literal_eval)
    val.rob_headline = val.rob_headline.apply(literal_eval)

    # print(type(val['rob_headline'][2]))
    # print(type(val['rob_articleBody'][2]))

    print(len(val['rob_headline'][2]))
    print(len(val['rob_headline'][2]))




    val['rob_cos'] = val.apply(lambda row: cosine_similarity(np.array(row['rob_articleBody']), np.array(row['rob_headline'])), axis = 0) # was 1 before

    # val.to_csv('nlp_csv2/rob_val.csv', index=False)
    val.to_csv(sv_pth, index=False)

create_cos_sim_column('nlp_csv2/val.csv', 'nlp_csv2/rob_val.csv')
# create_cos_sim_column('nlp_csv2/train.csv', 'nlp_csv2/rob_train.csv')
# create_cos_sim_column('nlp_csv2/test.csv', 'nlp_csv2/rob_test.csv')

print('works')
