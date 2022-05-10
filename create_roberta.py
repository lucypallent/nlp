# based on: https://huggingface.co/sentence-transformers/all-roberta-large-v1

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import scipy
from scipy import spatial
from scipy import sparse
from sklearn.metrics.pariwise import cosine_similarity

val = pd.read_csv('nlp_csv2/val.csv')
unique = pd.read_csv('nlp_csv2/unique.csv')

# merge on articleBody
val = val.merge(unique[['articleBody', 'rob_articleBody']], how='inner', on='articleBody')

val = val.merge(unique[['headline', 'rob_headline']], how='inner', on='headline')

print('successfully merged')

val['rob_cos'] = val.apply(lambda row: cosine_similarity(row['rob_articleBody'], row['rob_headline']), axis = 1)

val.to_csv('nlp_csv2/rob_val.csv', index=False)

print('works')
