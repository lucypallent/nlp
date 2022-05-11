# based on: https://huggingface.co/sentence-transformers/all-roberta-large-v1

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import scipy
from scipy import spatial

train = pd.read_csv('nlp_csv2/tfidf_train.csv')
# test = pd.read_csv('nlp_csv2/tfidf_test.csv')
val = pd.read_csv('nlp_csv2/tfidf_val.csv')

# train_body = pd.read_csv('nlp_csv2/tfidf_train_body.csv')

tra_articleBody = train['articleBody'].tolist()
tra_headline = train['Headline'].tolist()
# tes_articleBody = test['articleBody'].tolist()
# tes_headline = test['Headline'].tolist()
val_articleBody = val['articleBody'].tolist()
val_headline = val['Headline'].tolist()

def create_unique_list(original_list):
    new_list = []
    for i in original_list:
        if i not in new_list:
            new_list.append(i)
    return new_list

val_headline_unique = create_unique_list(val_headline)
val_articleBody_unique = create_unique_list(val_articleBody)
tra_headline_unique = create_unique_list(tra_headline)
tra_articleBody_unique = create_unique_list(tra_articleBody)
# tes_headline_unique = create_unique_list(tes_headline)
# tes_articleBody_unique = create_unique_list(tes_articleBody)

print('create unique')
headline_unique = val_headline_unique + tra_headline_unique #+ tes_headline_unique

articleBody_unique = val_articleBody_unique + tra_articleBody_unique #+ tes_articleBody_unique
print('create large unique')

headline_unique_df = pd.DataFrame(headline_unique, columns=['headline'])
articleBody_unique_df = pd.DataFrame(articleBody_unique, columns=['headline'])

# df = pd.DataFrame(list(zip(headline_unique, articleBody_unique)), columns =['headline', 'articleBody'])

# df.to_csv('nlp_csv2/unique.csv', index=False)

headline_unique_df.to_csv('nlp_csv2/headline_unique.csv', index=False)
articleBody_unique_df.to_csv('nlp_csv2/articleBody_unique.csv', index=False)

print('df saved')

print('WORKS!')
