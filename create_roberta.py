# based on: https://huggingface.co/sentence-transformers/all-roberta-large-v1

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import scipy
from scipy import spatial

train = pd.read_csv('nlp_csv2/train.csv')
test = pd.read_csv('nlp_csv2/test.csv')
val = pd.read_csv('nlp_csv2/val.csv')

tra_articleBody = train['articleBody'].tolist()
tra_headline = train['Headline'].tolist()
tes_articleBody = test['articleBody'].tolist()
tes_headline = test['Headline'].tolist()
val_articleBody = val['articleBody'].tolist()
val_headline = val['Headline'].tolist()

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')

def create_embed(sentences):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

train['rob_articleBody'] = create_embed(tra_articleBody)
train['rob_Headline'] = create_embed(tra_headline)
test['rob_articleBody'] = create_embed(tes_articleBody)
test['rob_Headline'] = create_embed(tes_headline)
val['rob_articleBody'] = create_embed(val_articleBody)
val['rob_Headline'] = create_embed(val_headline)

# caculate the cosine similarity between the two
train['rob_cos'] = train.apply(lambda row: scipy.spatial.distance.cosine(row['rob_articleBody'], row['rob_Headline']), axis = 1)
test['rob_cos'] = test.apply(lambda row: scipy.spatial.distance.cosine(row['rob_articleBody'], row['rob_Headline']), axis = 1)
val['rob_cos'] = val.apply(lambda row: scipy.spatial.distance.cosine(row['rob_articleBody'], row['rob_Headline']), axis = 1)

# save the dataframes
train.to_csv('nlp_csv2/rob_train.csv', index=False)
test.to_csv('nlp_csv2/rob_test.csv', index=False)
val.to_csv('nlp_csv2/rob_val.csv', index=False)

print('WORKS!')
