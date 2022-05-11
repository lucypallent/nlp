# based on: https://huggingface.co/sentence-transformers/all-roberta-large-v1

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import scipy
from scipy import sparse
from scipy import spatial
import math

# unique = pd.read_csv('nlp_csv2/unique.csv')

headline_unique_df = pd.read_csv('nlp_csv2/headline_unique.csv')
articleBody_unique_df = pd.read_csv('nlp_csv2/articleBody_unique.csv')

uni_articleBody = articleBody_unique_df['articleBody'].tolist()
uni_headline = headline_unique_df['headline'].tolist()

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')

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
    return sparse.csr_matrix(sentence_embeddings)

m = create_embed(uni_headline[:5])
len_headline = math.ceil(len(uni_headline) / 5)
for i in range(1, len_headline):
    m_tmp = create_embed(uni_headline[5*i:5*(i+1)])
    m = scipy.sparse.vstack((m, m_tmp))


print(m.todense().shape)
print(len(m.todense().tolist()))
print(len(m.todense().tolist()[0]))
headline_unique_df['rob_headline'] = m.todense().tolist()

# rob_uni_articleBody = create_embed(uni_articleBody[:5])
m = create_embed(uni_articleBody[:5])
len_articleBody = math.ceil(len(uni_articleBody) / 5)
for i in range(1, len_articleBody):
    m_tmp = create_embed(uni_articleBody[5*i:5*(i+1)])
    m = scipy.sparse.vstack((m, m_tmp))

articleBody_unique_df['rob_articleBody'] = m.todense().tolist()



# m_new = scipy.sparse.vstack((m1, m2))


# unique['rob_articleBody'] = rob_uni_headline
# unique['rob_headline'] = rob_uni_articleBody

print('updated unique dataframe')

# unique.to_csv('nlp_csv2/unique.csv', index=False)
headline_unique_df.to_csv('nlp_csv2/headline_unique.csv', index=False)
articleBody_unique_df.to_csv('nlp_csv2/articleBody_unique.csv', index=False)
print('df saved')


#
# # save the dataframes
# train.to_csv('nlp_csv2/rob_train.csv', index=False)
# test.to_csv('nlp_csv2/rob_test.csv', index=False)
# val.to_csv('nlp_csv2/rob_val.csv', index=False)


# # caculate the cosine similarity between the two
# train['rob_cos'] = train.apply(lambda row: scipy.spatial.distance.cosine(row['rob_articleBody'], row['rob_Headline']), axis = 1)
# test['rob_cos'] = test.apply(lambda row: scipy.spatial.distance.cosine(row['rob_articleBody'], row['rob_Headline']), axis = 1)
# val['rob_cos'] = val.apply(lambda row: scipy.spatial.distance.cosine(row['rob_articleBody'], row['rob_Headline']), axis = 1)
#
# # save the dataframes
# train.to_csv('nlp_csv2/rob_train.csv', index=False)
# test.to_csv('nlp_csv2/rob_test.csv', index=False)
# val.to_csv('nlp_csv2/rob_val.csv', index=False)

print('WORKS!')
