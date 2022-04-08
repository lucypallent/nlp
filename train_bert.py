import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer#, CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import torch

import neptune.new as neptune
random.seed(42); torch.manual_seed(42); np.random.seed(42)

run = neptune.init(
    project="lucypallent/natural-language-processing",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMmI3Y2EyOC1kZGMzLTRiNjgtYjY1MS04ZmZlMzA5MjJiYTYifQ==",
)  # your credentials

# url_trb = 'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_bodies.csv'
url_trb = 'nlp_csv/train_bodies.csv'
train_bodies = pd.read_csv(url_trb)

# url_trs = 'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_stances.csv'
url_trs = 'nlp_csv/train_stances.csv'
train_stances = pd.read_csv(url_trs)

# url_teb = 'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/test_bodies.csv'
url_teb = 'nlp_csv/test_bodies.csv'
test_bodies = pd.read_csv(url_teb)

# url_tes = 'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/test_stances_unlabeled.csv'
url_tes = 'nlp_csv/test_stances_unlabeled.csv'
test_stances = pd.read_csv(url_tes)

train = train_bodies.merge(train_stances, on='Body ID')
test = test_bodies.merge(test_stances, on='Body ID')

train['articleBody'] = train['articleBody'].str.lower()
train['Headline'] = train['Headline'].str.lower()

test['articleBody'] = test['articleBody'].str.lower()
test['Headline'] = test['Headline'].str.lower()


#Now read the file back into a Python list object
with open('nlp_csv/stop.txt', 'r') as f:
    stop = json.loads(f.read())
stop = set(stop)

# # stop = set(stopwords.words('english'))
#
train['articleBody'] = train['articleBody'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train['Headline'] = train['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test['articleBody'] = test['articleBody'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
test['Headline'] = test['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)

    return text.translate(table)

example = "I am #king"
print(remove_punct(example))

train['articleBody'] = train['articleBody'].apply(lambda x: remove_punct(x))
train['Headline'] = train['Headline'].apply(lambda x: remove_punct(x))

test['articleBody'] = test['articleBody'].apply(lambda x: remove_punct(x))
test['Headline'] = test['Headline'].apply(lambda x: remove_punct(x))

train['articleBody'] = train['articleBody'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train['Headline'] = train['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test['articleBody'] = test['articleBody'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
test['Headline'] = test['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

all_words = pd.concat([test['articleBody'], test['Headline'], train['articleBody'], train['Headline']], axis=0, ignore_index=True).to_frame().rename(columns={0: 'Headline'})

# bert transformer
# !pip install datasets
# !pip install transformers

from datasets import Dataset
tdidf = Dataset.from_pandas(all_words)

from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    result = tokenizer(examples["Headline"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# Use batched=True to activate fast multithreading!
tokenized_datasets = tdidf.map(
    tokenize_function, batched=True, remove_columns=['Headline']
)

def group_texts(examples):
    chunk_size = 128
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return default_data_collator(features)

train_size = len(lm_datasets)
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets.train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

# from transformers import TrainingArguments

# batch_size = 64
# Show the training loss with every epoch
# logging_steps = len(downsampled_dataset["train"]) // batch_size
# model_name = model_checkpoint.split("/")[-1]

# training_args = TrainingArguments(
#     output_dir=f"{model_name}-finetuned-test-headline",
#     overwrite_output_dir=True,
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     push_to_hub=True,
#     fp16=True,
#     logging_steps=logging_steps,
# )

# !curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
# !sudo apt-get install git-lfs

# from transformers import Trainer
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=downsampled_dataset["train"],
#     eval_dataset=downsampled_dataset["test"],
#     data_collator=data_collator,
# )

import math

# eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
#
# trainer.train()
#
# eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

### think this is where the new bit runs
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

from torch.utils.data import DataLoader
from transformers import default_data_collator

batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# !pip install accelerate

from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 3 # change to 200 or something
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

from huggingface_hub import get_full_repo_name

model_name = "distilbert-base-uncased-finetuned-test-headline"
repo_name = get_full_repo_name(model_name)
repo_name

from huggingface_hub import Repository

output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)

from tqdm.auto import tqdm
import torch
import math

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    run["training/epoch"].log(epoch)

    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss

        run["training/loss"].log(loss)

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
    run["eval/perplexity"].log(perplexity)


    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )

from transformers import pipeline
full_dataloader = DataLoader(
    tdidf['Headline'], batch_size=1#, collate_fn=default_data_collator
)

# saving model
model.save_pretrained('bert-model-test')

mask_filler = pipeline(
    'feature-extraction', model=model, tokenizer=tokenizer, framework='pt', device=0
)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader, full_dataloader, mask_filler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, full_dataloader, mask_filler
)

all_words['bert'] = ''
for step, batch in enumerate(full_dataloader):
    with torch.no_grad():
        title = accelerator.prepare(batch[0])
        all_words.iloc[step, 1] = mask_filler(title)

all_words.to_csv('nlp_csv/all_words.csv', index=False)

# import pickle
# file_name = "test.pkl"
#
# open_file = open(file_name, "wb")
# pickle.dump(x, open_file)
# open_file.close()
# #
