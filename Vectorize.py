from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import sys

model_names = {'bert-base-uncased': 'google-bert/bert-base-uncased', 'bert-base-cased': 'google-bert/bert-base-cased', 'covid-twitter-bert': 'digitalepidemiologylab/covid-twitter-bert', 'twhin-bert-base': 'Twitter/twhin-bert-base', 'SocBERT-base':'sarkerlab/SocBERT-base'}

bert_name = 'bert-base-uncased'

def do_cmd():
    if len(sys.argv) > 1:
        bert_name = sys.argv[1]
        if bert_name not in model_names:
            print("invalid <bert model name>!", bert_name)
            exit()
    else:
        print("invalid!: provide\npython Vectorize.py <bert-model-name>")
        exit()

def make_embedding(df, model_name, model_path, name):
    # Load the tokenizer and model
    model = None
    tokenizer  = None

    if model_name not in  ['twhin-bert-base', 'SocBERT-base']:
        model = BertModel.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    else:
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load your data

    tweets = df['tweet'].tolist() 
    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 800  # Adjust based on your available memory
    embeddings = []

    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=100)
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)




    # Concatenate all batch embeddings
    final_embeddings = np.concatenate(embeddings, axis=0)

    p = './dataset/embeddings'
    if not os.path.exists(p):
        os.makedirs(p)
    p = './dataset/embeddings/{name}'
    if not os.path.exists(p):
        os.makedirs(p)

    # Save embeddings to file
    np.save(f'./dataset/embeddings/{model_name}-{name}.npy', final_embeddings)


def make_embedding_with_df(df, model_name, model_path):
    # Load the tokenizer and model
    model = None
    tokenizer  = None

    if model_name not in  ['twhin-bert-base', 'SocBERT-base']:
        model = BertModel.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    else:
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load your data

    # df = pd.read_csv(data_path)
    tweets = df['tweet'].tolist()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 800  # Adjust based on your available memory
    embeddings = []

    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=100)
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)




    # Concatenate all batch embeddings
    final_embeddings = np.concatenate(embeddings, axis=0)


    # Save embeddings to file
    # np.save(f'./dataset/embeddings/{name}/{model_name}_embeddings.npy', final_embeddings)
    return final_embeddings


def main(df, name):
    print(f"working on {bert_name}..")
    make_embedding(df, bert_name, model_names[bert_name], name)
    print("done..")



if __name__=="__main__":
    do_cmd()
    for ele in ["train", "test", "val"]:
        df = pd.read_csv(f'./dataset/{ele}_data.csv')
        main(df, ele) 