import pickle
import pandas as pd
import torch.nn as nn
from sklearn.metrics import classification_report
import numpy as np
import sys
import Preprocess as processIt
import Vectorize as vectorIt
import DNN as dnn
import CNN as cnn
import os 
import torch


model_names = {'bert-base-uncased': 'google-bert/bert-base-uncased', 'bert-base-cased': 'google-bert/bert-base-cased', 'covid-twitter-bert': 'digitalepidemiologylab/covid-twitter-bert', 'twhin-bert-base': 'Twitter/twhin-bert-base', 'SocBERT-base':'sarkerlab/SocBERT-base'}




def do_cmd():
    path_to_csv= './dataset/val_data.csv'
    bert_model_name= "bert-base-uncased"
    path_dl_model = "Models_checkpoints\DNN\bert-base-cased.pth"
    if len(sys.argv)>3:
        path_to_csv = sys.argv[1]
        bert_model_name = sys.argv[2]
        path_dl_model = sys.argv[3]
        if not os.path.exists(path_to_csv) or not os.path.exists(path_dl_model):
            print("invalid paths!")
            exit()
        if bert_model_name not in model_names:
            print("invalid bert model name!")
            exit()
    else:
        print("invalid!\nprovide\t python3 EvalTestCustom.py <path to csv> <bert model name> <path to dl model>")
        exit()
    return path_to_csv, bert_model_name, path_dl_model



def do_prediction(X_test, y_test, model, path_dl_model):
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # Convert validation set
    with torch.no_grad():  # Inference mode, no need to compute gradients
        y_pred = model(X_test_tensor)
        y_pred = y_pred.squeeze()  # Adjust dimensions if necessary
        if  'CNN' in path_dl_model:
            y_pred = torch.argmax(y_pred, dim=1)


    y_pred_np = y_pred.numpy()  # Convert to NumPy array
    y_pred_labels = (y_pred_np > 0.5).astype(int)  # Convert probabilities to binary labels
    print("                 ",bert_model_name)
    print(classification_report(y_test, y_pred_labels))  # Assuming y_test contains true labels as a NumPy array
    print("                 ")


def main(path_to_csv, bert_model_name, path_dl_model):
    df = processIt.read_CSV(path_to_csv)
    df = df
    print("preprocessing .....")
    df['tweet'] = df['tweet'].apply(processIt.preprocessing)
    if df['label'][0] == 'real' or df['label'][0] == 'fake':
        df['label'] = df['label'].map({"real": 1, "fake": 0})
    print("done....")

    print("making Embeddings..")
    X_embeddings = vectorIt.make_embedding_with_df(df= df, model_name= bert_model_name, model_path= model_names[bert_model_name])
    print("done....")
    model = None
    inp_dim = 768

    if bert_model_name == 'covid-twitter-bert':
        inp_dim = 1024

    if   "DNN" in path_dl_model:
        model = dnn.DNNModel(inp_dim)
    if  "CNN" in path_dl_model:
        model = cnn.TextCNN(inp_dim)


    model.load_state_dict(torch.load(path_dl_model))

    X_test = X_embeddings
    y_test = df["label"].values

    do_prediction(X_test=X_test, y_test=y_test, model=model, path_dl_model=path_dl_model)



if __name__=="__main__": 
    path_to_csv, bert_model_name, path_dl_model = do_cmd()
    main(path_to_csv, bert_model_name, path_dl_model)

