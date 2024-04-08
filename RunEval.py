import os 
import sys
import pandas as pd
import torch
from sklearn.metrics import classification_report
import numpy as np
import CNN as cnn
import DNN as dnn
import Preprocess as processIt
import Vectorize as vecIt

model_names = ['bert-base-uncased', 'bert-base-cased', 'covid-twitter-bert', 'twhin-bert-base', 'SocBERT-base']
model_name = model_names[0]

checkpoints_path = "./Models_checkpoints/DNN"
test_split_path = "./dataset/test_data.csv"
embedding_path = './dataset/embeddings/'
dl_model_name = "DNN"

def do_cmd():
    if len(sys.argv) > 3:
        checkpoints_path = sys.argv[1]
        test_split_path = sys.argv[2]
        embedding_path = sys.argv[3]
        dl_model_name = sys.argv[4]

        if not os.path.exists(checkpoints_path) or not os.path.exists(test_split_path):
            print("Invalid paths!")
            exit()

        if dl_model_name not in ["DNN", "CNN", "AutoModel"]:
            print("Invalid DL model name!")
            exit()
    else:
        print("invalid format!\nprovide: python RunEval.py <checkpoints_path> <test_data.csv path> <vector embedding folder path> <DL Model name>")
        exit()

def main():
    print("checkpoints_path =",checkpoints_path)
    print("test_split_path=",test_split_path)
    print('dl_model_name=',dl_model_name)
    print("Assuming you have provided test data csv path")
    
    # print("processing data....")
    df = processIt.read_CSV(test_split_path)
    # df['tweet'] = df['tweet'].apply(processIt.preprocessing)
    
    if df['label'][0] == 'real' or df['label'][0] == 'fake':
        df['label'] = df['label'].map({"real": 1, "fake": 0})
    print("Done....")


    embeddings = {}
    print("loading making the embeddings..")
    for bert_name, model_path in vecIt.model_names.items():
        if not os.path.exists(f'{embedding_path}/{bert_name}-test.npy'):
            print(f"invalid run...\nfile missing for {bert_name}.npy\n\tplease run Preprocess.py and vectorize.py first!")
            exit()
        embeddings[bert_name] = np.load(f'{embedding_path}/{bert_name}-test.npy')
    print("Done....")


    model = None
    model_cov = None
    if dl_model_name == "DNN":
         model = dnn.DNNModel()
         model_cov = dnn.DNNModel(1024)
    if dl_model_name == "CNN":
         model = cnn.TextCNN()
         model_cov = cnn.TextCNN(1024)

    print("prediction will start...")
    for bert_name in model_names:
        curr_model  = None
        if bert_name == 'covid-twitter-bert':
            model_cov.load_state_dict(torch.load(f'{checkpoints_path}/{bert_name}.pth'))
            curr_model =  model_cov
        else:     
            model.load_state_dict(torch.load(f'{checkpoints_path}/{bert_name}.pth'))
            curr_model = model

        X_test = embeddings[bert_name]
        y_test = df["label"].values
        curr_model.eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # Convert validation set
        with torch.no_grad():  # Inference mode, no need to compute gradients
            y_pred = curr_model(X_test_tensor)
            y_pred = y_pred.squeeze()  # Adjust dimensions if necessary
            if dl_model_name == 'CNN':
                y_pred = torch.argmax(y_pred, dim=1)


        y_pred_np = y_pred.numpy()  # Convert to NumPy array
        y_pred_labels = (y_pred_np > 0.5).astype(int)  # Convert probabilities to binary labels
        print("                 ",bert_name)
        print(classification_report(y_test, y_pred_labels))  # Assuming y_test contains true labels as a NumPy array
        print("                 ")






if __name__=="__main__":
    do_cmd() 
    main()

