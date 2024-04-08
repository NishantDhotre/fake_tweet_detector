import torch
import pickle
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import numpy as np
import sys
import os 


model_names = ['bert-base-uncased', 'bert-base-cased', 'covid-twitter-bert', 'twhin-bert-base', 'SocBERT-base']
vectorized_path = ""


def do_cmd():
    paths_of_vectors = []

    if len(sys.argv) == 1:
        print("invalid format!\nprovide: python DNN.py <bert-vector-train-path> ....")
        exit()

    for i in range(1, 4):
        if len(sys.argv) > i:
            vectorized_path = sys.argv[i]
            if not os.path.exists(vectorized_path):
                print("Invalid paths!")
                exit()
            paths_of_vectors.append(vectorized_path)
            
    return paths_of_vectors


 


def retrieve_data(vectorized_path, name, model_name):
    if not os.path.exists(vectorized_path):
        print(f"invalid run...\nfile missing for {model_name}.npy\n\tplease run Preprocess.py and vectorize.py first!")
        exit()
    x = np.load(vectorized_path)

    path = f"./dataset/{name}_data.csv"
    if not os.path.exists(path):
        print(f"invalid run...\nfile missing for {name}_data.csv\n\tplease run Preprocess.py and vectorize.py first!")
        exit()
    df = pd.read_csv(path)
    y = df['label'].values  # This converts the first column to a numpy array
    return x, y

class DNNModel(nn.Module):
    def __init__(self, embedding_dim=768):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)  # First hidden layer
        self.dropout = nn.Dropout(0.2)  # 20% dropout
        self.fc2 = nn.Linear(128, 128)  # Second hidden layer
        self.fc3 = nn.Linear(128, 100)  # Second hidden layer
        self.dropout = nn.Dropout(0.2)  # 10% dropout
        self.fc4 = nn.Linear(100, 64)  # Third hidden layer
        self.fc5 = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for first layer
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # Activation function for second layer
        x = F.relu(self.fc3(x))  # Activation function for second layer
        x = self.dropout(x)
        x = F.relu(self.fc4(x))  # Activation function for second layer
        x = torch.sigmoid(self.fc5(x))  # Sigmoid activation function for output
        return x


def model_train(num_epochs, X_train_tensor, y_train_tensor, bard_name):
    model = None
    if 'covid-twitter-bert' == bard_name:
        model = DNNModel(embedding_dim=1024)
    else:
        model = DNNModel()

    criterion = nn.BCELoss()  # Since it's a binary classification problem
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Learning rate can be adjusted

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)  # Squeeze is used to adjust dimension
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def save_model(model, bard_name):
    p = './Models_checkpoints'
    if not os.path.exists(p):
        os.makedirs(p)
    p = './Models_checkpoints/DNN'
    if not os.path.exists(p):
        os.makedirs(p)
    torch.save(model.state_dict(), f'./Models_checkpoints/DNN/{bard_name}.pth')


def predictions_model(X_val, y_val, bard_name):
    model = DNNModel()
    if 'covid-twitter' in bard_name:
        model = DNNModel(embedding_dim=1024)

    model.load_state_dict(torch.load(f'./Models_checkpoints/DNN/{bard_name}.pth'))
    # Ensure the model is in evaluation mode before making predictions
    model.eval()

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)  # Convert validation set
    with torch.no_grad():  # Inference mode, no need to compute gradients
        y_pred = model(X_val_tensor)
        y_pred = y_pred.squeeze()  # Adjust dimensions if necessary

    y_pred_np = y_pred.numpy()  # Convert to NumPy array
    y_pred_labels = (y_pred_np > 0.5).astype(int)  # Convert probabilities to binary labels

    print(classification_report(y_val, y_pred_labels))  # Assuming y_val contains true labels as a NumPy array



def main(vectorized_path, model_name):
    print("retrieving data ....")
    X_train, y_train = retrieve_data(vectorized_path, "train", model_name)
    print("Done ....")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Assuming X_train is a sparse matrix from TF-IDF
    y_train_tensor = torch.FloatTensor(y_train)  # Convert labels to tensor
    epochs = 30

    print("training model ....")
    model = model_train( num_epochs= epochs, X_train_tensor= X_train_tensor, y_train_tensor= y_train_tensor, bard_name= model_name)
    save_model(model, model_name)
    print("Done ....")

    # X_val, y_val = retrieve_data(vectorized_path, "val", model_name)
    # predictions_model(X_val, y_val, model_name)



if __name__=="__main__": 
    path_of_vectors = do_cmd()
    for path_of_vector in path_of_vectors:
        model_name = "None"
        for models in model_names:
            if models in path_of_vector:
                model_name = models
                break
        if model_name == "None":
            print("Invalid path!> ", path_of_vector)
        else:    
            print("\tworking on \'",model_name,"\'")
            main(path_of_vector, model_name)
            print("\n\n")
