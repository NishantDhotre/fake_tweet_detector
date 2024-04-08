import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
import sys
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os

model_names = ['bert-base-uncased', 'bert-base-cased', 'covid-twitter-bert', 'twhin-bert-base', 'SocBERT-base']


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



class TextCNN(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=2):
        super(TextCNN, self).__init__()
        # Treat embedding dimension as sequence length for CNN
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (embedding_dim // 2 // 2), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape x to (batch_size, in_channels, sequence_length)
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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

def train_CNN(X_train_tensor, y_train_tensor,  epochs, bard_name):
    
    model = TextCNN()
    if 'covid-twitter' in bard_name:
        model = TextCNN(embedding_dim=1024,num_classes= 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Prepare data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.detach().clone())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    p = './Models_checkpoints'
    if not os.path.exists(p):
        os.makedirs(p)
    p = './Models_checkpoints/CNN'
    if not os.path.exists(p):
        os.makedirs(p)
    torch.save(model.state_dict(), f'./Models_checkpoints/CNN/{bard_name}.pth')


def do_prediction(X_val, y_val, bard_name):
    model = TextCNN()  # Reinitialize the model
    if 'covid-twitter' in bard_name:
        model = TextCNN(embedding_dim=1024,num_classes= 2)

    model.load_state_dict(torch.load(f'./Models_checkpoints/CNN/{bard_name}.pth'))

    model.eval()  # Ensure model is in evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # For tensors
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)  # Convert validation set
    print("model import done....")
    
    # Assuming batch_size is set to a smaller value you've chosen
    batch_size = 64  # or smaller if necessary
    # Calculate the number of batches
    n_batches = (X_val_tensor.size(0) + batch_size - 1) // batch_size


    print("doing prediction....")
    predicted_classes = []

    for i in range(n_batches):
        start_i = i * batch_size
        end_i = start_i + batch_size
        X_batch = X_val_tensor[start_i:end_i].to(device)  # Ensure it's on the right device

        with torch.no_grad():
            output = model(X_batch)
            predicted_class = torch.argmax(output, dim=1)
            predicted_classes.extend(predicted_class.cpu().numpy())  # Move back to CPU if necessary

    print("done it....")


    
    model = model.to('cpu')
    X_val_tensor = X_val_tensor.to('cpu')


    # Calculate and print classification report
    report = classification_report(y_val, predicted_classes)
    print(report)




def main(vectorized_path, model_name):
    print("retrieving data ....")
    X_train, y_train = retrieve_data(vectorized_path, "train", model_name)
    print("Done ....")


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Assuming X_train is a sparse matrix from TF-IDF
    y_train_tensor = torch.tensor(y_train).long()  # Convert labels to tensor


    print("training CNN.........")
    train_CNN( X_train_tensor= X_train_tensor, y_train_tensor= y_train_tensor,  epochs= 4, bard_name= model_name)
    print("Done.........")

    # print("prediction calc.........")
    # X_val, y_val = retrieve_data(vectorized_path, "val", model_name)
    # do_prediction(X_val, y_val, model_name)
    # print("Done.........")

 


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





