#!/bin/bash

# Run Preprocess.py with file path as argument
python Preprocess.py "$1"

# Run Vectorize.py with BERT model name as argument
python Vectorize.py "$2"

# Run DNN.py with   argument
python DNN.py $3 

# Run CNN.py with   argument
python CNN.py $3 

# Run AutoModel.py with  argument
# python AutoModel.py $3  

# Run RunEval.py with  argument
python RunEval.py  $4 $5 $6 $7



# Exmple:
#  ./runEndtoEnd.sh ./dataset/data.csv bert-base-uncased ./dataset/embeddings/bert-base-uncased-train.npy ./Models_checkpoints/DNN/ ./dataset/test_data.csv  ./dataset/embeddings/ DNN