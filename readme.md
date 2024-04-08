Name: Nishant Dhotre
Roll No: 23CS60R48

---

# Project README

## Overview
This project involves preprocessing input data, vectorizing text using a specified BERT model, training deep learning models, evaluating performance, and generating classification reports.

## Instructions

1. **Preprocess.py**: Splits and preprocesses raw data. 
   - Usage: `python Preprocess.py <file_path>`

2. **Vectorize.py**: Vectorizes preprocessed data using a BERT model specified via command-line. 
   - Usage: `python Vectorize.py <bert_model_name>`

3. **Model Training**: Train deep learning models (DNN, CNN, AutoModel) with vectorized data.
   - **DNN.py, CNN.py, AutoModel.py**: Takes vectorized file path as input. 
      - Usage: `python <ModelScript.py> <vectorized_data_path>`
      - you can add till 1 to 5 `<bert-vector-train-path>` paths one after another (all bert models)

4. **RunEval.py**: Evaluates models using checkpoint path, test split path, and model type. 
   - Usage: `python RunEval.py <checkpoints_path> <test_data.csv path> <vector embedding folder path> <DL Model name>`

5. **Model Checkpoints**: Save and share model checkpoints via Google Drive with 'editor' privileges.

6. **runEndtoEnd.sh**: Shell script that sequentially runs all tasks, generating a classification report. 
   - Usage: `./runEndtoEnd.sh <dataset-path> <bert-model-name>  <that-model-train-embedding-path> <model-chekipoint-path> <test-csv-path> <embedding-folder-path> <dl-model-name> `
   - Exmple:
   `./runEndtoEnd.sh ./dataset/data.csv bert-base-uncased ./dataset/embeddings/bert-base-uncased-train.npy ./Models_checkpoints/DNN/ ./dataset/test_data.csv  ./dataset/embeddings/ DNN`

7. **Report**: Includes hyperparameters, classification report outputs, and Google Drive link to saved model checkpoints.

8. **EvalTestCustom.py**: Performs inference on a held-out test set.
   - Usage: `python EvalTestCustom.py <path to csv> <bert model name> <path to dl model>`
   - Takes a CSV of social media posts and labels, BERT model type, and DL model path as input.
   - Preprocesses, vectorizes, loads the DL model, runs inference, and reports classification outputs.

## Additional Notes
- Ensure all Python files and shell scripts are executable and in the correct directory.
- Modify paths and model names as per your setup.

--- 

## Directory Description:
   - `.Models_checkpoints/`:
         contains check point for all DL models in their respective folder
      - `/DNN`,`/CNN`,`/AutoModel`: these folder have all 5 check point for respective bert model.
   - `./dataset`:contains datasets
   - `./dataset/embeddings/`: contains embedding for `test/train/val` of all bert model example `<bert-model-name-test>, <bert-model-name-test>, <bert-model-name-val>`


## command inputs:
- `python Preprocess.py <file_path>`
  - python Preprocess.py ./dataset/data.csv
-  `python Vectorize.py <bert_model_name>`
   - python Vectorize.py   bert-base-cased
-  `python <DNN.py> <vectorized_data_path>`
   - python DNN.py ./dataset/embeddings/bert-base-cased-train.npy
-  `python <CNN.py> <vectorized_data_path>`
   - python CNN.py ./dataset/embeddings/bert-base-cased-train.npy
-  `python <AutoModel.py> <vectorized_data_path>`
   - python AutoModel.py ./dataset/embeddings/bert-base-cased-train.npy
-  `python RunEval.py <checkpoints_path> <test_data.csv path> <vector embedding folder path> <DL Model name>`
   - python RunEval.py ./Models_checkpoints/DNN ./dataset/test_data.csv ./dataset/embeddings/ DNN
-  `./runEndtoEnd.sh <dataset-path> <bert-model-name>  <that-model-train-embedding-path> <model-chekipoint-path> <test-csv-path> <embedding-folder-path> <dl-model-name> `
   - ./runEndtoEnd.sh ./dataset/data.csv bert-base-uncased ./dataset/embeddings/bert-base-uncased-train.npy ./Models_checkpoints/DNN/ ./dataset/test_data.csv  ./dataset/embeddings/ DNN