import re
import emoji
import string
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

p = './dataset/data.csv'
def do_cmd():
    if len(sys.argv) > 1:
        p = sys.argv[1]
        if not os.path.exists(p):
            print("Invalid paths!")
            exit()
    else:
        print("invalid!: provide\npython Preprocess.py <dataset path_including file name>")
        exit()


# -----------------------------------------------------------------reading csv file
def read_CSV(p):
    path = p
    df = pd.read_csv(path)
    return df
 


# #---------------------------------------------------------------- Task-1: Prepare dataset

def make_csv(name, X, y):
    data = pd.DataFrame({'tweet': X, 'label': y})
    p = './dataset'
    if not os.path.exists(p):
        os.makedirs(p)
    file_path = f'{p}/{name}.csv'
    data.to_csv(file_path, index=False)

# train test validation split
def prep_data(df):
    X = df["tweet"]
    y = df["label"]

    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_test, X_val, y_test, y_val = train_test_split(X_remaining, y_remaining, test_size = 0.5, random_state = 0)

    make_csv('train_data', X_train, y_train)
    make_csv('test_data', X_test, y_test)
    make_csv('val_data', X_val, y_val)

     

# # --------------------------------------------------------------  Task-2: Preprocessing Social Media Post

def preprocessing(text):
    # Compile regular expressions outside the function
    url_pattern = re.compile(r'https?://([a-zA-Z0-9\-$_.+!*\'(),;:@&=]+)')

    # Pre-load stop words
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    # Initialize PorterStemmer
    ps = PorterStemmer()

    # Replace URLs with the part after "https://"
    text = url_pattern.sub(r'\1', text)
    
    # Remove '#' and '@'
    text = text.replace("#", "").replace("@", "")
    
    # Convert emojis to text and lowercase the entire text
    text = emoji.demojize(text).lower()
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words and punctuations in one step using list comprehension
    filtered_words = [word for word in words if word not in stop_words and word not in punctuation]
    
    # Stem words using list comprehension
    stemmed_words = [ps.stem(word) for word in filtered_words]
    
    # Join the processed words back into a single string
    return ' '.join(stemmed_words)


 
# ----------------------------------------------------------
    
def main():
    print("Working on vectorization....")
    df = read_CSV(p)
    df['tweet'] = df['tweet'].apply(preprocessing)
    df['label'] = df['label'].map({"real": 1, "fake": 0})

    prep_data(df)
    print("Done....")
    
     
 

if __name__=="__main__": 
    do_cmd()
    main() 