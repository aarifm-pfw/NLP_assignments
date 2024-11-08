## Name: Mohamed Aarif Mohamed Sulaiman
## PFW ID: 900405565

## HW $4
## Prof. Jonathan Russert

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#nltk.download('punkt')
#nltk.download('stopwords')

#glove_model = api.load("glove-wiki-gigaword-100") #Enable only when running the code for the first time.
#glove_model.save('glove-wiki-gigaword-100') #Comment lines 23 and 24 when running the code for consecutive times since
#loading the GloVe embedding model takes time.
glove_model = KeyedVectors.load('glove-wiki-gigaword-100') # It is used to save the Glove model locally and refers from it.

# Declare the vectorizer and SVD globally
vectoriser = TfidfVectorizer()
svd = None  

def load_dataset(data_file):
    df = pd.read_csv(data_file, sep='\t' , header=None, names=['text','label'])
    df['label'] = df['label'].map({0:'negative' , 1:'positive'})
    return df

def preprocess_corpus(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    stemmer = PorterStemmer()

    tokens = [stemmer.stem(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words and word not in punct]
    return ' '.join(tokens)
    
def create_sentiment_vectors(train_file):
    df = load_dataset(train_file)
    sentiment_vectors = {'positive': [], 'negative': []}

    # Iterate over the two sentiment labels
    for label in ['positive', 'negative']:
        label_df = df[df['label'] == label]
        
        sentiment_words = [preprocess_corpus(text) for text in label_df['text']]
        
        #GloVe model to retrieve vectors for words that are in the vocabulary
        valid_vectors = [glove_model[word] for review_words in sentiment_words for word in review_words.split() if word in glove_model]
        
        if valid_vectors:
            # Compute the average vector for the current sentiment label
            label_vectors = np.mean(valid_vectors, axis=0)
            sentiment_vectors[label] = label_vectors

    return sentiment_vectors
    
    
def predict_sentiment(test_file, sent_vects):
    df = load_dataset(test_file)
    correct_predictions = 0

    for index, row in df.iterrows():
        words = preprocess_corpus(row['text'])
        word_vectors = [glove_model[word] for word in words.split() if word in glove_model]

        if word_vectors:
            text_vector = np.mean(word_vectors, axis=0)
        else:
            continue

        similarities = {label: cosine_similarity(text_vector.reshape(1, -1), vec.reshape(1, -1))[0][0] for label, vec in sent_vects.items()}
        predicted_label = max(similarities, key=similarities.get)

        if predicted_label == row['label']:
            correct_predictions += 1

    accuracy = correct_predictions / len(df)
    print(f'Accuracy for normal methods: {accuracy}')
    
sent_vects = create_sentiment_vectors('rotten_tomatoes_train.tsv')
predict_sentiment('rotten_tomatoes_test.tsv', sent_vects)

def create_better_vectors(train_file):
    global vectoriser, svd
    df = load_dataset(train_file)
    sentiment_vectors = {'positive': [], 'negative': []}

    train_text = df['text'].apply(preprocess_corpus)
    vectoriser.fit(train_text)
    
    # Transform the training text into TF-IDF vectors
    train_text_transformed = vectoriser.transform(train_text)
    
    # Reduce dimensionality with SVD 
    n_components = min(300, train_text_transformed.shape[1])
    svd = TruncatedSVD(n_components=n_components)
    train_text_reduced = svd.fit_transform(train_text_transformed)

    for label in ['positive', 'negative']:
        label_df = df[df['label'] == label]
        label_text = label_df['text'].apply(preprocess_corpus)
        label_text_transformed = vectoriser.transform(label_text)
        label_text_reduced = svd.transform(label_text_transformed)
        
        # Average the vectors for the label
        sentiment_vectors[label] = np.mean(label_text_reduced, axis=0)

    return sentiment_vectors

def predict_better(test_file, sent_vects):
    global vectoriser, svd
    df = load_dataset(test_file)
    correct_predictions = 0
    
    for index, row in df.iterrows():
        words = preprocess_corpus(row['text'])
        
        # Transform the words using the fitted vectoriser and SVD
        words_transformed = vectoriser.transform([words])
        words_vector = svd.transform(words_transformed)
        
        # Compute cosine similarity with sentiment vectors
        similarities = {label: cosine_similarity(words_vector.reshape(1, -1), vec.reshape(1, -1))[0][0] for label, vec in sent_vects.items()}
        
        predicted_label = max(similarities, key=similarities.get)

        # Check if the prediction matches the actual label
        if predicted_label == row['label']:
            correct_predictions += 1

    # Calculate and print accuracy
    accuracy = correct_predictions / len(df)
    print(f'Accuracy using TF-IDF and SVD: {accuracy}')
    
better_vects = create_better_vectors('rotten_tomatoes_train.tsv')
predict_better('rotten_tomatoes_test.tsv', better_vects)