## Name: Mohamed Aarif Mohamed Sulaiman
## PFW ID: 900405565

## HW $5
## Prof. Jonathan Russert

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from gensim.models import KeyedVectors
import gensim.downloader as api
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
from sklearn.exceptions import ConvergenceWarning

nltk.download('punkt')
nltk.download('stopwords')

glove_model = api.load("glove-wiki-gigaword-100") #Enable only when running the code for the first time.
glove_model.save('glove-wiki-gigaword-100') #Comment lines 23 and 24 when running the code for consecutive times since
#loading the GloVe embedding model takes time.
glove_model = KeyedVectors.load('glove-wiki-gigaword-100') # It is used to save the Glove model locally and refers from it.
label_encoder = LabelEncoder()

def load_dataset(data_file):
    df = pd.read_csv(data_file, sep='\t' , header=None, names=['text','label'])
    return df

def preprocess_corpus(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    stemmer = PorterStemmer()

    tokens = [stemmer.stem(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words and word not in punct]
    return ' '.join(tokens)
    
def get_average_embeddings(text_tokens, word_model):
  embeddings = [word_model[word] for word in text_tokens if word in word_model] #GloVe model to retrieve embeddings for words that are in the vocabulary
  if embeddings:
    return np.mean(embeddings, axis=0) # Compute the average vector for the current sentiment label
  else:
    return np.zeros(word_model.vector_size)
    
def train_MLP_model_average(path_to_train_file):
    df = load_dataset(path_to_train_file)
    X = []
    for text in df.iloc[:,0]:       
        text = preprocess_corpus(text)
        avg_embeddings = get_average_embeddings(text, glove_model)
        X.append(avg_embeddings)
    X = np.array(X)
    y = df.iloc[:, 1].values
    
    y_encoded = label_encoder.fit_transform(y)
    
    # Define and train MLP model
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='relu', max_iter=100, random_state=42, learning_rate_init=0.001, alpha=0.001)
    # param_grid = {
    #     'hidden_layer_sizes': [(100, 100, 100)],
    #     'activation': ['relu', 'tanh'],
    #     'solver': ['adam', 'sgd'],
    #     'alpha': [0.0001, 0.001, 0.01],
    #     'learning_rate': ['constant', 'adaptive'],
    #     'learning_rate_init': [0.001, 0.01]
    # }

    # grid_search = GridSearchCV(estimator=mlp_model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X, y)

    # best_model = grid_search.best_estimator_ # Retrieve the best model and parameters
    # best_params = grid_search.best_params_
    # mlp_model = best_model
    # print("Best parameters found: ", best_params)
    # print("Best cross-validation accuracy: ", grid_search.best_score_)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp_model.fit(X, y_encoded)
    
    return mlp_model
    
def get_position_weighted_embedding(tokens, word_vectors):
    embeddings = []
    weights = []
    
    for i, token in enumerate(tokens):
        if token in word_vectors:
            pos = i / len(tokens)  # Calculate position weight (higher for start and end)
            weight = 1 + np.exp(-((pos - 0.5) ** 2) / 0.1)  # Gaussian-like weighting
            
            embeddings.append(word_vectors[token] * weight)
            weights.append(weight)
    
    if not embeddings:
        return np.zeros(word_vectors.vector_size)
    
    return np.average(embeddings, axis=0, weights=weights)
    
def get_xor_features(tokens, word_vectors, feature_size=100):
    if not tokens:
        return np.zeros(feature_size)
    
    base_embedding = get_average_embeddings(tokens, word_vectors) # Get base embedding
    
    xor_features = np.zeros(feature_size) # Create XOR features
    for i in range(feature_size):
      xor_features[i] = float(int(base_embedding[i]) ^ int(base_embedding[(i + 1) % feature_size]))
    
    return xor_features
    
def train_MLP_model_student(path_to_train_file, method):
    df = load_dataset(path_to_train_file)
    texts = df['text'].apply(lambda x: preprocess_corpus(x))
    
    if method == 'xor':
        X = np.array([get_xor_features(tokens, glove_model) for tokens in texts])
    elif method == 'position':
        X = np.array([get_position_weighted_embedding(tokens, glove_model) for tokens in texts])
    else:
        raise ValueError(f"Unknown method: {method}")

    y = df.iloc[:, 1].values
    y_encoded = label_encoder.fit_transform(y)

    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='relu', max_iter=100, random_state=42, learning_rate_init=0.001, alpha=0.001)
    # param_grid = {
    #     'hidden_layer_sizes': [(100, 100, 100)],
    #     'activation': ['relu', 'tanh'],
    #     'solver': ['adam', 'sgd'],
    #     'alpha': [0.0001, 0.001, 0.01],
    #     'learning_rate': ['constant', 'adaptive'],
    #     'learning_rate_init': [0.001, 0.01]
    # }

    # grid_search = GridSearchCV(estimator=mlp_model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X, y)

    # best_model = grid_search.best_estimator_ # Retrieve the best model and parameters
    # best_params = grid_search.best_params_
    # mlp_model = best_model
    # print("Best parameters found: ", best_params)
    # print("Best cross-validation accuracy: ", grid_search.best_score_)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp_model.fit(X, y_encoded)
    
    return mlp_model
    
def test_MLP_model(path_to_test_file, MLP_model,input):
    # Load test data
    test_data = load_dataset(path_to_test_file)
    
    # Prepare embeddings based on input method
    X_test = []
    for text in test_data.iloc[:,0]:       
        text = preprocess_corpus(text)
        if input == 'average':
          avg_embeddings = get_average_embeddings(text, glove_model)
          X_test.append(avg_embeddings)

        elif input == 'xor':
          xor_features = get_xor_features(text, glove_model)
          X_test.append(xor_features)

        elif input == 'position':
          position_weighted_embedding = get_position_weighted_embedding(text, glove_model)
          X_test.append(position_weighted_embedding)

        else:
            raise ValueError(f"Unknown input method: {input}")
    X_test = np.array(X_test)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    probabilities = MLP_model.predict_proba(X_test)[:, 1]  # Probability of being positive
    predictions = MLP_model.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    # Append results to test data
    test_data['Probability_Positive'] = probabilities
    test_data['Prediction'] = predicted_labels
    evaluate_MLP_model(test_data)
    
def evaluate_MLP_model(test_data):
    true_labels = test_data.iloc[:, 1]  
    predicted_labels = test_data['Prediction']
    
    true_labels_encoded = label_encoder.transform(true_labels)
    predicted_labels_encoded = label_encoder.transform(predicted_labels)
    
    accuracy = accuracy_score(true_labels_encoded, predicted_labels_encoded)     # Calculate accuracy, precision, and confusion matrix
    print(f"Accuracy: {accuracy}")
    precision = precision_score(true_labels_encoded, predicted_labels_encoded)
    print(f"Precision: {precision}")
    recall = recall_score(true_labels_encoded, predicted_labels_encoded)
    print(f"Recall: {recall}")
    conf_matrix = confusion_matrix(true_labels_encoded, predicted_labels_encoded)
    print("Confusion Matrix")
    print(conf_matrix)
    

print("**********************\n")   
print("\t\t Average Embeddings Models")
average_model = train_MLP_model_average("rotten_tomatoes_train.tsv")
test_MLP_model("rotten_tomatoes_test.tsv", average_model, "average")
print("**********************\n") 
print("\t\t XOR model")
xor_model = train_MLP_model_student("rotten_tomatoes_train.tsv","xor")
test_MLP_model("rotten_tomatoes_test.tsv", xor_model, "xor")
print("**********************\n")  
print("\t\t Position WEighted Embeddings")
pos_model = train_MLP_model_student("rotten_tomatoes_train.tsv","position")
test_MLP_model("rotten_tomatoes_test.tsv", pos_model, "position")
print("**********************\n") 