import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from nltk.corpus import opinion_lexicon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

nltk.download('opinion_lexicon')

def extract_features(text):
    
    length = len(text) #1: Length of text
    num_words = len(text.split()) #2: Number of words
    num_punctuation = len(re.findall(r'[^\w\s]', text)) #3: Number of punctuation marks
    num_uppercase = sum(1 for c in text if c.isupper()) #4: Number of uppercase letters

    positive_words = set(opinion_lexicon.positive()) #5: Number of positive and negative sentiment words 
    negative_words = set(opinion_lexicon.negative())

    num_positive_words = sum(1 for word in text.split() if word.lower() in positive_words)
    num_negative_words = sum(1 for word in text.split() if word.lower() in negative_words)

    avg_word_length = sum(len(word) for word in text.split()) / num_words if num_words > 0 else 0 #6: Calculating Average word length
    
    negation_words = {"not", "no", "never", "n't"} #7: Presence of negation
    has_negation = any(word in text.lower().split() for word in negation_words)

    return [length, num_words, num_punctuation, num_uppercase, num_positive_words, num_negative_words, avg_word_length, has_negation]
    
def train_LR_model(path_to_train_file):
    data = pd.read_csv(path_to_train_file, sep='\t', names=['text', 'label'], header=0)
    
    X = np.array([extract_features(text) for text in data['text']])     # Extract features and labels
    y = data['label'].values

    scaler = StandardScaler() # Scale features
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'solver': ['liblinear', 'saga'],      # Solvers that support L1 or L2
        'max_iter': [100, 200, 300],          # Number of iterations
        'penalty': ['l1', 'l2']                # Regularization types
    }
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_scaled, y) 

    best_params = grid_search.best_params_

    best_model = LogisticRegression(**best_params)
    best_model.fit(X_scaled, y) 

    return best_model, scaler

def test_LR_model(path_to_test_file, LR_model, scaler):
    test_data = pd.read_csv(path_to_test_file , sep='\t', names=['text', 'label'], header=0 )

    X_test = np.array([extract_features(text) for text in test_data['text']])
    X_test_scaled = scaler.transform(X_test)

    probabilities = LR_model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
    predictions = LR_model.predict(X_test_scaled)  # Class predictions
    actual_labels = test_data['label'].values

    # Calculating metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, pos_label=1)  
    recall = recall_score(actual_labels, predictions, pos_label=1)
    f1 = f1_score(actual_labels, predictions, pos_label=1)
    conf_matrix = confusion_matrix(actual_labels, predictions)

    print("\n ************************** \n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print(classification_report(actual_labels, predictions))
    print("\n ************************** \n")

    #Plotting the confusion matrix
    class_names = ['Negative', 'Positive']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    output = test_data.copy()
    output['probability'] = probabilities
    output['prediction'] = ['positive' if p == 1 else 'negative' for p in predictions]
    output.to_csv('test_predictions.tsv', sep='\t', index=False)

warnings.filterwarnings("ignore")
model, scaler = train_LR_model('rotten_tomatoes_train.tsv')
test_LR_model('rotten_tomatoes_test.tsv', model, scaler)