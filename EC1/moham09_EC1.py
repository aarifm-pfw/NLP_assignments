## Name: Mohamed Aarif Mohamed Sulaiman
## PFW ID: 900405565

## EC #1
## Prof. Jonathan Russert

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_NB_model(path_to_train_file):
    data = pd.read_csv(path_to_train_file, sep='\t', names=['text', 'label'], header=0)
    model_count = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    model_tfidf = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Define parameters for GridSearchCV
    param_grid = {
        'vectorizer__max_features': [1000, 5000, None],  
        'classifier__alpha': [0.5, 1.0, 1.5]              
    }
    
    #Grid Search for CountVectorizer
    grid_search_count = GridSearchCV(model_count, param_grid, cv=5, scoring='accuracy')
    grid_search_count.fit(data['text'], data['label'])
    
    #Grid Search for TfidfVectorizer
    grid_search_tfidf = GridSearchCV(model_tfidf, param_grid, cv=5, scoring='accuracy')
    grid_search_tfidf.fit(data['text'], data['label'])

    return grid_search_count.best_estimator_, grid_search_tfidf.best_estimator_
    
def test_NB_model(path_to_test_file, NB_model):
    test_data = pd.read_csv(path_to_test_file, sep='\t', names=['text', 'label'], header=0)

    probabilities = NB_model.predict_proba(test_data['text']) 
    prob_positive = probabilities[:, 1] 
    prob_negative = probabilities[:, 0]  
    predictions = NB_model.predict(test_data['text'])

    output = test_data.copy()
    output['probability_positive'] = prob_positive
    output['probability_negative'] = prob_negative
    output['prediction'] = ['positive' if label == 1 else 'negative' for label in predictions]

    output.to_csv('test_predictions.tsv', sep='\t', index=False)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_data['label'], predictions)
    precision = precision_score(test_data['label'], predictions)
    recall = recall_score(test_data['label'], predictions)
    f1 = f1_score(test_data['label'], predictions)
    conf_matrix = confusion_matrix(test_data['label'], predictions)
    
    # Print the metrics
    print("\n ************************** \n")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    print(classification_report(test_data['label'], predictions))
    print("\n ************************** \n")

    #Plotting the confusion matrix
    class_names = ['Negative', 'Positive']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
model1 , model2 = train_NB_model('rotten_tomatoes_train.tsv')
print("\t \t Evaluation metrics for NB model when using Count Vectoriser\n")
test_NB_model('rotten_tomatoes_test.tsv', model1)
print("\t \t Evaluation metrics for NB model when using TF-IDF Vectoriser\n")
test_NB_model('rotten_tomatoes_test.tsv', model2)