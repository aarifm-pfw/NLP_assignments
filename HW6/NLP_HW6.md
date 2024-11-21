**CS 59000 – 05 Natural Language Processing**

**Prof. Jonathan Russert**

**Homework - #6**

**Name: Mohamed Aarif Mohamed Sulaiman**

**PFW ID: 900405565**

**Introduction:**

Sentiment analysis on financial data can significantly impact stock prices, trading strategies, and market predictions. In this homework, we are using several NLP models, both traditional and state-of-the-art to classify sentiment of financial test data into positive, negative and neutral. The models evaluated in this report include pre-trained transformer models for sentiment analysis, a zero-shot classification model, and a Logistic Regression model based on features extracted from text using DistilBERT.

The dataset used in this homework is “**Financial Phrasebank**”, a polar sentiment dataset of sentences from financial news. Although this dataset is inbuilt in the dataset library, it only has training set which has around 2000 rows. Hence, the original dataset is imported via csv format which consists of 4840 sentences from English language where financial news are categorized by sentiment. No explicit text preprocessing such as stemming, lemmatization, or removal of stop words was needed, as the pre-trained models handle tokenization and text normalization internally.

**Models Used:**

1. **FinancialBERT** Sentiment Analysis (ahmedrachid/FinancialBERT-Sentiment-Analysis)

A pre-trained BERT-based transformer model fine-tuned for sentiment analysis in financial contexts developed by Ahmed Rachid Hazourli

1. **DistilRoBERTa** Sentiment Analysis (mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)

A pre-trained DistilRoBERTa, a lighter version of the RoBERTa model, optimized for financial sentiment analysis developed by Manuel Romero

1. **Zero-Shot Classification** (tasksource/deberta-small-long-nli)

A zero-shot classification model based on DeBERTa where model allows classification without the need for fine-tuning on domain-specific data, making it useful when training data is not available.

The model was tested with predefined candidate labels: "positive", "neutral", and "negative" and it was developed by a team called tasksource.

1. **Feature Extraction with DistilBERT** and Logistic Regression

DistilBERT was used to extract fixed-size feature vectors from each piece of text. These features were used as input to a Logistic Regression classifier to predict sentiment.

**Methodology:**

- The text data was passed through each of the sentiment analysis models (FinancialBERT, DistilRoBERTa) using the transformers pipeline API.
- The models returned predictions that were extracted and converted into sentiment labels.
- The DeBERTa model was used for zero-shot classification. The model was provided with three candidate labels ("positive", "neutral", "negative") and predicted the most likely label for each instance in the test set.
- DistilBERT was used to extract features from the text, which were averaged into a fixed-size vector for each text instance. These vectors were then passed to a Logistic Regression classifier for sentiment prediction.
- The Logistic Regression model was trained on the extracted features and evaluated using the metrics.

**Results:**

| **Metrics values in %** | **FinancialBERT-Sentiment-Analysis** | **distilroberta-finetuned-financial-news-sentiment-analysis** | **Zero-Shot Classification with DeBERTa** | **Logistic Regression (DistilBERT Features)** |
| --- | --- | --- | --- | --- |
| Accuracy | 95.35 | 85.97 | 81.21 | 85.55 |
| Precision | 95.47 | 85.93 | 81.38 | 85.48 |
| Recall | 95.35 | 85.96 | 81.21 | 85.55 |
| F1 score | 95.36 | 85.88 | 80.58 | 85.44 |

**Analysis:**

FinancialBERT outperforms the other models in all metrics with **95.35% accuracy, 95.47% precision, 95.35% recall, and 95.36% F1-score**.

1. FinancialBERT is specifically **fine-tuned on financial text data**, which makes it well-suited to understand the context, terminology, and nuances of financial language. This leads to highly accurate sentiment predictions.
2. It also handles financial phrases more effectively than general-purpose models. It’s adept at differentiating between subtle sentiment differences in financial contexts, which might be challenging for models trained on general text.
3. DistilRoBERTa (**85.97% accuracy, 85.93% precision, 85.96% recall, and 85.88% F1-score**) is a lighter version of RoBERTa, which trades off some performance for efficiency (fewer parameters and faster inference). This reduced model size may limit its ability to capture some of the fine-grained relationships in financial text.
4. Although it is fine-tuned on financial news, the model may not have been exposed to the exact phrases or jargon found in the Financial PhraseBank dataset. This could lead to slightly lower performance compared to FinancialBERT, which may have been trained on more specific or varied financial data.s
5. DeBERTa is a zero-shot model (**81.21% accuracy, 81.38% precision, 81.21% recall, and 80.58% F1-score**), meaning it was not specifically trained on the task of sentiment analysis for financial texts. It leverages its ability to classify using pre-defined candidate labels, but it doesn’t have the domain-specific expertise that FinancialBERT and DistilRoBERTa possess.
6. The above model wasn’t fine-tuned for sentiment analysis in the financial context, which is why it might struggle to perform as well as models that were specifically designed for this task.
7. While Logistic Regression with features from DistilBERT is a reasonable approach, it doesn’t match the performance of the transformer-based models. The feature-based method doesn’t capture the complexity of sentiment analysis as well as fine-tuned neural networks (**85.55% accuracy, 85.48% precision, 85.55% recall, and 85.44% F1-score**).

**Limitations of Comparisons:**

The comparison of models using standard metrics like accuracy, precision, recall, and F1-score may not fully capture the nuances of model performance in real-world applications. These metrics are useful for classification tasks, but they may not fully account for real-world challenges such as model robustness, generalizability, or computational efficiency, which would also play a role in model selection for deployment.

**Future Work:**

In future research, combining the predictions from multiple models could improve accuracy. For example, an ensemble of FinancialBERT, DistilRoBERTa, and DeBERTa (zero-shot) might be able to capture different aspects of financial sentiment and lead to more robust predictions.

**Conclusion:**

In conclusion, **FinancialBERT** is the most suitable model for financial sentiment analysis, outperforming other models due to its domain-specific training. **DistilRoBERTa** and **Zero-shot classification** provides a good trade-off between speed and accuracy but lacks the edge that FinancialBERT has. Logistic Regression using features from a pre-trained model can work, but it doesn’t match the capabilities of fine-tuned transformer-based models for sentiment analysis tasks.
