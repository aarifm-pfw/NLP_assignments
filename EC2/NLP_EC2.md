**CS 59000 – 05 Natural Language Processing**

**Prof. Jonathan Russert**

**Extra Credit - #02**

**Name: Mohamed Aarif Mohamed Sulaiman** 

**PFW ID: 900405565**

**Introduction:**

In this homework, the sentiment classification is done using DistilBERT language model on a rotten tomatoes dataset (used in our previous homeworks). The objective is to classify text reviews as either positive or negative, and evaluate the model's performance in terms of accuracy, precision, recall, and F1-score.

The model is imported and then fine-tuned using PyTorch and Hugging Face's transformers library. The training process leverages the GPU when available to accelerate computations.

**Model building and fine-tuning:**

DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than google-bert/bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark. The DistilBERT model is selected due to its computational efficiency and strong baseline performance. Themodel is built as follows:

- DistilBERT model is loaded from Hugging Face’s model hub. 
- **num\_labels=2** specifies that this is a binary classification task. The output layer is a classification head with two logits (one for each class).

The training arguments for hyperparameters such as learning rate, batch size and training epochs are set as follows:

- **output\_dir**: Directory where model checkpoints and logs will be saved.
- **evaluation\_strategy**: Determines when to evaluate the epochs. Here, it evaluate at the end of each epoch
- **learning\_rate**=2e-5. Specifies the initial learning rate for the optimizer. Small values like 2e-5 work well for fine-tuning transformer models.
- **per\_device\_train\_batch\_size / per\_device\_eval\_batch\_size**. It specifies batch size during training and evaluation. Larger batch sizes improve GPU utilization but require more memory.
- **num\_train\_epochs**=5 states the number of times the entire dataset is passed through the model during training.
- **weight\_decay**=0.01. Regularization to reduce overfitting by penalizing large weights.
- **warmup\_steps**=500. This step specifies the number of steps for learning rate warm-up. It gradually increases the learning rate to avoid sudden large updates at the start.
- **logging\_dir**="./logs". Directory to save logs for monitoring.
- **logging\_steps**=10: Log metrics every 10 steps.
- **save\_steps**=500: Save the model every 500 steps.
- **save\_total\_limit**=2: Retain only the two most recent model checkpoints.

The Trainer API from Hugging Face simplifies the training and evaluation process where it combines the model, datasets, tokenizer, and training arguments into a single object to automate training and evaluation. 

- **model=model**: Specifies the model to train (DistilBERT with the classification head).
- **args=training\_args**: Passes the training arguments defined earlier.
- **train\_dataset=train\_dataset**: The training dataset used to fine-tune the model. This is a custom dataset object built earlier.
- **eval\_dataset=test\_dataset**: The dataset used for evaluation.
- tokenizer=tokenizer: The tokenizer used to preprocess the text data. Ensures tokenized inputs match DistilBERT's requirements.
- **compute\_metrics**:  A custom function to compute evaluation metrics:
1. p: A prediction object containing:
1. p.label\_ids: True labels.
1. p.predictions: Predicted logits.
1. accuracy\_score: Calculates accuracy by comparing true labels (p.label\_ids) with predicted labels (p.predictions.argmax(-1)).

**Results:**

**Accuracy**: 0.8368

**Precision**: 0.8343

**Recall**: 0.8405

**F1 Score:** 0.8374

**Confusion Matrix:** [[444 ,89],[ 85, 448]]

|**Classification report**|**Precision**|**Recall**|**F1-score**|**Support**|
| :-: | :-: | :-: | :-: | :-: |
|Negative|0\.84|0\.83|0\.84|533|
|Positive|0\.83|0\.84|0\.84|533|
| |||||
|Accuracy| | |0\.84|1066|
|Macro Avg|0\.84|0\.84|0\.84|1066|
|Weighted Avg|0\.84|0\.84|0\.84|1066|

**Analysis:**

The model has a relatively balanced confusion matrix with a slightly higher number of false positives than false negatives, indicating the model is more prone to incorrectly classifying negative reviews as positive as **True Positives (448), False Positives (89), True Negatives (444) and False Negatives (85).**

- The model was trained for **5 epochs**. If the model has not converged, increasing the number of epochs could help which in turn would increase the training time significantly more; if overfitting is occurring, reducing the epochs or adding regularization techniques (e.g., dropout) might help.
- While the class distribution is roughly balanced, real-world sentiment datasets often suffer from subtle biases. For example, users may write longer, more detailed reviews for negative products, which could make negative samples harder to classify. Incorporating different types of **data augmentation** (such as back-translation, paraphrasing, or random sampling) can help mitigate these biases.

**Different approaches / Surprises:**

Before arriving at the final piece of code, several combinations and iterations of methods were tried in the code to get a better accuracy and precision.

1. Model layer freezing: Freezing the base layers of the model might help to increase the efficiency, but when I tried do the same it significantly reduced to accuracy and others metrics to 76%. So, experimented with unfreezing the top layers.
1. Preprocessing of data was initially formulated and tried. Similar to the above step, it also reduced the accuracy because it removed many significant important punctuations and uppercase words which might lead to incorrect results.

**Comparison with previous models:**

|**Classification Terms (all values in %)**|**Logistic Regression Model**|**Naïve Bayes Model**|**Sentiment Vector Model**|**Distil BERT model**|
| :-: | :-: | :-: | :-: | :-: |
|Accuracy|66\.57|79\.15|70\.31|83\.68|
|Precision|67\.25|80\.51|NA|83\.43|
|Recall|64\.47|76\.8|NA|84\.05|

**DistilBERT** uses token embeddings, leveraging the **transformer architecture's self-attention** mechanism, which allows it to capture long-range dependencies and complex semantic relationships in the data. This makes it significantly more powerful than traditional models like Naïve Bayes or Logistic Regression, which rely on simpler, fixed feature representations like bag-of-words or TF-IDF. The **Sentiment Vector Model** may use pre-trained word embeddings (in my case, I used GloVe embedding model), which can capture semantic relationships better than the simple models, but it still falls short of the transformer models in handling complex dependencies in the text**. Naïve Bayes** is a simple, probabilistic model that works well on text classification tasks with feature independence assumptions. It struggles to capture deep relationships in text, resulting in lower performance. **Logistic Regression** is also a relatively simple model and lacks the complexity to model the intricate relationships in textual data, hence its lower accuracy, precision, and recall compared.

**Conclusion:**

In conclusion, we fine-tuned **DistilBERT** model for binary sentiment classification, achieving an higher accuracy and other metrics. It also outperforms all other simple models like **Logistic Regression, Naïve Bayes and Sentiment Vector Model**. This highlights the power of deep learning and transformer models for text classification tasks like sentiment analysis, especially in comparison to traditional machine learning methods for NLP tasks.
