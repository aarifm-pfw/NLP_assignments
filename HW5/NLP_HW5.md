**CS 59000 – 05 Natural Language Processing**

**Prof. Jonathan Russert**

**Homework - #5**

**Name: Mohamed Aarif Mohamed Sulaiman**

**PFW ID: 900405565**

**Methodology:**

The methodology involved in this project is creating three mandatory functions, namely train\_MLP\_model\_average, train\_MLP\_model\_student and test\_MLP\_model for classification purposes. In addition to the above functions, several others functions are also employed which are helpful for loading the dataset, preprocessing the text corpus and get the average embeddings, XOR embeddings and Positional embeddings. The code uses the same libraries that is used in the previous homework for loading GloVe model embeddings for word vector representation along with an MLP (Multi-Layer Perceptron) classifier.

**train\_MLP\_model\_average(train\_file):**

The main motivation of this function is to train a Multi-layer perceptron (MLP), a neural network classifier using the average word embedding which uses the simple mean of the word vectors that is given by **GloVe model**.  After loading the dataset into the environment, for each text in the given data

- Preprocessing the text corpus using the preprocess\_corpus function.
- Computing the average embeddings using the get\_average\_embeddings function which calculates the average word embeddings for a given text. Each embedding is a numerical vector representing the semantic meaning of the word.
- Stores into a list which is used as the input for training.

Once the above step is done, we are training the neural network to predict labels based on the word embeddings obtained. Important point to note here is we are creating an MLP with 

- Three hidden layers of 100 neurons each
- ReLU activation function
- Maximum 100 training iterations
- Learning rate of 0.001
- L2 regularization alpha of 0.001

**train\_MLP\_model\_student(train\_file):**

Similar to the above function, we are now training an MLP classifier using XOR and position weighted embeddings which will be selected based on the user input

1) **XOR:** Creates new features using XOR operations on embeddings dimensions. This XOR models helps in capturing the non-linear relationships between features that can help the model to learn complex patterns. The function get\_xor\_features() is called to do the following operation.

   It takes adjacent pairs of embedding dimensions, converts those values to integers and performs the bitwise XOR (^) operation. 

1) **Position-weighted embeddings:** Weights words based on their position in the text which gives more importance to words at the beginning and end. This approach is particularly useful in sentiment bearing words which often appear at the start or end of reviews. The function get\_position\_weighted\_embeddings() is called to do the following operation.

   It calculates the position-based weight using a Gaussian like function where ‘pos’ is normalized 0 to 1, creating a bell shaped curve which is cantered at 0.5. The weights are then calculated accordingly by applying the formula.

**test\_MLP\_model(test\_file, MLP\_model, input):**

This function loads test data, processes it accordingly to the specified input method and generates predictions. Similar to the training method all the above steps are carried out, based on the MLP model that is processed before. After that prediction is done using MLP classifier.

**Results & Analysis:**

|**Terms (all values in %)**|**Average Embedding Model**|**XOR model**|**Position Weighted Model**|
| :-: | :-: | :-: | :-: |
|**Accuracy**|55\.63|51\.22|51\.5|
|**Precision**|54\.9|50\.66|51\.18|
|**Recall**|63\.04|92\.45|64\.72|
|**Confusion Matrix**|[[257, 276], [197, 336]]|[[53,480], [40, 493]]|[[204,329] , [188, 345]]|

|**Confusion Matrix / Methods** |**Average Embeddings Model**|**XOR Model**|**Position embedding Model**|
| :-: | :-: | :-: | :-: |
|**True Positives (TP)**|336|493|345|
|**False Positives (FP)**|276|480|329|
|**True Negatives (TN)**|257|53|204|
|**False Negatives (FN)**|197|40|188|

The **Average Embedding Model** has the **highest accuracy at 55.63%**, while the XOR and Position Weighted models perform slightly worse, **both around 51%.** This suggests that, on a general basis, the Average Embedding Model is more effective at correctly classifying instances.

Precision is once again highest for the Average Embedding Model (54.9%), indicating that it has better specificity or fewer false positives compared to the other two. The XOR and Position Weighted models have lower precision, meaning they may have classified more non-positive samples as positive.

The XOR model stands out with a remarkably **high recall of 92.45%,** which means it is very effective at identifying all positive samples. However, the high recall may come at the expense of precision, as seen in its **lower precision score (50.66%).** This indicates the XOR model may be more biased towards predicting positives, resulting in more false positives.

The low performance is likely a combination of several factors:

1) The default **MLP parameters in scikit-learn** may not be optimized for this specific task. Parameters such as the learning rate, activation functions, number of epochs (or iterations), and optimizer choices can significantly affect performance. The hidden layer configuration might also be suboptimal for this data.
1) Also, the scikit-learn MLP is relatively basic compared to frameworks like TensorFlow or PyTorch. It lacks some advanced features, such as adaptive learning rates, advanced optimizers (e.g., AdamW, RMSprop), and support for GPU acceleration. These limitations may hinder performance on larger or more complex datasets.
1) The choice of activation function might not be ideal. Alternative activations like tanh or sigmoid might work better for certain types of data.
1) The selection of embeddings like **Averaging embeddings**, is a simplistic approach that can lead to loss of context or nuanced meaning, as it doesn’t capture word order or interactions between words. **XOR operations** on embedding dimensions are unconventional for NLP tasks, and such binary operations may introduce noise rather than useful information. **Position-weighted embeddings** may not fully capture sentence-level dependencies or complex word interactions. If only basic weights are applied to words at certain positions, the model may fail to grasp deeper semantic nuances.
1) MLPs are generally not designed to capture sequential dependencies, which are crucial in NLP. Models like RNNs, LSTMs, or transformers are better suited for capturing the contextual dependencies in text.

**Problems faced:**

In order to increase the accuracy, I tried to implement the best parameters using GridSearchCV component by trying different combinations like having hidden layer neuron counts to 200, activation functions like ‘tanh’ and others. The computation time significantly increased to more than 10 to 12 minutes. But that effort resulted in vain where the accuracy is remained the same to the obtained result. The code is still provided in the commented sections. (Please uncomment and test the models for GridSearchCV parameter)

**Comparisons with previous models:**

|**Terms (all values in %)**|**Average Embedding Model**|**XOR model**|**Position Weighted Model**|**Logistic Regression Model**|**Naïve Bayes Model**|**Sentiment Vector Model**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|**Accuracy**|55\.63|51\.22|51\.5|66\.57|79\.15|70\.31|
|**Precision**|54\.9|50\.66|51\.18|67\.25|80\.51|NA|
|**Recall**|63\.04|92\.45|64\.72|64\.47|76\.8|NA|

**Conclusion:**

In conclusion, the **Naïve Bayes Model** clearly outperforms all others, achieving the highest accuracy, precision, and recall. It appears well-suited for this dataset, likely due to text classification characteristics that align with Naïve Bayes’ assumptions. The **Logistic Regression Model and Sentiment Vector Model** perform reasonably well, especially in comparison to the MLP-based models. They offer good accuracy and balanced performance, indicating that simpler models with straightforward representations can effectively capture relevant features in this dataset. The **MLP-based models** (Average Embedding, XOR, and Position Weighted) perform poorly, particularly the XOR and Position Weighted Models. These models likely struggle because their embedding techniques are not well-suited for capturing semantic relationships in text data, and MLPs may not be the best architecture for text classification without advanced feature engineering or more complex embeddings.
