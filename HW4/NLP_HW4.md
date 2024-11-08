**CS 59000 – 05 Natural Language Processing Prof. Jonathan Russert** 

**Homework - #4** 

**Name: Mohamed Aarif Mohamed Sulaiman**  

**PFW ID: 900405565** 

**Methodology:** 

The methodology involved in this project is creating four mandatory functons namely, create\_sentiment\_vectors,  predict\_sentiment,  create\_better\_vectors  and  predict\_better  for sentment analysis purpose. In addition to that, two more functions are being employed for loading the dataset and preprocessing the text corpus. The code also uses various libraries ( which is used in the previous assignments as well) such as numpy, pandas, scikit-learn, nltk, TF-IDF vectorizer and genism for including glove models,  etc., 

GloVe (Global Vectors for Word Representation) model is an unsupervised learning algorithm for obtaining vector representations for words. For this context of sentiment analysis for movie reviews using rotten tomatoes dataset, we are using dataset model that has been trained on a combination of Wikipedia and the Gigaword corpus, which contains a large amount of text data. This  corpus  is  designed  to  give  the  model  a  broad  understanding  of  language,  and  of dimensionality  100  for  faster  computing.  While  higher  dimensions  capture  more  semantic relationships it will require more memory and computational power. Hence, we are choosing a better Glove model. 

To incorporate these pre-trained embedding models into the working environment, the Gensim library’s  downloader  API  was  employed.  The  **api.load("glove-wiki-gigaword-100")**  function facilitates the efficient loading of the model. 

**create\_sentiment\_vectors(train\_file):** 

The primary purpose of this function is to generate average sentiment vectors for positive and negative sentiments based on the provided training dataset using on GloVe embeddings. 

- After loading the training dataset from load\_dataset function, we are preprocesing the text corpus to clean and tokenize each reviews present. 
- Then,  creating  a  list  of  valid  word  vectors  which  then  will  be  employed  for  each preprocessed review and splitting the words and retrieves their corresponding vectors from the GloVe model, but only if the word exists in the GloVe vocabulary. 
- Now, computing the mean of the valid vectors to create a single average vector for the sentiment  label  and  Updating  the  sentiment\_vectors  dictionary  with  the  computed average vector for the current sentiment label. 

**Predict\_sentiment(test\_file,sent\_vect):** 

After creating the sentiment vectors for the positive and negative words, we’re now predicting the sentiment of each review in the test dataset by comparing their average word vectors to the precomputed sentiment vectors and calculating the accuracy 

- Doing the preliminary steps of loading and preprocessing similar to previous function, we’re  computing  the  mean  of  the  valid  test  word  vectors  to  create  a  single  vector representation for the entire text. 
- Now,  employing  cosine  similarity  to  determine  which  sentiment  vector  (positive  or negative) is closest to the test word vector. 
- Compares predictions to actual labels with the highest similarity score and calculates accuracy. 

**create\_better\_vectors(train\_file)** 

Here, implementing an improved method for creating sentiment vectors using TF-IDF (Term Frequency – Inverse Document Frequency) and Singular Value Decomposition (SVD). 

- This function uses TF-IDF vectorization which transform the text data into TF-IDF vectors, and weigh the importance of words in the context of the entire dataset. 
- Next,  Reducing  the  dimensionality  of  the  TF-IDF  vectors  using  SVD,  which  helps  in capturing the most significant features. 
- Similar to create\_sentiment\_vectors, but now it uses the transformed TF-IDF vectors to compute average vectors for each sentiment label. 

**predict\_better(test\_file, sent\_vects):** 

Similar  to  **Predict\_sentiment(test\_file,sent\_vect)**  ,  this  function  predicts  the  sentiment  of reviews in a test dataset using the sentiment vectors created from the training data. It employs the fitted TF-IDF vectorizer and SVD model for transformation and similarity computation.  

**Result analysis:** 



|**Method** |**Accuracy** |
| - | - |
|Sentiment Vectors |63\.88% |
|Better Sentiment Vectors |70\.36% |

The results indicate a significant improvement in accuracy when using the "**Better Sentiment Vectors**" method (70.36%) compared to the initial "**Sentiment Vectors**" method (63.88%). This improvement can be attributed to several factors: 

- The **use of TF-IDF** allows for a more nuanced representation of text. It emphasizes the importance of words that are more relevant to specific documents while down weighting common words across the entire dataset. This results in a richer and more informative feature set. 
- The model is likely **less influenced by common or less informative words** (reduction of noise), leading to better feature selection for sentiment classification. 
- **SVD helps reduce the dimensionality** of the feature space while preserving the most significant components. This reduction can enhance model performance by simplifying the data structure, making it easier for the classifier to learn from the data without overfitting. 
- The **use of cosine similarity** for measuring the closeness between the text vector and sentiment vectors provides a more robust metric, particularly in high-dimensional spaces. It focuses on the angle between vectors rather than their magnitude, which can be beneficial when comparing text representations. 
- By calculating the **average sentiment vector** for positive and negative sentiments, the method  smooths  out  individual  variances  and  focuses  on  the  overall  sentiment characteristics. This leads to more reliable representations of sentiment. 

**Difficulties faced:** 

1. The use of Truncated SVD for dimensionality reduction posed several problems when trying to do svd.fit\_transform with a constant number, especially during predict\_better function  where  the  vector  dimensions  were  mismatching.  So,  to  set  the  dimension properly, according to the vector size and number of features, we’re finding the minimum value between 300 (highest dimension of a vector) and train dataset dimensions. 
1. Different  TF-IDF  vectorizer  were  used  initially  for  create\_better\_vectors()  and predict\_better() which once again resulted in the same vector dimensions mismatching and the accuracy resulted in 0. So, to correct that, we have used one vectorizer which is declared  globally.  Once  TF-IDF  vectorizer  is  initialized  and  set  in  the create\_better\_vectors() function, the same is used in the predict\_better() by calling the same variable globally. 

**Surprises:** 

Even with the addition of TF-IDF vectorizer method, the accuracy increased by around 6.5% to  get  70.36%  which  is  not the  optimal  increase with  the TF-IDF  vector  method.  The expected outcome was greater than 75% but with the obtained value, a few more inclusion steps in create\_better() and predict\_better() might provide better results. 

**Conclusion:** 

In conclusion, this code provides a comprehensive framework for performing sentiment analysis on  text  data,  utilizing  various  NLP  techniques  from  basic  GloVe  embeddings  to  more sophisticated TF-IDF and SVD improving the overall model accuracy and robustness. 
