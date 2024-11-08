**CS 59000 – 05 Natural Language Processing Prof. Jonathan Russert** 

**Homework - #3** 

**Name: Mohamed Aarif Mohamed Sulaiman**  

**PFW ID: 900405565** 

**Methodology:** 

The  methodology  involved  in  this  project  is  crea ng  two  func ons  namely, train\_LR\_model and test\_LR\_model for sen ment analysis purpose. In addi on to that, one more func on is being employed called “extract\_features” func on which is used similar to the purpose of preprocessing of input text corpus. The code also uses various libraries ( which is used in the previous assignments as well) such as numpy, pandas, scikit-learn, nltk, etc., 

**Extract\_features:** 

Feature extrac on is an important step here which iden fies and extracts relevant informa on from raw data, aims to reduce dimensionality and improve the quality of data for the Logis c Regression algorithm to be fed into the model. The overall objec ve is to provide a quan ta ve analysis of the text corpus. 

1. **Length of text corpus**: Length of text in tokens can provide insights into the depth of expression. So, based on the length, longer texts might elaborate more on sen ments. 
1. **Total number of words**: Similar to length, word count provides informa on how stringer the sen ments are followed based on the volume of the words used. 
1. **Number  of  punctua on  count**:  Punctua on  marks  like  exclama on  depicts  how  the emo onal tone is expresses such as excitement, happiness or anger. Other punctua on marks like ques on mark indicates confusion or curiosity. 
1. **Total uppercase le ers**: The use of uppercase le ers can indicate the writer’s style and employing excessive might lean towards more polarized sen ments. 
1. **Total number of posi ve and nega ve sen ment words**: The frequency of posi ve and nega ve  words  can  be  a  direct  indicator  of  what  kind  of  a  sen ment  the  corpus  is showcasing. 
1. **Average word length**: Tokens using more longer, complex words might be used in more formal or intense contexts, poten ally correla ng with specific sen ments. 
7. **Nega on words:** Token using words like not, no, never might change the meaning of the sen ment. Hence those words are also counted. 

**train\_LR\_model:** 

The logis c regression model is trained using the extracted features. Feature Scaling is performed using StandardScarler() func on which standardizes the feature set to have a mean of 0 and standard devia on of 1, which will help in improved convergence and performance at the training stage. 

Here, in order to fine-tune the Logis c Regression model, we are adjus ng the parameters with GridSearchCV() func on. We can tweak the parameters by giving a set of values  such that model will try different set of combina ons. The parameters used are: 

- C: Inverse of regulariza on strength; smaller values specify stronger regulariza on. 
- Solver: Algorithm to use for op miza on. Different solvers have different capabili es. 
- Max\_iter: Maximum number of itera ons for the solver to converge. 
- Penalty: Type of regulariza on (L1 or L2). 

The above-men oned parameters will try to find the best set of parameters based on F1-score for the input and target variables in the provided model, i.e., Logis c model. These parameters can be used building the standard model and tes ng purpose. 

**test\_LR\_model:** 

Once the model is trained using best parameters, the model is tested using test dataset and it is evaluated using various parameters such as accuracy, precision, recall, F1 score, and a confusion matrix. These metrics provide a comprehensive view of the model's performance, allowing for insights into both the strengths and weaknesses of the predic ons. 

**Evalua on Metrics & Interpreta on:** 

**Accuracy: 0.6657 (or 66.57%)** 

The model correctly classifies approximately 66.57% of the  me in the test set. Accuracy is generally not considered for classifica on tasks since it can be misleading in cases of class imbalances. 

**Precision: 0.6725 (or 67.25%)** 

Precision measures the accuracy of posi ve predic ons, i.e., the propor on of true 

posi ves  among all predicted posi ves. (TP / (TP + FP)). Here, the model predicted 67.25% 

were actually posi ve. This indicates that model is effec ve at iden fying true posi ves but it is not efficient enough. 

**Recall: 0.6485 (or 64.85%)** 

Recall measures the completeness of posi ve predic ons, i.e., the propor on of true 

posi ves among all actual posi ve instances (TP / (TP + FN)). Here, the model correctly 

iden fied approximately 64.85%  which reflects the model’s ability to find all relevant cases. However, a value below 70% is not accurate enough. 

**F1 Score: 0.6597 (or 66%)** 

The F1 score is the harmonic mean of precision and recall, providing a balanced 

evalua on of both metrics. Here, the F1 score of 0.6597 suggests that while the model performs reasonably well, there is room for improvement. 

**Confusion Matrix:** [[366,167] , [189,343]] 

![](Aspose.Words.f92cf810-0be6-4c98-8612-d50a0030105f.001.jpeg)

- **True Posi ves (TP)**: 343 instances were correctly predicted as posi ve. 
- **True Nega ves (TN)**: 366 instances were correctly predicted as nega ve. 
- **False Posi ves (FP)**: 167 instances were incorrectly predicted as posi ve (i.e., they were actually nega ve). 
- **False Nega ves (FN)**: 189 instances were incorrectly predicted as nega ve (i.e., they were actually posi ve). 

**How did the model’s performance is increased?** 

Ini ally,  while  developing  the  logic  for  feature  extrac on,  only  five  features  were implemented such as **length of text corpus, Total number of words, Number of punctua on count,  Total  uppercase  le ers  and  Total  sen ment  words**  (not  separated  into  posi ve  or nega ve). When trying to train and test the model using these features, the precision and recall score were approximately around 55%. 

Next, the sen ment words are segregated into posi ve sen ment and nega ve sen ment words for the an cipa on of more accurate performance. But the scores increased merely ~2 to 3%. 

A er that, we have added one more feature “**Average word length**” which resulted in the final precision and recall scores. One more feature is also added “**Nega on words**” for checking any nega on words which might change the sen ment of the sentence. It didn’t change any results in precision or recall but a slight increase in True Posi ve and True Nega ve values. 

Hence, with the increase in feature scaling, the model’s performance is gradually increasing and if more steps are added, the model might have high accuracy. 

- If more advanced techniques like TF-IDF, n-gram, parts-of-speech tagging or Word2Vec libraries  are  used  for  feature  extrac on,  the  model’s  performance  might  have  been improved. 
- Considering  more  complex  models  like  Random  Forest,  Gradient  Boos ng  Machines, Neural Networks like LSTM or BERT can significantly improve efficiency. 
- Given  the  poten al  class  imbalances  between  true  posi ves  and  false  posi ves, techniques such as SMOTE (Synthe c Minority Over-sampling Technique) to generate synthe c samples for the minority class can help improve the model's performance. 

No problems were faced during programming or execu on. 

**Conclusion:** 

In  conclusion,  the  sen ment  analysis  and  classifica on  using  logis c  regression  provides  a structured and straight forward approach for text classifica on. However, exploring alterna ve methodologies by enhancing feature extrac on and using different models can lead to improved results. 
