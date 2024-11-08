## Name: Mohamed Aarif Mohamed Sulaiman
## PFW ID: 900405565

## HW #2
## Prof. Jonathan Russert

#Importing the necessary libraries
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt_tab')

#Normalizatin technique used in the previous assignment
def normalization(text):
    text = re.sub(r'\s+', ' ', text).strip()        #1. remove extra whitespaces
    text = re.sub(r'\d+', '', text)                 #2. remove digits
    text = re.sub(r'[^\w\s]', '', text)             #3. remove punctuations
    text = text.lower()                             #4. convert to lowercase
    text = re.sub(r'\s{2,}', ' ', text)             #5. replace multiple spaces
    text = re.sub(r'\b(\w+)(ing|ed|ly|es)\b', r'\1', text)  #6. remove common suffixes
    text = re.sub(r'\b(\w+ies)\b', r'\1y', text)    #7. handle words ending in -ies
    return text

#Train function
def train_LM(path_to_train_file):
    
    with open(path_to_train_file, 'r', encoding='utf-8') as f:
        text = f.read()    
    
    text = normalization(text)
    tokenized_text = word_tokenize(text)
    
    #Trigram model calculation
    train_data , padded_sents = padded_everygram_pipeline(3,[tokenized_text])
    model = MLE(3)  # Trigram model
    model.fit(train_data, padded_sents)
    
    return model
    
def LM_generate(LM_model, prompt):
    #Converting input prompt to lowercase
    prompt_tokens = word_tokenize(prompt.lower())
    
    #Generating next 15 tokens based on the input prompt
    generated_tokens = LM_model.generate(15,text_seed=prompt_tokens)
    output = ' '.join(prompt_tokens + generated_tokens).replace('<s>', '').strip()
    return output

model_war = train_LM('waroftheworlds.txt')
model_inferno = train_LM('dantesinferno.txt')

prompts = ['I am glad you' , "it hasn't come to my attention that" , 'the monsters  attacked by' ,
'I can never imagine a use cause for' , 'my 2 important' , 'I hvae ']

for prompt in prompts:
    output_war = LM_generate(model_war, prompt)
    print("************* War of the Worlds ******************** \n")
    print(f"Prompt: {prompt}\nOutput for War of the Worlds: {output_war}\n")
    print("************** Dante's Inferno********************* \n")
    output_inferno = LM_generate(model_inferno, prompt)
    print(f"Prompt: {prompt}\nOutput for Dante's Inferno: {output_inferno}\n")