## Name: Mohamed Aarif Mohamed Sulaiman
## PFW ID: 900405565

## HW #1
## Prof. Jonathan Russert

from collections import Counter
import re
import os

def summarize_corpus(file_location):
    with open(file_location, 'r', encoding='utf-8') as file:
        text = file.read()
    
    file_name = os.path.splitext(os.path.basename(file_location))[0]
    
    #finding tokens
    tokens = re.findall(r'\S+', text)
    total_tokens = len(tokens)
    
    #finding types
    token_counts = Counter(tokens)
    total_types = len(token_counts)
    
    print(f"Summarization details for {file_name}")
    print(f"Total number of tokens: {total_tokens}")
    print(f"Total number of types: {total_types}")
    
    histogram_file_name = f"histogram_{file_name}.txt"
    
    #saving into a text file for histogram
    with open(histogram_file_name, 'w', encoding='utf-8') as hist_file:
        for token in sorted(token_counts):
            hist_file.write(f"{token} {token_counts[token]}\n")
            
    print(f"Histogram saved to {histogram_file_name} \n")

summarize_corpus("dantesinferno.txt")
summarize_corpus("waroftheworlds.txt")

def normalize_corpus(file_location, outfile_location):
    
    def normalization(text):
        text = re.sub(r'\s+', ' ', text).strip()        #1. remove extra whitespaces
        text = re.sub(r'\d+', '', text)                 #2. remove digits
        text = re.sub(r'[^\w\s]', '', text)             #3. remove punctuations
        text = text.lower()                             #4. convert to lowercase
        text = re.sub(r'\s{2,}', ' ', text)             #5. replace multiple spaces
        text = re.sub(r'\b(\w+)(ing|ed|ly|es|s)\b', r'\1', text)  #6. remove common suffixes
        text = re.sub(r'\b(\w+ies)\b', r'\1y', text)    #7. handle words ending in -ies
        return text

    with open(file_location, 'r', encoding='utf-8') as file:
        text = file.read()
        
    normalized_text = normalization(text)    
    
    with open(outfile_location, 'w', encoding='utf-8') as outfile:
        outfile.write(normalized_text)
        
normalize_corpus("dantesinferno.txt","normalized_dantesinferno.txt")
normalize_corpus("waroftheworlds.txt","normalized_waroftheworlds.txt")

summarize_corpus("normalized_dantesinferno.txt")
summarize_corpus("normalized_waroftheworlds.txt")