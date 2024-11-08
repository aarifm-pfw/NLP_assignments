**CS 59000 – 05 Natural Language Processing Prof. Jonathan Russert** 

**Homework - #2** 

**Name: Mohamed Aarif Mohamed Sulaiman** 

**PFW ID: 900405565** 

**Methodology:** 

The  methodology  involved  in  this  project  is  crea ng  two  main  func ons  train\_LM  and LM\_generate which does the preprocessing, tokeniza on, trigram genera on, training the model, building the model and tes ng the model. All these done using NLTK library present in python. It can also be done spaCy , tensorflow and keras library which increases the space capacity and efficiency of the model performance. 

**Train\_LM(path\_to\_train\_file)** 

This func on implemented trains the language model using the provided text corpus file.  **Descrip on:** 

- A er  impor ng  the  file,  the  normaliza on  func on  (  which  is  developed  for  the previous assignment is used over here) is called to preprocess / clean the text corpus. 
- The normalized text is tokenized into words using word\_tokenize library. 
- Trigrams are created from the tokenized text and a set of unique tokens are created to form the vocabulary. 
- Once these feature correc on steps are done, a maximum likelihood es ma on (MLE) trigram model is instan ated and the model is trained using the generated trigrams and vocabulary. 

**LM\_generate(LM\_model,prompt)** 

This func on is implemented to generate text using a language model. This process makes the model to create contextually relevant text based on the input, which is the basic fundamentals for many chatbots (including ChatGPT!!!)   

**Descrip on:** 

- Instead of doing the en re normaliza on steps, we are changing the input prompt to lowercase to ensure uniformity in tokeniza on, and ge ng the individual tokens using the word\_tokenize library 
- In  LM\_model.generate(),  the  argument  ‘15’  represents  the  number  of  tokens  to generate. If we want to increase the generated token count, we can add an argument and keep it dynamic to change it accordingly. 
- The generate method returns a list of tokens that the model predicts to follow the prompt. 
- The ‘join’ method combines the list of input prompt and generated token into a single string. The ‘replace’ method removes any instances of special token <s> from the output string. And the strip() method removes any trailing whitespaces. 

**Problems faced:** 

Two main problems I faced while coding this homework is : 

1. The availability of ‘punkt’. Even though the nltk.download(‘punkt’) is specified and the command is successfully run, it repeatedly showed “LookUpError” sta ng the tokenize and punkt folder is not found. When I tried downloading it to a specific loca on  and calling the model from the same loca on, the same error repeated again several  mes.  The folders are physically present in the system but the code IDE environment wasn’t able to detect it.  

   **Solu on:**  I  have  added  ‘punkt\_tab’  and  the  same  is  downloaded  and  used  for   word\_tokenize. 

2. Similar to the above problem, padded\_everygram\_pipeline library couldn’t able   import into the working environment. A er checking the same program in online jupyter notebook compiler, the program is working fine. 

   **Solu on:** A er upda ng the nltk library to latest version, the error was not occurring. 

**Observa ons and outputs:** 

The following prompts are given as input:  

- 'I am glad you'  
- "it hasn't come to my a en on that"  
- 'the monsters  a acked by'  
- 'I can never imagine a use cause for'  
- 'my 2 important'  
- 'I hvae ' 

![](Aspose.Words.50669548-d957-4bf9-bad8-d31b78c372c6.001.jpeg)

![](Aspose.Words.50669548-d957-4bf9-bad8-d31b78c372c6.002.jpeg)

The observa on for the output is done as follows: 



<table><tr><th colspan="1" rowspan="2"><b>Prompt</b> </th><th colspan="4" valign="top"><b>Observation / Inference</b> </th></tr>
<tr><td colspan="1"><b>War of the Worlds</b> </td><td colspan="1"><b>Result Opinion</b> </td><td colspan="1"><b>Dante's Inferno</b> </td><td colspan="1"><b>Result Opinion</b> </td></tr>
<tr><td colspan="1">I am glad you </td><td colspan="1">disjointed referencing “fields and lands” </td><td colspan="1">Not likely to be in document </td><td colspan="1">“pack of dogs” and a journey </td><td colspan="1">Fits the document theme </td></tr>
<tr><td colspan="1">"it hasn't come to my attention that"  </td><td colspan="1">Martian heel and scientific </td><td colspan="1">Kind of fits the theme </td><td colspan="1">Play before the hair </td><td colspan="1">Fits the document theme </td></tr>
</table>



|<p>the monsters  attacked </p><p>by </p>|monstrous and alien nature of the attackers |Partly fits the theme |reflects on tyrants and rebellion |Fits the document theme |
| :- | :-: | :-: | :-: | :-: |
|I can never imagine a use cause for  |references the moon and nature |Partly fits the theme |themes of conflict and spiritual consequence |Highly fits the document |
|my 2 important  |mention of “tax exempt status” and “invisible” |Not likely to be in document |includes an allusion to enterprise |Fits the document theme |
|I hvae |mention of mind pamphlet and  |Not likely to be in document |mention of "outstride" and "ground" |Highly fits the document |

**Note:** The prompt “I hvae” is specifically put to test how the model reacts for spelling mistaken words. For this input even Inferno text gives handles it and matches the context. Also, “my 2 important” prompt is put for checking how the model reacts to numerical input. For that, “tax status” is included to reference about quan ty. 

**Important Inferences:** 

- The  outputs  from  “**Dante’s  Inferno”**  generally  maintains  thema c  coherence referencing core ideas of the text like absolu on, moral and philosophical concepts whereas **“War of the Worlds”** document is vague and not upholding the concepts of the novel. 
- Also, the narra ve style for the Dante’s Inferno is s cking towards its content style whereas outputs from War of the Worlds some mes result in awkward phrasing or less relevant content. 
- The discrepancies in coherence and relevance in "War of the Worlds" outputs may indicate a need for a larger and more diverse training dataset or addi onal fine-tuning on specific themes or styles within the text. 

**Conclusion:** 

In conclusion, both language models seem to have trained and tried to replicate the style and language coherence of the respec ve text documents with Dante’s inferno highly matching and War of the Worlds matching par ally. 
