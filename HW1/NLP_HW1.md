**CS 59000 – 05 Natural Language Processing Prof. Jonathan Russert** 

**Homework - #1** 

**Name: Mohamed Aarif Mohamed Sulaiman PFW ID: 900405565** 

**Summarize\_corpus func on:** 

Given in the original document and based on the lectures, a “token” is a word separated by whitespaces. It is a way of separa ng a text or a corpus into smaller units called tokens which are further used for upcoming feature engineering steps in NLP algorithms. 

On the other hand, “types” refer to unique number of tokens or categories or classes that tokens belong to. 

**Descrip on:** 

- A er impor ng the file and loading into the working space, the  regular expression (re) library is used to separate the corpus into tokens. It splits the corpus based on spaces and newlines. 
- Counts the total number of tokens in the corpus and total number of types using ‘Counter’ library which creates a frequency count of each unique token. 
- We print the numbers and sta s cs. 
- The func on then opens a file called "histogram.txt" in write mode. 
- It writes each token and its corresponding frequency to the file in a sorted order. Each line contains a token followed by its frequency. 

A er  inpu ng  dantesinferno.txt  file  into  summarize\_corpus  func on,  we  get  the  following output: 

Code: summarize\_corpus("dantesinferno.txt") 

O/p: Total number of tokens: 118876 

Total number of types: 21987 

Similarly, a er inpu ng waro heworlds.txt file, we get the results:

Code: summarize\_corpus("waro heworlds.txt") 

Total number of tokens: 63117 

Total number of types: 11414 

Finding the tokens for normalized texts of the inputs: 

Code: summarize\_corpus("normalized\_dantesinferno.txt") Total number of tokens: 115988 

Total number of types: 9801 

Code: summarize\_corpus("normalized\_waro heworlds.txt") Total number of tokens: 63025 

Total number of types: 6053 

So, we see that total number of tokens and total number of types are difference of having nearly one-fourth or even one-fi h of original token that are present. So, there are more words that are repeated and a smaller number of unique words. 

Also, the histogram files for the above two text files are also created which lists the frequency count of each and every tokens. 

**Normalize\_corpus func on:** 

The above summarize\_corpus func ons does tokeniza on without any preprocessing or feature engineering techniques required for the NLP algorithms to implement. This is useful for analyzing the text or corpus without changing its structure. 

Normalized tokeniza on is a type of preprocessing technique which makes the corpus clean by doing  further  transforma ons  and  func ons  such  as  conver ng  all  token  into  lowercase, removing  punctua ons  and  remove  digits  or  numbers.  This  results  in  more  accurate  and meaningful tokens for analysis and NLP tasks to perform easily. 

**Descrip on:** 

- The func on takes two input arguments, path of the original file and path of the output file where the normalized corpus will be saved. 
- There are six different func ons which does the normaliza on techniques here. All these are done using re (regular expression) library. 
- Firstly, we are removing the extra white spaces and replacing it with a single space. Addi onally, we’re also removing any leading or trailing white space with strip() func on. 
- Secondly, we’re removing any number digits and replacing them with empty string. 
- Next, we’re removing all the punctua ons like full stops, commas, exclama ons marks, etc.,  
- Conver ng all the remaining tokens into lowercase. 
- Once again, ‘replace\_mul ple\_spaces’ func ons replaces two or more  occurrences of mul ple white spaces. 
- Lastly, lemma za on is done which removes common plural word suffixes like -s, -ing , - ly, etc., 
- A er the en re process is done, the normalized text corpus is merged into one and the output is wri en into the text file that is provided in the loca on of the output file. 

An important point to be noted in here is that, these steps of normaliza on are to be done in a specified order and changing the order of the func ons will affect the output and subsequently it will affect the output of the func on.  

For example, if we put the “conver ng to lowercase” func on at the start, then certain uppercase le ers will be converted to lowercase before other processing (e.g., punctua on removal). In this case, the text becomes normalized in terms of case before handling spaces, digits, etc. Further, if you move it down any transforma ons that rely on case-sensi ve opera ons (like removing punctua on) would take place first, which could change the outcome for certain inputs. 

Snapshot  comparison  of  dantesinferno.txt  and  normalized\_dantesinferno.txt  (Randomly selec ng any six words in the histogram files) 



||In non- normalised Dante's Inferno text file |In Normalized Dante's Inferno text file |
| :- | :-: | :-: |
|Tokens |Occurences ||
|ambition |7 |13 |
|already |53 |69 |
|cheat |5 |10 |
|distinguish |10 |29 |
|flame |13 |38 |
|gutenberg |3 (Present in all Uppercase) |88 |

From the above table, we can observe that words like GUTENBERG are converted into lowercase and doing several other normaliza on processes, the result number is increased. 

Snapshot  comparison  of  waro heworlds.txt  and  normalized\_waro heworlds.txt  (Randomly selec ng any six words from the histogram files) 



||In non- normalised War of the Worlds text file |In Normalized War of the Worlds text file |
| :- | :-: | :-: |
|Tokens |Occurences ||
|business |3 |Not present due to normalization |
|fighting-machine |9 |Not present due to normalization |
|gone |20 |28 |
|have |221 |225 |
|mansions |2 |Not present due to normalization |
|present |3 |41 |

From the above table we can observe that tokens like business and mansions are not present due to lemma za on process and figh ng-machine is not present due to removal of punctua ons. Hence,  in  the  whole  file  the  tokens  and  types  will  be  reduced  which  is  seen  above  in summarize\_corpus 

In  conclusion,  summariza on  and  normaliza on  func on  which  has  been  developed preprocesses the text corpus into feature engineered for the NLP algorithm to perform efficiently. The histogram files of non-normalized files and normalized gives a clear differen a on between the tokens that were available and how it changes or gets lost while doing normaliza on. 
