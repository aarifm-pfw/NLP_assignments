**CS 59000 – 05 Natural Language Processing**

**Prof. Jonathan Russert**

**Homework - #07**

**Name: Mohamed Aarif Mohamed Sulaiman**

**PFW ID: 900405565**

**Introduction:**

In this homework we are using multiple large language models (LLMs) from the Hugging Face platform including **GPT-2, Meta's Llama-3.2-1B, Google's Gemma-2-2B, and Hugging Face's SmolLM2-1.7B-Instruct** models, using those pretrained language models and tokenizers from the Hugging Face library.

The objective is to generate text based on various input prompts using multiple decoding strategies, including greedy search, random sampling, beam search, and beam search with random sampling. The Hugging Face transformers library is employed to load the models and tokenize the input data, while the Hugging Face Hub facilitates the model loading and authentication process. The generated text is then saved for comparison.

**Model Selection:**

There are n number of Text generation models available in Hugging Face library. We selected the above-mentioned models as there smaller in size when trying to download the safetensors required for the text generation and tokenization. Also, these models don’t require higher computation powers or high-end resources. When others models such as EleutherAI/gpt-neox-20b and microsoft/Phi-3.5-mini-instruct are used to test, it either crashed out in Google Colab ( the environment which I normally use) due to lower System RAM or it consumes the entire disk space due to the huge chunks of model.safetensors to be downloaded. So, the right models which are fine-tuned to the entire extent of text corpus, light weight and highly efficient has to be selected which are those that are mentioned. Each of these models brings its own strengths to the table, so the choice should align with factors like the complexity of the task, resource availability, and the need for instruction-following behavior.

**Text Generation techniques:**

As per the requirement, the following four text generation techniques are used.

1. **Greedy search:** Greedy search is the simplest and most straightforward text generation method. The model generates the next token by always choosing the token with the highest probability at each step. This approach tends to generate the most likely (or "safe") sequence of tokens based on the model’s predictions, but it may lead to repetitive or suboptimal text in some cases.
2. **Random sampling:** In random sampling, the next token is chosen randomly according to the model’s probability distribution over possible next tokens. Instead of always picking the highest probability token, the model samples from the entire distribution, meaning even low-probability tokens can be selected.
3. **Beam Search (beams = 3):** Beam search is a more advanced and popular technique used for generating text that balances between exploration and exploitation. It keeps track of the top **N** candidate sequences (called "beams") at each generation step. With beams = 3, the model will maintain the top 3 most likely sequences at each step, expanding them and choosing the next token based on the highest probability from any of these beams. This method explores more potential sequences than greedy search, leading to more diverse and sometimes higher-quality outputs.
4. **Beam Search with Random Sampling (beams =3):** This method combines **beam search** with **random sampling** in order to introduce more randomness into the search process. It allows the model to explore less likely, but potentially more creative, sequences while still considering multiple candidate sequences at each step. With beams = 3, the model maintains 3 candidate sequences, but at each step, the token for each beam is selected through random sampling (rather than always selecting the most probable token).

**Result & Analysis:**

Each model generates responses to each prompt under different configurations of greedy, random sampling techniques. Once the responses are generated, those are saved in a text file and it is presented here.

**Note: The output file generated and its tabular form is attached along with this project as text file and excel file format respectively. Please kindly look into the excel file for detailed and seamless view as I was not able to paste the tabular form here properly.**

1. **For GPT-2 model:**

**Prompt 1:** "Once upon a time, there lived a King named Julius Caesar":

**Random sampling** output was the most effective one the sentences are not repeated and it adds more information from birth year to death year and other information on how he has conquered ,etc.,.

**Prompt 2:** "This movie was a good example of"

Both **Random Sampling and Beam Search with random sampling** output were more appealing to me than other decoding techniques. Since the prompt is one ending prompt, the output can be anything. But coherent, crisp and adding more details to the movie and not repeating the words were of utmost importance.

**Prompt 3:** "A farmer has 17 sheep, and all but 9 run away. How many are left?"

Now this prompt is a logical question just to test how the LLM models will work on these type of questions even though the pipelines are just text generation models.

No model gave the right answer but interestingly, Beam Search with random sampling tried to develop a context and story out of the input prompt which is **highly surprising and totally unexpected one**. Even random sampling did that but the former one did it much better.

**Prompt 4:** "The future of artificial intelligence is"

Once again, an open-ended prompt. **Beam Search** generates a focused and insightful continuation on the future of AI. It emphasizes the role of researchers and the upcoming advancements in AI, aligning well with the prompt’s topic. It avoids unnecessary tangents and stays relevant to the theme.

1. **For Meta- Llama2 model:**

**Prompt 1:** "Once upon a time, there lived a King named Julius Caesar"

**Beam Search** provides a more logical continuation of the story. It introduces more character depth, describing Julius Caesar as a man of many faults, with a love for power. This shows a thoughtful progression in the narrative.

**Prompt 2:** "This movie was a good example of"

**Beam Search** is highly effective here because the output is highly consistent and straightforward keeping up to the theme.

**Prompt 3:** "A farmer has 17 sheep, and all but 9 run away. How many are left?"

Every decoding techniques repeated the input prompt again and so the model can’t able to process logical questions clearly.

**Prompt 4:** "The future of artificial intelligence is"

**Greedy Search** works well here as it provides a clear and direct explanation of AI’s impact, emphasizing its revolutionary effect on how we live and work. The text stays focused and straightforward, which is suitable for this type of informative prompt.

1. **For Google- Gemma 2 model:**

**Prompt 1:** "Once upon a time, there lived a King named Julius Caesar"

**Random sampling** provided the most compelling narrative for the story about King Julius Caesar. It is also coherent and engaging the story.

**Prompt 2:** "This movie was a good example of"

**Surprisingly,** all the outputs generated for this prompts for each decoding techniques were really great. In those things, **Beam Search** produced a more detailed and balanced movie critique. It highlights the **engaging story** , in bold characters as the output is generated, and gives a deeper analysis of the plot's captivating nature. This is especially important for a movie critique, where a thoughtful review is desired. Other techniques like Greedy Search and Beam Search with random sampling were also upto the point.

**Prompt 3:** "A farmer has 17 sheep, and all but 9 run away. How many are left?"

Only this model solved the logical question ( so special appreciation to it). **Greedy search** is optimal here because it generates a simple, direct answer to the word problem, which is crucial for this type of question. The answer is clear and correct: "9". It doesn’t deviate or introduce unnecessary complexity.

**Prompt 4:** "The future of artificial intelligence is"

**Beam Search with Random Sampling** was the most effective for this prompt because it combines the structured coherence of Beam Search with the creativity of Random Sampling. This is especially useful for discussing a complex topic like AI, which requires both accurate information and a bit of variation in its explanation. The output contains multiple perspectives and a mix of both the potential and challenges of AI, making it more nuanced.

1. **For SmolLM model:**

**Prompt 1:** "Once upon a time, there lived a King named Julius Caesar"

**Beam Search with random sampling** provided a coherent and engaging story about King Julius Caesar.

**Prompt 2:** "This movie was a good example of"

**Beam Search with Random Sampling** produced a more nuanced and thoughtful critique. The movie review is reflective and detailed, covering aspects like entertainment value and character development.

**Prompt 3:** "A farmer has 17 sheep, and all but 9 run away. How many are left?"

**Surprisingly,** **Beam search** developed options like multiple choice questions and included the right answer in those options as well. This was totally **unexpected** one.

**Prompt 4:** "The future of artificial intelligence is"

**Beam Search** gave the most well-rounded and insightful response about the future of AI. It was structured and discussed both the potential and the challenges of AI, making it a balanced exploration of the topic.

Overall, Google Gemma 2 model outperforms all the other models and generates the best output for all kinds of different inputs like Logical question, open ended prompt, factual prompt, etc,.,

| **Prompt** | **Best Model** | **Best decoding technique** |
| --- | --- | --- |
| Once upon a time, there lived a King named Julius Caesar | Google Gemma2 | Random sampling |
| This movie was a good example of | Google Gemma2 | Beam Search |
| A farmer has 17 sheep, and all but 9 run away. How many are left? | Google Gemma2 | Greedy search |
| The future of artificial intelligence is | Google Gemma2 | Beam search with random sampling |

**Limitations of Comparisons:**

- Evaluating the quality of output from different models and decoding techniques can be highly subjective. What might seem like a "better" output for one person (e.g., creativity in the response) could be considered irrelevant or off-topic to another (e.g., precision or factual correctness).
- There is no single "correct" answer. Models can produce various valid outputs, and it's difficult to create a definitive **ground truth** to objectively compare responses.
- A **good comparison** can be hard to achieve when comparing models that have distinct strengths and weaknesses, especially if their internal mechanisms are not fully transparent.
- A given decoding technique’s performance might be influenced by the architecture and training specifics of the model. For example, **Beam Search with Random Sampling** might work well in one model due to its ability to balance structure and creativity, but fail in another model due to the model's inability to properly integrate randomness.
- The choice of test prompts may not always be representative of the full capabilities of a model. Certain models or techniques may excel in specific domains (e.g., storytelling or logical reasoning), but the evaluation may not fully capture their potential across other domains.

**Future Use:**

These text generation models and pipelines could be employed by extensive fine-tuning on specific tasks. This will strengthen the various models by being more coherent and varied.

**Conclusion:**

In conclusion, this homework we take a systematic comparison of text generation models from different organizations, namely GPT-2, Meta's Llama-3.2-1B, Google's Gemma-2-2B, and Hugging Face's SmolLM2-1.7B-Instruct. By utilizing various decoding strategies such as greedy search, random sampling, beam search, and beam search with random sampling, it explored how different models behave in response to identical prompts and generation strategies, offering insights into their unique strengths and weaknesses.
