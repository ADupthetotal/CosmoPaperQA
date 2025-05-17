# CosmoPaperQA

## Features
CosmoPaperQA is a dataset designed to test AI RAG (Retrieval Augmented Generation) for cosmology research applications.  A current bottleneck in research into utilising AI in cosmological reseach is the lack of a specific cosmology research dataset.

This repository also includes a "proof of concept" algorithm that is designed to perform automated evaluation of answers given by RAG agents when answering complex research retrieval questions, like those which are included in the the CosmoPaperQA dataset. This new evaluation algorithm is called Embed_AI. 

## Description
### OpenAI evaluator agent
The OpenAI evaluator agent takes in the generated answer, the "correct" or "ideal" answer and the original question as input. It is then asked if the two answers convey the same point, given the context of the question. The possible outputs are "Same", "Similar" or "Different".

### Embed_AI evaluation
Embed_AI utilises both the custom OpenAI evaluator agent and cosine similarities of vector embeddings between extracted key phrases of the generated and "correct" answers. It uses both the evaluator agent and the cosine similarities to give a "correct" or "incorrect" evaluation of the generated answers.

### PaperQA2 RAG
The PaperQA2 RAG takes in a question and documents and uses the information in the documents to answer the question. The code involves gathering and ranking evidence/chunks from source papers, based on their relevance to answering the given question, and then generating the answer to the question based on the top ranked chunks. The [PaperQA2](https://github.com/Future-House/paper-qa) code does also allow for citation traversal, using Semantic Scholar or CrossRef, but it was not implemented in this code or used to generate the given results.

### OpenAI RAG
The OpenAI RAG also takes in a question and documents and uses the information in the documents to answer the question. The code follows the same RAG methodology as the PaperQA2 algorithm. The main difference between the two is that the PaperQA2 RAG uses cosine similarities to rank the chunks from the papers, while OpenAI RAG uses semantic search to rank the chunks.

### Simple OpenAI RAG
This RAG, instead of ranking chunks explicity in the program code and using those to generate the answer, uses the native OpenAI file search tool to answer the question in one step. Again, semantic search is used to find the relevant pieces of information.

### Mistral RAG
First, the Mistral RAG agent uses the Mistral OCR to convert the source papers into a markdown format. Then, Mistral RAG uses document chunks' text embeddings to create a vector store of the source papers. Cosine similarity between the chunks and the question is used to get the relevant information/chunks to answer the question. Finally, the Mistral AI uses the extracted chunks to generate the answer to the original question.

## Implementing the Codebase
The python code in the "functions" directory uses the InspectAI framework to do performance evaluation using the CosmoPaperQA dataset and the Embed_AI evaluation algorithm. 

Either the OpenAI RAG framework, the Mistral RAG framework or the [PaperQA2](https://github.com/Future-House/paper-qa) RAG framework can be used to generate answers to the questions in CosmoPaperQA. 

Use the functions contained in OpenAI_RAG_Agent/Mistral_RAG_agent/PaperQA2_RAG_agent, Embed_AI and Inspect_AI_framework and then run the line `inspect_ai_eval(rag_agent, eval_agent, embedding_answers)` to implement the RAG answer generation and performance evaluation of the CosmoPaperQA dataset.

Ensure that the following libraries are installed before running the python scripts: 
[os](https://docs.python.org/3/library/os.html), [openai](https://pypi.org/project/openai/), [pydantic](https://pypi.org/project/pydantic/), [numpy](https://pypi.org/project/numpy/), [re](https://docs.python.org/3/library/re.html), [rake_nltk](https://pypi.org/project/rake-nltk/), [pylatexenc.latex2text](https://pypi.org/project/pylatexenc/), [nltk](https://www.nltk.org/), [json](https://docs.python.org/3/library/json.html), [typing](https://docs.python.org/3/library/typing.html) and [inspect_ai](https://inspect.aisi.org.uk/). PaperQA2_RAG_agent additionally requires [paperqa](https://github.com/Future-House/paper-qa).

## Edited LitQA2 Dataset
LitQA2_edit is an edited version of [LitQA2](https://github.com/Future-House/LAB-Bench/blob/main/LitQA2/litqa-v2-public.jsonl) used to compare CosmoPaperQA to other datasets designed for performance evaluation of AI agents in scientific research.

## Summary of Results
The errors quoted here are calculated using the standard error (SE), assuming a binomial distribution to the correctness and accuracy statistics. Correctness is the percentage of generated answers that were evaluated to be correct and accuracy is the percentage of the given automated evaluation results that agree with the human evaluations. The OpenAI GPT-4o-mini model was used to power all of the OpenAI agents used to generate the following results. The vector embeddings for the Embed_AI algorithm used the text-embedding-ada-002 model. The Mistral LLM model used to power the RAG agent was mistral-medium-latest and the Mistral text embedding model used to faciliate the Mistral RAG agent was mistral-embed.

|                                    | Correctness using OpenAI evaluator agent(%)| Correctness using Embed_AI evaluation(%)| Correctness using human evaluation(%)|
|------------------------------------|--------------------------------------------|-----------------------------------------|--------------------------------------| 
|        CosmoPaperQA with OpenAI RAG|                                 82&plusmn;4|                              79&plusmn;4|                           71&plusmn;4|
| CosmoPaperQA with Simple OpenAI RAG|                                 85&plusmn;3|                              82&plusmn;4|                           74&plusmn;4|
|      CosmoPaperQA with PaperQA2 RAG|                                 91&plusmn;3|                              87&plusmn;3|                           75&plusmn;4|
|       ComsoPaperQA with Mistral RAG|                                 73&plusmn;4|                              65&plusmn;5|                           47&plusmn;5|
|         LitQA2_edit with OpenAI RAG|                                 85&plusmn;4|                              75&plusmn;4|                           79&plusmn;4|
|  LitQA2_edit with Simple OpenAI RAG|                                 79&plusmn;4|                              72&plusmn;4|                           79&plusmn;4|
|       LitQA2_edit with PaperQA2 RAG|                                 80&plusmn;4|                              73&plusmn;4|                           87&plusmn;3|
|        LitQA2_edit with Mistral RAG|                                 78&plusmn;4|                              71&plusmn;4|                           66&plusmn;5|

|                                    | Accuracy of Embed_AI evaluation(%)| Accuracy of OpenAI evaluator agent(%)|
|------------------------------------|-----------------------------------|--------------------------------------|
|        CosmoPaperQA with OpenAI RAG|                        71&plusmn;4|                           76&plusmn;4|
| CosmoPaperQA with Simple OpenAI RAG|                        86&plusmn;3|                           87&plusmn;3|
|      CosmoPaperQA with PaperQA2 RAG|                        81&plusmn;4|                           82&plusmn;4|
|       ComsoPaperQA with Mistral RAG|                        73&plusmn;4|                           65&plusmn;5|
|         LitQA2_edit with OpenAI RAG|                        89&plusmn;3|                           89&plusmn;3|
|  LitQA2_edit with Simple OpenAI RAG|                        83&plusmn;4|                           86&plusmn;3|
|       LitQA2_edit with PaperQA2 RAG|                        83&plusmn;4|                           88&plusmn;3|
|        LitQA2_edit with Mistral RAG|                        88&plusmn;3|                           85&plusmn;4|

Getting these results cost around £25, with the PaperQA RAG runs making up the majority of that cost.

The following results used the text-embedding-3-large model for the vector embeddings for the Embed_AI algorithm. Also, as the OpenAI models used for these results are significantly more costly, a random sample of half of the questions from the respective datasets was used to gather these results. The o3-mini model used a reasoning_effort parameter of "medium". For each run in the following results, the OpenAI model used to power the evaluator agent was the same as the one used to power the corresponding RAG agent.

|                                              | Correctness using OpenAI evaluator agent(%)| Correctness using Embed_AI evaluation(%)| Correctness using human evaluation(%)|
|----------------------------------------------|--------------------------------------------|-----------------------------------------|--------------------------------------| 
|         CosmoPaperQA with OpenAI RAG (gpt-4o)|                                 88&plusmn;5|                              86&plusmn;5|                           80&plusmn;6|
|        CosmoPaperQA with OpenAI RAG (o3-mini)|                                 86&plusmn;5|                              82&plusmn;5|                           84&plusmn;5|
|  CosmoPaperQA with Simple OpenAI RAG (gpt-4o)|                                 84&plusmn;5|                              82&plusmn;5|                           76&plusmn;6|
| CosmoPaperQA with Simple OpenAI RAG (o3-mini)|                                 90&plusmn;4|                              86&plusmn;5|                           86&plusmn;5|

|                                              | Accuracy using Embed_AI evaluation(%)| Accuracy using OpenAI evaluator agent(%)|
|----------------------------------------------|--------------------------------------|-----------------------------------------|
|         CosmoPaperQA with OpenAI RAG (gpt-4o)|                           90&plusmn;4|                              88&plusmn;5|
|        CosmoPaperQA with OpenAI RAG (o3-mini)|                           87&plusmn;5|                              87&plusmn;5|
|  CosmoPaperQA with Simple OpenAI RAG (gpt-4o)|                           90&plusmn;4|                              92&plusmn;4|
| CosmoPaperQA with Simple OpenAI RAG (o3-mini)|                           81&plusmn;5|                              82&plusmn;5|

Getting these results cost around £20, giving a total of around £45.

## Key Findings
Even though the Embed_AI algorithm's correctness here is still not in agreement with human evaluation, the accuracy of the Embed_AI algorithm is greater than or equal to the custom OpenAI evaluator agent alone, across nearly all of the data. The one notable exception is the PaperQA2 RAG agent answering the questions from the LitQA2_edit dataset, but even that isn't statistically significant.

The Embed_AI algorithm is unrefined and there is room for much improvement. As can be seen in the table above, even with this unrefined implementation, Embed_AI does not perform significantly worse than the OpenAI evaluator agent alone. This is somewhat impressive as most of the answers have a relatively high correctness and both the embedding_answers function and the OpenAI evaluator agent have to say that the generated and ideal answers are similar for the Embed_AI algorithm to return "CORRECT". Therefore, if the embedding_answers function had zero utility in performance evaluation, one would expect the Embed_AI accuracy to be significantly lower than the OpenAI evaluator agent accuracy, but this is not observed.

From the results gathered, the PaperQA2, the OpenAI RAG agent and the Simple OpenAI RAG agent answers were observed to have similar levels of correctness, measured against human evaluation and with the same datasets, with differences in the mean correctnesses of up to around 4%. The one notable exception was, again, the PaperQA2 RAG agent answering the questions from the LitQA2_edit dataset.

This suggests that the differences in methodology between the PaperQA2 RAG agent, the OpenAI RAG agent and the Simple OpenAI RAG agent are negligible. The PaperQA2 RAG agent answering the questions from the LitQA2_edit dataset is an anomalous datapoint, for both the accuracy and correctness. The original [LitQA2](https://github.com/Future-House/LAB-Bench/blob/main/LitQA2/litqa-v2-public.jsonl) dataset questions that I used to construct the LitQA2_edit dataset contained some questions that were used to optimise the performance of PaperQA2 and some "unseen" questions that were used to ensure that PaperQA2 did not overfit to the optimisations. This particular datapoint being anomalous in the correctness measure seems to suggest that some overfitting in PaperQA2 to the LitQA2 dataset is present. This is not enough evidence to definitively make that conclusion, but this something to keep in mind when interpreting these results.

The runs involving the more costly OpenAI models did indeed demonstrate that there was a measurable increase in performance between the small gpt-4o-mini model and the larger, and more costly, gpt-4o and o3-mini models. This demonstrates that the CosmoPaperQA dataset really provides a measure of agentic performance with answering cosmology research questions. However, one interesting thing to note is that the o3-mini model provides a measurable increase in performance compared to the gpt-4o model for answering the CosmologyQA questions, despite gpt-4o being significantly more costly than o3-mini. This appears to validate OpenAI's claims that the o3-mini model is ["delivering exceptional STEM capabilities—with particular strength in science, math, and coding"](https://openai.com/index/openai-o3-mini/). 

Finally, it is worth noting that using the text-embedding-3-large model for the vector embeddings, as compared to the text-embedding-ada-002 model, did not give a significant increase in performance to the Embed_AI algorithm, compared to the OpenAI evaluator agent alone.

## Acknowledgements
Thanks to [Dr Boris Bolliet](https://github.com/borisbolliet) for advice and support during the development of this project.
