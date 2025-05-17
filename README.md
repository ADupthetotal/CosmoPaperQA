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
The errors quoted here are calculated using the standard error (SE), assuming a binomial distribution to the correctness and accuracy statistics.

Correctness is the percentage of generated answers that were evaluated to be correct.

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


Accuracy is the percentage of the given automated evaluation results that agree with the human evaluations.

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

The OpenAI GPT-4o-mini LLM model was used to power all of the OpenAI agents used to generate the above results. The text embeddings for OpenAI used the text-embedding-ada-002 model. The Mistral LLM model used to power the RAG agent was mistral-medium-latest and the Mistral text embedding model used to faciliate the Mistral RAG agent was mistral-embed. In total, getting these results cost around £25, with the PaperQA RAG runs making up the majority of that cost.

## Key Findings
Even though the Embed_AI algorithm's correctness here is still not in agreement with human evaluation, the accuracy of the Embed_AI algorithm is greater than or equal to the custom OpenAI evaluator agent alone, across nearly all of the data. The one notable exception is the PaperQA2 RAG agent answering the questions from the LitQA2_edit dataset, but even that isn't statisitcally significant.

Also, this particular implementation is unrefined and there is room for much improvement. As can be seen in the table above, even with this unrefined implementation, there is a measurable improvement in accuracy from the custom OpenAI evaluator to the Embed_AI algorithm across some of the data. In all cases, the differences between the two methods is not statistically significant. Therefore, Eval_AI does not perform significantly worse than the OpenAI evaluator agent alone, which is somewhat impressive as most of the answers have a relatively high correctness and both the embedding_answers function and OpenAI evaluator have to say that the generated and ideal answers are similar for the Embed_AI algorithm to return "CORRECT".

From the results gathered, the PaperQA2, the OpenAI RAG agent and the Simple OpenAI RAG agent answers were observed to have similar levels of correctness, measured against human evaluation and with the same datasets, with differences in the mean correctnesses of up to around 4%. The one notable exception was, again, the PaperQA2 RAG agent answering the questions from the LitQA2_edit dataset.

This seems to suggest that the differences in methodology between the PaperQA2 RAG agent, the OpenAI RAG agent and the Simple OpenAI RAG agent are negligible. The PaperQA2 RAG agent answering the questions from the LitQA2_edit dataset is an anomalous datapoint, for both the accuracy and correctness. The original [LitQA2](https://github.com/Future-House/LAB-Bench/blob/main/LitQA2/litqa-v2-public.jsonl) dataset questions that I used to construct the LitQA2_edit dataset contained some questions that were used to optimise the performance of PaperQA2 and some "unseen" questions that were used to ensure that PaperQA2 did not overfit to the optimisations. This seems to suggest that some overfitting in PaperQA2 to the LitQA2 dataset is present. This is not enough evidence to definitively make that conclusion, but this something to keep in mind when interpreting these results.

## Acknowledgements
Thanks to [Dr Boris Bolliet](https://github.com/borisbolliet) for advice and support during the development of this project.
