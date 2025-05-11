# CosmoPaperQA

## Features
CosmoPaperQA is a dataset designed to test AI RAG (Retrieval Augmented Generation) for cosmology research applications.  A current bottleneck in research into utilising AI in cosmological reseach is the lack of a specific cosmology research dataset.

This repository also includes a "proof of concept" AI evaluation algorithm that is designed to perform automated performance evaluation on RAG agents when answering complex research retrieval questions, like those which are included in the the CosmoPaperQA dataset. This new AI evaluation algorithm is called Embed_AI. 

## Description
### OpenAI evaluator agent
The OpenAI evaluator agent takes in the generated answer, the "correct" or "ideal" answer and the original question as input. It is then asked if the two answers convey the same point, given the context of the question. The possible outputs are "Same", "Similar" or "Different".

### Embed_AI evaluation
Embed_AI utilises both the custom OpenAI evaluator agent and cosine similarities of vector embeddings between extracted key phrases of the generated and "correct" answers. It uses both the evaluator agent and the cosine similarities to give a "correct" or "incorrect" evaluation of the generated answers.

### PaperQA2 RAG
The PaperQA2 RAG takes in a question and documents and uses the information in the documents to answer the question. The code involves gathering and ranking evidence/chunks from source papers, based on their relevance to answering the given question, and then generating the answer to the question based on the top ranked chunks. The [PaperQA2](https://github.com/Future-House/paper-qa) code does also allow for citation traversal, using Semantic Scholar or CrossRef, but it was not implemented in this code or used to generate the given results.

### OpenAI RAG
The OpenAI RAG also takes in a question and documents and uses the information in the documents to answer the question. The code follows the same RAG methodology as the PaperQA2 algorithm. The main difference between the two is that the PaperQA2 RAG uses cosine similarities to rank the chunks from the papers, while OpenAI RAG uses semantic search to rank the chunks.

## Implementing the Codebase
The python code in the directory "Code_Breakdown" uses the InspectAI framework to do performance evaluation using the CosmoPaperQA dataset and the Embed_AI evaluation algorithm. 

Either the OpenAI RAG framework or the [PaperQA2](https://github.com/Future-House/paper-qa) RAG framework can be used to generate answers to the questions in CosmoPaperQA. 

Use the functions contained in OpenAI_RAG_Agent/PaperQA2_RAG_agent, Embed_AI and Inspect_AI_framework and then run the line `inspect_ai_eval(rag_agent, eval_agent, embedding_answers)` to implement the RAG answer generation and performance evaluation of the CosmoPaperQA dataset.

Ensure that the follwoing libraries are installed before running the python scripts: 
[os](https://docs.python.org/3/library/os.html), [openai](https://pypi.org/project/openai/), [pydantic](https://pypi.org/project/pydantic/), [numpy](https://pypi.org/project/numpy/), [re](https://docs.python.org/3/library/re.html), [rake_nltk](https://pypi.org/project/rake-nltk/), [pylatexenc.latex2text](https://pypi.org/project/pylatexenc/), [nltk](https://www.nltk.org/), [json](https://docs.python.org/3/library/json.html), [typing](https://docs.python.org/3/library/typing.html) and [inspect_ai](https://inspect.aisi.org.uk/). PaperQA2_RAG_agent additionally requires [paperqa](https://github.com/Future-House/paper-qa).

## Edited LitQA2 Dataset
LitQA2_edit is an edited version of [LitQA2](https://github.com/Future-House/LAB-Bench/blob/main/LitQA2/litqa-v2-public.jsonl) used to compare CosmoPaperQA to other datasets designed for performance evaluation of AI agents in scientific research.

## Summary of Results
Correctness is the percentage of generated answers that were evaluated to be correct.

|                               | Correctness using OpenAI evaluator agent| Correctness using Embed_AI evaluation| Correctness using human evaluation|
|-------------------------------|-----------------------------------------|--------------------------------------|-----------------------------------|
|   CosmoPaperQA with OpenAI RAG|                                      80%|                                   77%|                                70%|
| CosmoPaperQA with PaperQA2 RAG|                                      90%|                                   87%|                                73%|
|    LitQA2_edit with OpenAI RAG|                                      85%|                                   75%|                                79%|
|  LitQA2_edit with PaperQA2 RAG|                                      80%|                                   73%|                                79%|

Accuracy is the percentage of the given automated evaluation results that agree with the human evaluations.

|                               | Accuracy of OpenAI evaluator agent| Accuracy of Embed_AI evaluation|
|-------------------------------|-----------------------------------|--------------------------------|
|   CosmoPaperQA with OpenAI RAG|                                70%|                             76%|
| CosmoPaperQA with PaperQA2 RAG|                                73%|                             81%|
|    LitQA2_edit with OpenAI RAG|                                78%|                             89%|
|  LitQA2_edit with PaperQA2 RAG|                                79%|                             83%|

The OpenAI GPT-4o-mini LLM model was used to power all of the agents used to generate the above results. In total, getting these results cost around £20, with the PaperQA RAG runs making up the vast majority of that cost.

## Key Findings
Even though the Embed_AI algorithm's correctness here is still not in agreement with human evaluation, the accuracy of the Embed_AI algorithm is greater than the custom OpenAI evaluator agent alone, across all data. 

Also, this particular implementation is unrefined and there is room for much improvement. As can be seen in the table above, even with this unrefined implementation, there is a measureable improvement in accuracy from the custom OpenAI evaluator to the Embed_AI algorithm. Therefore, Eval_AI can increase the accuracy of performance evaluation, even unrefined. 

From the results gathered, the PaperQA2 and the OpenAI RAG agent answers were observed to have similar levels of correctness, measured against human evaluation and with the same datasets, with differences of less than 3% across all results gathered. 

We conjecture that this similarity in correctness between the OpenAI RAG agent and the PaperQA2 RAG agents (across the CosmoPaperQA and the LitQA2_edit datasets) is due to this similar methodology in RAG implementation. If this is true, then the design of the OpenAI RAG agent was successful in its aim to mimic the behaviour of the PaperQA2 code.

## Acknowledgements
Thanks to [Dr Boris Bolliet](https://github.com/borisbolliet) for advice and support during the development of this project.
