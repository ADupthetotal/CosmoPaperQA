# CosmoPaperQA
A dataset designed to test AI RAG (Retrieval Augmented Generation) for cosmology research applications, called CosmoPaperQA. Also includes a "proof of concept" AI evaluation algorithm that is designed to perform automated performance evaluation for complex research retrieval questions, like those which are included in the the CosmoPaperQA dataset. This new AI evaluation algorithm is called Embed_AI. 

The python code above uses the InspectAI framework to do performance evaluation using the CosmoPaperQA dataset. Evaluate_CosmoPaperQA_OpenAI.py uses the native OpenAI vector file stores (the vector store needs to be created seperately) and a simple OpenAI question answering agent to generate answers to the questions in CosmoPaperQA. Evaluate_CosmoPaperQA_PaperQA2.py uses [PaperQA2](https://github.com/Future-House/paper-qa) to generate the answers to the questions in CosmoPaperQA.

Libraries that are required for both python scripts are [os](https://docs.python.org/3/library/os.html), [openai](https://pypi.org/project/openai/), [pydantic](https://pypi.org/project/pydantic/), [numpy](https://pypi.org/project/numpy/), [re](https://docs.python.org/3/library/re.html), [rake_nltk](https://pypi.org/project/rake-nltk/), [pylatexenc.latex2text](https://pypi.org/project/pylatexenc/), [nltk](https://www.nltk.org/), [json](https://docs.python.org/3/library/json.html), [typing](https://docs.python.org/3/library/typing.html) and [inspect_ai](https://inspect.aisi.org.uk/). Evaluate_CosmoPaperQA_PaperQA2.py additionally requires [paperqa](https://github.com/Future-House/paper-qa).

LitQA2_edit is an edited version of [LitQA2](https://github.com/Future-House/LAB-Bench/blob/main/LitQA2/litqa-v2-public.jsonl) used to compare CosmoPaperQA to other datasets designed for performance evaluation of AI agents in scientific research.

Even though the Embed_AI algorithm implemented here is still not in agreement with human evaluation, it is still more accurate than AI evaluation alone. Also, this particular implementation is unrefined and there is room for much improvement.

For the CosmoPaperQA dataset, with OpenAI RAG answering, accuracy of the Embed_AI algorithm vs the AI algorithm only is 76% vs 70% (respectively).

For the CosmoPaperQA dataset, with PaperQA2 RAG answering, accuracy of the Embed_AI algorithm vs the AI algorithm only is 81% vs 73% (respectively).

For the LitQA2_edit dataset, with OpenAI RAG answering, accuracy of the Embed_AI algorithm vs the AI algorithm only is 89% vs 78% (respectively).

For the LitQA2_edit dataset, with PaperQA2 RAG answering, accuracy of the Embed_AI algorithm vs the AI algorithm only is 83% vs 78% (respectively).

Thus, even with this unrefined implementation, there is a measureable improvement with the Embed_AI algorithm versus the AI evaluation only.

Thanks to [Dr Boris Bolliet](https://github.com/borisbolliet) for advice and support during the development of this project.
