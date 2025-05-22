import os
from openai import OpenAI
from pydantic import BaseModel, Field
from paperqa import Settings, Docs

async def rag_agent(question, vector_store, rag_model) -> str:
    """
    Runs the PaperQA2 RAG, returning the answer to the inputted question, given the vector_store object.
    
    Args:
        question: Question to be answered
        vector_store: Documents to be searched for answers (object class is part of PaperQA2 library)
        rag_model: LLM that can be used to power PaperQA2 RAG
    Returns:
        Answer to the inputted question
    """
    settings.temperature=0.0
    settings.llm = rag_model

    answer_response = await vector_store.aquery(
        query=question,
        settings=settings   
    )
    response = answer_response.model_dump()
    return response["answer"]

#"directory" is directory in which papers used in RAG can be found
directory = "/home/adrian/Documents/University Work/Part III Project/cmbagent_dataset/Source_Papers"

settings = Settings(
    parsing={
        "use_doc_details": False,  # Disable metadata extraction
        "disable_doc_valid_check": True  # Skip document validation
    },
)

vector_store = Docs()
# valid extensions include .pdf, .txt, .md, and .html
full_paths = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory)]

for doc in (full_paths):
    vector_store.add(doc, settings=settings)
    
from typing import Literal
import numpy as np
from numpy.linalg import norm
import re
import time
from rake_nltk import Rake
from pylatexenc.latex2text import LatexNodes2Text
import nltk
import json

client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

#code to set up keyphrase extraction + extraction of questions and ideal from files
nltk.download('stopwords')
nltk.download('words')
english_words = set(nltk.corpus.words.words())

custom_stopwords = set(nltk.corpus.stopwords.words('english')) - {"no", "not", "than", "more", "same", "before", "after", "now", "then", "above", "below", "over", "under", "like", "other", "such", "few", "most", "some", "between"}  # Keep logical comparatives- important for RAG analysis 

class eval_format(BaseModel):
    Evaluation: Literal["Same", "Similar", "Different"] = Field(
    description=r"""If a point is conveyed in both answers, as responses to the associated question, output "Same".
    If a similar points is conveyed in both answers, as responses to the associated question, output "Similar".
    If all of the points are different in both answers, as responses to the associated question, output "Different".""")

#function for the evaluation agent
async def eval_agent(question, answer, ideal, eval_model) -> str:
    """
    Runs the OpenAI Evaluation AI
    
    Args:
        question: Question that the two answers are answering (included for context)
        answer: Generated answer to the question
        ideal: "Ideal" answer the generated answer is to be compared to.
        eval_model: OpenAI model to power the agent
    
    Returns:
        Evaluation in the form of "Same", "Similar" or "Different". If the API call fails, returns "N/A"
    """
    eval_message="""
    You are an evaluation agent tasked with comparing the given two different answers to the same question. 
    Focus on the meaning of both answers, in the context of the question, when formulating your evaluation.
    If you are unsure about the above criteria for the answers to the associated question, output "Unsure".
    Ensure that differences between numerical values and results between the two answers are emphasised in your analysis, unless the question specifically allows for approximations/inexact numerical values. 
    Then, if the question specifically allows for approximations/inexact numerical values, only compare the numerical values approximately.
    """
    eval_assistant = client.beta.assistants.create(
        name="eval_test",
        instructions=eval_message,
        model=eval_model, 
        temperature = 0.0,
        top_p = 0.2,
        response_format= {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": eval_format.model_json_schema()
            },
        }
    )
    
    thread = client.beta.threads.create(
                    messages=[],
                )
    
    parsed = client.beta.threads.messages.create(
                    thread_id=thread.id,
                    content=question+answer+str(ideal),
                    role='user',
                )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=eval_assistant.id,
        # pass the latest system message as instructions
        instructions=eval_message,
    )
    run = client.beta.threads.runs.retrieve(run.id, thread_id=thread.id)
    while run.status=="queued" or run.status=="in_progress":
        time.sleep(0.1)
        run = client.beta.threads.runs.retrieve(run.id, thread_id=thread.id)
    if run.status=="completed":
        response_messages = client.beta.threads.messages.list(thread.id, order="asc")
        for message in response_messages.data:
            for content in message.content:
                output=content.text.value
                if output.startswith("{"):
                    data=json.loads(output)
                    try:
                        evaluation=data.get("Evaluation")
                    except:
                        print("Evaluation not found", end="\r", flush=True)
    if not ("evaluation" in locals()):
        evaluation="N/A"
    try:
        client.beta.assistants.delete(assistant_id=eval_assistant.id)
    except:
        pass
    return evaluation
    
def preprocess_text(text):
    """
    Preprocesses text for keyphrase extraction
    """
    # Replace decimals/commas in numbers with an underscore and replace hyphens with underscores, generally (except for negative numbers).
    #It is only these cases that the sentence tokenizer in Rake doesn't seem to handle well
    text = re.sub(r'(\d+)\.(\d+)', r'\1_\2', text)
    text = re.sub(r'(\d+)\,(\d+)', r'\1\2', text)
    #cursive l in text is formatted strangely in ChatGPT output
    text = text.replace("`", "l")
    # Pattern explanation:
    # (?<!\s)-(?!\d) - matches hyphens not preceded by whitespace or followed by digit
    # | - OR
    # (?<=\s)-(?=\D) - matches hyphens preceded by whitespace and followed by non-digit
    text = re.sub(r'(?<!\s)-(?!\d)|(?<=\s)-(?=\D)', '_', text)
    return text
    
#function for the enbedding answers algorithm
async def embedding_answers(answer, ideal, custom_stopwords, english_words) -> str:
    """
    Novel part of AI evaluation algorithm. This algorithm extracts the keyphrases from the generated and "ideal" answers and then compares the cosine similarity of the vector embeddings between the keyphrases of the "ideal" answer and the generated answer. It gets the maximum cosine similarity for each keyphrase in the "ideal" answer and takes the mean of all of them. This mean is the returned "score". There is some additional preprocessing due to formatting and additional handling of "names" that may not have a meaningful vector embedding, but that is the main idea.
    
    Args:
        answer: Generated answer to the question
        ideal: "Ideal" answer the generated answer is to be compared to.
        custom_stopwords: A list of common words for the keyphrase extractor to automatically ignore.
        english_words: A list of words in english
    Returns:
        A mean score between 0 and 1 (in practise, between ~0.7 and 1). Generated answer considered "correct" if mean score >=0.8 
    """
    #tell Rake to leave logical comparatives alone
    r = Rake(stopwords=custom_stopwords)
    #Extraction given the text.
    text1=preprocess_text(answer)
    #ideal is formatted using latex for CosmoPaperQA
    ideal=LatexNodes2Text().latex_to_text(ideal)
    text2=preprocess_text(ideal)
    r.extract_keywords_from_text(text1)
    key_phrases1=r.get_ranked_phrases()
    r.extract_keywords_from_text(text2)
    key_phrases2=r.get_ranked_phrases()
    result_1=[]
    for string_ideal in key_phrases2:
        #check for "names" that need to be matched exactly
        #checks that string_ideal is one word with at least one letter and that is not in english
        if (not (" " in string_ideal)) and (any(char.isalpha() for char in string_ideal)) and (not (string_ideal in english_words)):
            #if this word does exist in the answer...
            string_ideal=string_ideal.replace("_", "")
            #sort out odd formatting issues surrounding underscores in "names"
            if (string_ideal in text1.lower()):
                #we have a match!
                result_1.append(1)
            else:
                #if not, no match, therefore "incorrect"
                #0.7 works welll as "incorrect" cosine similarity for text-embedding-ada-002 model. 
                #If using text-embedding-3-large model, 0.3 works better as "incorrect" cosine similarity
                result_1.append(0.7)
        else:
            max_cos=0
            check=0
            for string_gen in key_phrases1:
                if (string_ideal==string_gen and check==0):
                    max_cos=1
                    result_1.append(max_cos)
                    check=1
            if (max_cos!=1):
                resp1 = client.embeddings.create(
                    input=string_ideal,
                    model="text-embedding-ada-002",
                    encoding_format= "float",
                )
                for string_gen in key_phrases1:
                    resp2 = client.embeddings.create(
                        input=string_gen,
                        model="text-embedding-ada-002",
                        encoding_format= "float",
                    )
                    a=np.array(resp1.data[0].embedding)
                    b=np.array(resp2.data[0].embedding)
                    cos=np.dot(a,b)/(norm(a)*norm(b))
                    if (cos>max_cos):
                        max_cos=cos
                result_1.append(max_cos)
    #mean is a crude way to combine these scores.
    #will consider "correct" if mean >=0.8, otherwise "incorrect" (also a crude metric)
    #0.8 value is designed for text-embedding-ada-002 model. If using text-embedding-3-large model, 0.4 works better
    return np.mean(np.array(result_1)) 

"""
Demonstration of how eval_agent and embedding_answers are used to perform the Embed_AI performance evaluation algorithm

eval_ai = await eval_agent(this_question, this_answer, this_ideal, eval_model)
embedding_eval = await embedding_answers(this_answer, this_ideal, custom_stopwords, english_words)

if (embedding_eval >= 0.8 and (eval_ai in ["Same", "Similar"])):
            #Embed_AI evaluation algorithm will consider a generated answer "correct" if both the embedding_eval score is >=0.8 and if the AI evaluation returns "Same" or "Similar".
"""
import os
from openai import OpenAI
from pydantic import BaseModel, Field, conlist
import numpy as np
from numpy.linalg import norm
import re
from rake_nltk import Rake
from pylatexenc.latex2text import LatexNodes2Text
import nltk
import json
from typing import Literal
from typing import Any

from inspect_ai.solver import (
    TaskState,
    solver,
)
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    stderr,
    scorer,
)
from inspect_ai.solver import bridge
from inspect_ai import eval

import pandas as pd

#poor man's pass by ref
class CSVHolder:
    def __init__(self, value):
        self.value = value

client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

question=[]
ideal=[]

#input single question here
question.append(r'In the "Cosmology with one galaxy?" paper, how is the direct comparison of the performance of the IllustrisTNG and the SIMBA simulations performed?')
ideal.append(r"There is no correspondence between simulations among the IllustrisTNG and the SIMBA sets. Thus, a direct comparison between the two is not performed.")

def inspect_ai_eval(rag_agent, eval_agent, embedding_answers):
    """
    Function to implement inspect_ai evaluation using rag_agent and eval_agent functions

    Args:
        rag_agent: function that implements the RAG for the generated answers to the questions in the dataset
        eval_agent: function that implements the AI evaluation agent for Embed_AI
        embedding_answers: function that implements the cosine similarity of vector embeddings component of Embed_AI
    Returns:
        nominally None, but will print out a csv file with all of the relevant evaluation and RAG information present.
    """
    #setup mytasks for evaluation
    mytasks = []
    for i in range(len(question)):
        mytasks.append({
            "input": question[i],
            "target": ideal[i]
        })
    #setup output DataFrame
    new_output_holder= CSVHolder(pd.DataFrame({
        'question': pd.Series(dtype='object'),
        'answer': pd.Series(dtype='object'),
        'ideal': pd.Series(dtype='object'),
        'AI_eval': pd.Series(dtype='object'),
        'embedding_eval': pd.Series(dtype='float'),
        'evaluation': pd.Series(dtype='object')
    }))
    
    async def my_agent(task_input: list[dict[str, Any]]) -> str:
        #replace rag_agent function if needed, to implement custom RAG agent.
        #Can put OpenAI RAG agent or PaperQA2 RAG agent here
        answer = await rag_agent(task_input[0]["content"], vector_store, "gpt-4o-mini")
        return answer
    
    @solver
    def my_solver():
        async def solve(state: TaskState) -> TaskState:
            result = await my_agent(state["input"])
            return {"output":result}
        return solve
    
    @task
    def my_task(tasks):
        return Task(
            dataset=[Sample(
                input=tasks[i]["input"],
                target=tasks[i]["target"]
            ) for i in range(len(tasks))],
            solver = bridge(my_solver()),
            #replace "gpt-4o-mini" with model you want to use for evaluation AI
            scorer = my_scorer("gpt-4o-mini", custom_stopwords, english_words, new_output_holder),
        )
    
    @scorer(metrics=[accuracy(), stderr()])
    def my_scorer(eval_model: str, custom_stopwords: set, english_words: set, new_output_holder):
        async def score(state: TaskState, target: Target) -> Score:
            this_question = state.input_text
            this_answer = state.output.completion
            this_ideal = target.text

            #replace eval_agent function if needed, to implement custom evaluation agent
            eval_ai = await eval_agent(this_question, this_answer, this_ideal, eval_model)
            embedding_eval = await embedding_answers(this_answer, this_ideal, custom_stopwords, english_words)
    
            #sort out new_output pass by ref
            if (embedding_eval >= 0.8 and (eval_ai in ["Same", "Similar"])):
                #Embed_AI evaluation algorithm will consider a generated answer "correct" if both the embedding_eval score is >=0.8 and if the AI evaluation returns "Same" or "Similar".
                new_entry=pd.DataFrame({"question":this_question, "answer":this_answer, "ideal":this_ideal, "AI_eval": eval_ai, "embedding_eval": embedding_eval, "evaluation":"CORRECT"}, index=[0])
                new_output_holder.value=pd.concat([new_output_holder.value, new_entry], ignore_index=True)
                new_output_holder.value.to_csv("output_Single_Question_Example.csv", index=False)
                return Score(value=CORRECT)
            else:
                new_entry=pd.DataFrame({"question":this_question, "answer":this_answer, "ideal":this_ideal, "AI_eval": eval_ai, "embedding_eval": embedding_eval, "evaluation":"INCORRECT"}, index=[0])
                new_output_holder.value=pd.concat([new_output_holder.value, new_entry], ignore_index=True)
                new_output_holder.value.to_csv("output_Single_Question_Example.csv", index=False)
                return Score(value=INCORRECT)
        return score
    
    new_output_holder.value.to_csv("output_Single_Question_Example.csv", index=False)
    logs = eval(
        my_task(mytasks)
    )
    print(logs)
    for log in logs:
        print(log.results)
    return None
    
inspect_ai_eval(rag_agent, eval_agent, embedding_answers)
