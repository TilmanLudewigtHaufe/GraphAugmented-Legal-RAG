import os
import re
import ast
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import glob
from dotenv import load_dotenv
import logging
import json
import networkx as nx
from operator import itemgetter
import math
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MinMaxScaler
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
api_key = os.getenv("OPENAI_API_KEY")

system_prompt_fact_extract = (
    """You are a fact extractor who pulls out key facts from a given German query. 
    The purpose of this task is to structure German legal scenarios for later query purposes. 
    You are provided with a query (a sentence or a paragraph). 
    Your task is to extract the key facts mentioned in the given query. 
    These facts should represent the main points as per the context.
    Thought 1: As you parse through the query, identify the key facts mentioned in it. 
    Facts may include conditions, entities, locations, organizations, persons, acronyms, documents, services, concepts, etc. 
    Facts should be as atomistic as possible.
    Thought 2: Each fact is independent and does not need to have a relationship with other facts. 
    Each fact stands on its own.
    Format your output as a Python list. 
    Each element of the list is a string representing a fact, like the following:
    ['fact 1', 'fact 2', 'fact 3', ...]
    Ensure that each fact is distinct, with no duplicates or minor variations of another fact.
    Example: Query: "frau x wurde nach 5 jahren im betrieb mit 10 mitarbeitern gekündigt, weil sie lange krank ist.
    sie ist schwerbehindert und hat 10 kinder."
    Response: ['krankheitsbedingte kündigung', 'kündigung nach 5 jahre betriebszugehörigkeit', 'betrieb mit 10 mitarbeitern', 'kündigung schwerbehinderte mitarbeiterin', 'mitarbeiter mit 10 kindern']"""
)

def generate_fact_extract(query):

    client = OpenAI(api_key=api_key)
    #gpt-4-1106-preview #gpt-3.5-turbo-1106
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system",
             "content":
                 f"{system_prompt_fact_extract}"
             },
            {"role": "user", "content": query}
        ]
    )

    response = completion.choices[0].message.content #only the content
    # Convert the string representation of a list to an actual list
    response_list = ast.literal_eval(response)


    return response_list

while True:
    query = input("Prompt: ")
    if query.lower() in ['quit', 'q', 'exit']:
        break

    fact_extract = generate_fact_extract(query)
    print(fact_extract)