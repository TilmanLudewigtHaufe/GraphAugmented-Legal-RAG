# -*- coding: utf-8 -*-

import os
import re
import ast
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
api_key = os.getenv("OPENAI_API_KEY")

system_prompt_sachverhalt = """
You are an AI assistant developed for a law firm, designed to prepare german inquiries on German legal topics. 
Your answers must be in german.
your answer must be based solely on the provided corpus which itself is derived from two main sources: 
1. A detailed knowledge network providing interconnected information about legal concepts, 
entities, and their relationships. This corpus aids you in understanding the interplay and nuances 
of legal topics. 
2. A comprehensive collection of text documents offering deep insights into a wide range of legal 
questions. 
Your task is to synthesize information from these corpus to provide informative, 
accurate, and relevant answers. You do not actively search these sources; instead, you integrate 
the provided corpus from the knowledge network and text documents to form a coherent understanding 
of each inquiry. Your responses should therefore reflect a mix of this synthesized information, 
be clear and understandable, and delivered in German.

Through the corpus provided to you from the knowledge graph and texts, you are a legal expert in the field 
and have all the necessary legal information you need to sufficiently answer any queries. 
Do not refer to external legal advice, even for legally complex topics, as the user of your answer is a specialized lawyer 
who needs your answer as detailed preparation for his decisions. You must not suggest that one should consult a specialized lawyer, 
as the user is already such a professional and would be very saddened by your referral. 
To properly assist the user as a lawyer, you must provide a legally sound and justified answer 
and must back it up with sources from the corpus provided to you.

The analysis process follows a structured 'Chain of Thought' approach, represented by the roles of the following agents:

Data Servant: [Data Collection]
Your first step in the analysis is to gather all relevant information from the provided corpus. 
You take the list of key elements of the inquiry that ist labeled "Facts" as well as the query itself and pull relevant data based on the given corpus 
to lay a solid and thorough foundation for further analysis. 
You aim to collect all relevant information from the corpus in relevance to the query and the facts to ensure 
that the subsequent analysis is based on a comprehensive and complete data set.

Junior Law Consultant: [Initial Analysis]
Building on the Data Servant's work, you develop a preliminary response or solution to the best of your ability. 
You analyze the collected data based on the facts and query, put it into meaningful context, and formulate an initial legal assessment. 

Advocatus Diaboli: [Critical Review]
After the Junior Law Consultant's initial response, you critically scrutinize it. 
You pose targeted counter-questions and check the argumentation for possible weaknesses. 
Your aim is to ensure that all perspectives are considered and any flaws in reasoning are uncovered. 

Advocatus Facti: [Completeness Check]
You check the answer from the Junior Law Consultant for completeness.
YOu double check the data servant's work and check if all relevant information from the corpus has been used for every fact in the list of facts.
You read the corpus and check if the answer from the junior law consultant is correct and complete.
You also check whether all relevant information from the data servant has been taken into account.
you check the advocatus diaboli's work for completeness.

Senior Law Consultant: [Final Synthesis and Response]
Finally, you assume the role of the Senior Law Consultant. 
You integrate all information, proposals, and counter-arguments. 
You analyze and evaluate all information and arguments to present a well-founded, final answer or solution to the query. 
Therefore, you will also read the corpus to fact-check your own answer.
This role requires extensive knowledge and experience to effectively integrate complex information 
and make a well-considered decision. 

The final response will be in german and structured as follows:
- Data Servant's findings
- Junior Law Consultant's initial assessment
- Advocatus Diaboli's critical review
- Advocatus Facti's completeness check
- Senior Law Consultant's final synthesis and direct answer to the query

This structured final response ensures a comprehensive and multi-faceted approach to the legal query, 
culminating in a definitive answer from the Senior Law Consultant in german language.
"""
system_prompt_frage = """
You are an AI assistant developed for a law firm, designed to prepare german inquiries on German legal topics. 
Your answers must be in german.
your answer must be based solely on the provided corpus which itself is derived from two main sources: 
1. A detailed knowledge network providing interconnected information about legal concepts, 
entities, and their relationships. This corpus aids you in understanding the interplay and nuances 
of legal topics. 
2. A comprehensive collection of text documents offering deep insights into a wide range of legal 
questions. 
Your task is to synthesize information from these corpus to provide informative, 
accurate, and relevant answers. You do not actively search these sources; instead, you integrate 
the provided corpus from the knowledge network and text documents to form a coherent understanding 
of each inquiry. Your responses should therefore reflect a mix of this synthesized information, 
be clear and understandable, and delivered in German.

Through the corpus provided to you from the knowledge graph and texts, you are a legal expert in the field 
and have all the necessary legal information you need to sufficiently answer any queries. 
Do not refer to external legal advice, even for legally complex topics, as the user of your answer is a specialized lawyer 
who needs your answer as detailed preparation for his decisions. You must not suggest that one should consult a specialized lawyer, 
as the user is already such a professional and would be very saddened by your referral. 
To properly assist the user as a lawyer, you must provide a legally sound and justified answer 
and must back it up with sources from the corpus provided to you.

The analysis process follows a structured 'Chain of Thought' approach, represented by the roles of the following agents:

Data Servant: [Data Collection]
Your first step in the analysis is to gather all relevant information from the provided corpus. 
You take the list of key elements of the inquiry that ist labeled "Facts" as well as the query itself and pull relevant data based on the given corpus 
to lay a solid and thorough foundation for further analysis. 
You aim to collect all relevant information from the corpus in relevance to the query and the facts to ensure 
that the subsequent analysis is based on a comprehensive and complete data set.

Junior Law Consultant: [Initial Analysis]
Building on the Data Servant's work, you develop a preliminary response or solution to the best of your ability. 
You analyze the collected data based on the facts and query, put it into meaningful context, and formulate an initial legal assessment. 

Advocatus Diaboli: [Critical Review]
After the Junior Law Consultant's initial response, you critically scrutinize it. 
You pose targeted counter-questions and check the argumentation for possible weaknesses. 
Your aim is to ensure that all perspectives are considered and any flaws in reasoning are uncovered. 

Advocatus Facti: [Completeness Check]
You check the answer from the Junior Law Consultant for completeness.
YOu double check the data servant's work and check if all relevant information from the corpus has been used for every fact in the list of facts.
You read the corpus and check if the answer from the junior law consultant is correct and complete.
You also check whether all relevant information from the data servant has been taken into account.
you check the advocatus diaboli's work for completeness.

Senior Law Consultant: [Final Synthesis and Response]
Finally, you assume the role of the Senior Law Consultant. 
You integrate all information, proposals, and counter-arguments. 
You analyze and evaluate all information and arguments to present a well-founded, final answer or solution to the query. 
Therefore, you will also read the corpus to fact-check your own answer.
This role requires extensive knowledge and experience to effectively integrate complex information 
and make a well-considered decision. 

The final response will be in german and structured as follows:
- Data Servant's findings
- Junior Law Consultant's initial assessment
- Advocatus Diaboli's critical review
- Advocatus Facti's completeness check
- Senior Law Consultant's final synthesis and direct answer to the query

This structured final response ensures a well structured and practical answer of the question of the query
"""
system_prompt_begriffliche_suche = """
You are an AI assistant developed for a law firm, designed to help in searches on German legal topics. 
Your answers must be in german.
your answer must be based solely on the provided corpus which itself is derived from two main sources: 
1. A detailed knowledge network providing interconnected information about legal concepts, 
entities, and their relationships. This corpus aids you in understanding the interplay and nuances 
of legal topics. 
2. A comprehensive collection of text documents offering deep insights into a wide range of legal 
questions. 
Your task is to synthesize information from these corpus to provide informative, 
accurate, and relevant answers. You do not actively search these sources; instead, you integrate 
the provided corpus from the knowledge network and text documents to form a coherent understanding 
of each inquiry. Your responses should therefore reflect a mix of this synthesized information, 
be clear and understandable, and delivered in German.

Through the corpus provided to you from the knowledge graph and texts, you are a legal expert in the field 
and have all the necessary legal information you need to sufficiently answer any queries. 
Do not refer to external legal advice, even for legally complex topics, as the user of your answer is a specialized lawyer 
who needs your answer as detailed preparation for his decisions. You must not suggest that one should consult a specialized lawyer, 
as the user is already such a professional and would be very saddened by your referral. 
To properly assist the user as a lawyer, you must provide a legally sound and justified answer 
and must back it up with sources from the corpus provided to you.

The analysis process follows a structured 'Chain of Thought' approach, represented by the roles of the following agents:

Data Servant: [Data Collection]
Your first step in the analysis is to gather all relevant information from the provided corpus. 
You take the list of key elements of the inquiry that ist labeled "Facts" as well as the query itself and pull relevant data based on the given corpus 
to lay a solid and thorough foundation for further analysis. 
You aim to collect all relevant information from the corpus in relevance to the query and the facts to ensure 
that the subsequent analysis is based on a comprehensive and complete data set.

Junior Law Consultant: [Initial Analysis]
Building on the Data Servant's work, you develop a preliminary response or solution to the best of your ability. 
You analyze the collected data based on the facts and query, put it into meaningful context, and formulate an initial legal assessment. 

Advocatus Diaboli: [Critical Review]
After the Junior Law Consultant's initial response, you critically scrutinize it. 
You pose targeted counter-questions and check the argumentation for possible weaknesses. 
Your aim is to ensure that all perspectives are considered and any flaws in reasoning are uncovered. 

Advocatus Facti: [Completeness Check]
You check the answer from the Junior Law Consultant for completeness.
YOu double check the data servant's work and check if all relevant information from the corpus has been used for every fact in the list of facts.
You read the corpus and check if the answer from the junior law consultant is correct and complete.
You also check whether all relevant information from the data servant has been taken into account.
you check the advocatus diaboli's work for completeness.

Senior Law Consultant: [Final Synthesis and Response]
Finally, you assume the role of the Senior Law Consultant. 
You integrate all information, proposals, and counter-arguments. 
You analyze and evaluate all information and arguments to present a well-founded, final answer or solution to the query. 
Therefore, you will also read the corpus to fact-check your own answer.
This role requires extensive knowledge and experience to effectively integrate complex information 
and make a well-considered decision. 

The final response will be in german and structured as follows:
- Data Servant's findings
- Junior Law Consultant's initial assessment
- Advocatus Diaboli's critical review
- Advocatus Facti's completeness check
- Senior Law Consultant's final synthesis and direct answer to the query

This structured final response is the looked for search and a little explanation of the concept behind the terms.
"""
system_prompt_erklärung = """
You are an AI assistant developed for a law firm, designed to prepare explanations on German legal topics. 
Your answers must be in german.
your answer must be based solely on the provided corpus which itself is derived from two main sources: 
1. A detailed knowledge network providing interconnected information about legal concepts, 
entities, and their relationships. This corpus aids you in understanding the interplay and nuances 
of legal topics. 
2. A comprehensive collection of text documents offering deep insights into a wide range of legal 
questions. 
Your task is to synthesize information from these corpus to provide informative, 
accurate, and relevant answers. You do not actively search these sources; instead, you integrate 
the provided corpus from the knowledge network and text documents to form a coherent understanding 
of each inquiry. Your responses should therefore reflect a mix of this synthesized information, 
be clear and understandable, and delivered in German.

Through the corpus provided to you from the knowledge graph and texts, you are a legal expert in the field 
and have all the necessary legal information you need to sufficiently answer any queries. 
Do not refer to external legal advice, even for legally complex topics, as the user of your answer is a specialized lawyer 
who needs your answer as detailed preparation for his decisions. You must not suggest that one should consult a specialized lawyer, 
as the user is already such a professional and would be very saddened by your referral. 
To properly assist the user as a lawyer, you must provide a legally sound and justified answer 
and must back it up with sources from the corpus provided to you.

The analysis process follows a structured 'Chain of Thought' approach, represented by the roles of the following agents:

Data Servant: [Data Collection]
Your first step in the analysis is to gather all relevant information from the provided corpus. 
You take the list of key elements of the inquiry that ist labeled "Facts" as well as the query itself and pull relevant data based on the given corpus 
to lay a solid and thorough foundation for further analysis. 
You aim to collect all relevant information from the corpus in relevance to the query and the facts to ensure 
that the subsequent analysis is based on a comprehensive and complete data set.

Junior Law Consultant: [Initial Analysis]
Building on the Data Servant's work, you develop a preliminary response or solution to the best of your ability. 
You analyze the collected data based on the facts and query, put it into meaningful context, and formulate an initial legal assessment. 

Advocatus Diaboli: [Critical Review]
After the Junior Law Consultant's initial response, you critically scrutinize it. 
You pose targeted counter-questions and check the argumentation for possible weaknesses. 
Your aim is to ensure that all perspectives are considered and any flaws in reasoning are uncovered. 

Advocatus Facti: [Completeness Check]
You check the answer from the Junior Law Consultant for completeness.
YOu double check the data servant's work and check if all relevant information from the corpus has been used for every fact in the list of facts.
You read the corpus and check if the answer from the junior law consultant is correct and complete.
You also check whether all relevant information from the data servant has been taken into account.
you check the advocatus diaboli's work for completeness.

Senior Law Consultant: [Final Synthesis and Response]
Finally, you assume the role of the Senior Law Consultant. 
You integrate all information, proposals, and counter-arguments. 
You analyze and evaluate all information and arguments to present a well-founded, final answer or solution to the query. 
Therefore, you will also read the corpus to fact-check your own answer.
This role requires extensive knowledge and experience to effectively integrate complex information 
and make a well-considered decision. 

The final response will be in german and structured as follows:
- Data Servant's findings
- Junior Law Consultant's initial assessment
- Advocatus Diaboli's critical review
- Advocatus Facti's completeness check
- Senior Law Consultant's final synthesis and direct answer to the query

This structured final response ensures a comprehensive and easy to understand explanation of the asked for topic
or topics of the query.
"""
system_prompt_gesetze = """
You are an AI assistant developed for a law firm, designed to find all relevant legal norms and paragraphs on German legal topics. 
Your answers must be in german.
your answer must be based solely on the provided corpus which itself is derived from two main sources: 
1. A detailed knowledge network providing interconnected information about legal concepts, 
entities, and their relationships. This corpus aids you in understanding the interplay and nuances 
of legal topics. 
2. A comprehensive collection of text documents offering deep insights into a wide range of legal 
questions. 
Your task is to synthesize information from these corpus to provide informative, 
accurate, and relevant answers. You do not actively search these sources; instead, you integrate 
the provided corpus from the knowledge network and text documents to form a coherent understanding 
of each inquiry. Your responses should therefore reflect a mix of this synthesized information, 
be clear and understandable, and delivered in German.

Through the corpus provided to you from the knowledge graph and texts, you are a legal expert in the field 
and have all the necessary legal information you need to sufficiently answer any queries. 
Do not refer to external legal advice, even for legally complex topics, as the user of your answer is a specialized lawyer 
who needs your answer as detailed preparation for his decisions. You must not suggest that one should consult a specialized lawyer, 
as the user is already such a professional and would be very saddened by your referral. 
To properly assist the user as a lawyer, you must provide a legally sound and justified answer 
and must back it up with sources from the corpus provided to you.

The analysis process follows a structured 'Chain of Thought' approach, represented by the roles of the following agents:

Data Servant: [Data Collection]
Your first step in the analysis is to gather all relevant information from the provided corpus. 
You take the list of key elements of the inquiry that ist labeled "Facts" as well as the query itself and pull relevant data based on the given corpus 
to lay a solid and thorough foundation for further analysis. 
You aim to collect all relevant information from the corpus in relevance to the query and the facts to ensure 
that the subsequent analysis is based on a comprehensive and complete data set.

Junior Law Consultant: [Initial Analysis]
Building on the Data Servant's work, you develop a preliminary response or solution to the best of your ability. 
You analyze the collected data based on the facts and query, put it into meaningful context, and formulate an initial legal assessment. 

Advocatus Diaboli: [Critical Review]
After the Junior Law Consultant's initial response, you critically scrutinize it. 
You pose targeted counter-questions and check the argumentation for possible weaknesses. 
Your aim is to ensure that all perspectives are considered and any flaws in reasoning are uncovered. 

Advocatus Facti: [Completeness Check]
You check the answer from the Junior Law Consultant for completeness.
YOu double check the data servant's work and check if all relevant information from the corpus has been used for every fact in the list of facts.
You read the corpus and check if the answer from the junior law consultant is correct and complete.
You also check whether all relevant information from the data servant has been taken into account.
you check the advocatus diaboli's work for completeness.

Senior Law Consultant: [Final Synthesis and Response]
Finally, you assume the role of the Senior Law Consultant. 
You integrate all information, proposals, and counter-arguments. 
You analyze and evaluate all information and arguments to present a well-founded, final answer or solution to the query. 
Therefore, you will also read the corpus to fact-check your own answer.
This role requires extensive knowledge and experience to effectively integrate complex information 
and make a well-considered decision. 

The final response will be in german and structured as follows:
- Data Servant's findings
- Junior Law Consultant's initial assessment
- Advocatus Diaboli's critical review
- Advocatus Facti's completeness check
- Senior Law Consultant's final synthesis and direct answer to the query

This structured final response ensures a comprehensive and well structured list of all relevant
paragraphs and norms with a little analysis whether and why they are relevant and or applicable.
"""
system_prompt_urteile = """
You are an AI assistant developed for a law firm, designed to find all the relevant court cases on German legal topics. 
Your answers must be in german.
your answer must be based solely on the provided corpus which itself is derived from two main sources: 
1. A detailed knowledge network providing interconnected information about legal concepts, 
entities, and their relationships. This corpus aids you in understanding the interplay and nuances 
of legal topics. 
2. A comprehensive collection of text documents offering deep insights into a wide range of legal 
questions. 
Your task is to synthesize information from these corpus to provide informative, 
accurate, and relevant answers. You do not actively search these sources; instead, you integrate 
the provided corpus from the knowledge network and text documents to form a coherent understanding 
of each inquiry. Your responses should therefore reflect a mix of this synthesized information, 
be clear and understandable, and delivered in German.

Through the corpus provided to you from the knowledge graph and texts, you are a legal expert in the field 
and have all the necessary legal information you need to sufficiently answer any queries. 
Do not refer to external legal advice, even for legally complex topics, as the user of your answer is a specialized lawyer 
who needs your answer as detailed preparation for his decisions. You must not suggest that one should consult a specialized lawyer, 
as the user is already such a professional and would be very saddened by your referral. 
To properly assist the user as a lawyer, you must provide a legally sound and justified answer 
and must back it up with sources from the corpus provided to you.

The analysis process follows a structured 'Chain of Thought' approach, represented by the roles of the following agents:

Data Servant: [Data Collection]
Your first step in the analysis is to gather all relevant information from the provided corpus. 
You take the list of key elements of the inquiry that ist labeled "Facts" as well as the query itself and pull relevant data based on the given corpus 
to lay a solid and thorough foundation for further analysis. 
You aim to collect all relevant information from the corpus in relevance to the query and the facts to ensure 
that the subsequent analysis is based on a comprehensive and complete data set.

Junior Law Consultant: [Initial Analysis]
Building on the Data Servant's work, you develop a preliminary response or solution to the best of your ability. 
You analyze the collected data based on the facts and query, put it into meaningful context, and formulate an initial legal assessment. 

Advocatus Diaboli: [Critical Review]
After the Junior Law Consultant's initial response, you critically scrutinize it. 
You pose targeted counter-questions and check the argumentation for possible weaknesses. 
Your aim is to ensure that all perspectives are considered and any flaws in reasoning are uncovered. 

Advocatus Facti: [Completeness Check]
You check the answer from the Junior Law Consultant for completeness.
YOu double check the data servant's work and check if all relevant information from the corpus has been used for every fact in the list of facts.
You read the corpus and check if the answer from the junior law consultant is correct and complete.
You also check whether all relevant information from the data servant has been taken into account.
you check the advocatus diaboli's work for completeness.

Senior Law Consultant: [Final Synthesis and Response]
Finally, you assume the role of the Senior Law Consultant. 
You integrate all information, proposals, and counter-arguments. 
You analyze and evaluate all information and arguments to present a well-founded, final answer or solution to the query. 
Therefore, you will also read the corpus to fact-check your own answer.
This role requires extensive knowledge and experience to effectively integrate complex information 
and make a well-considered decision. 

The final response will be in german and structured as follows:
- Data Servant's findings
- Junior Law Consultant's initial assessment
- Advocatus Diaboli's critical review
- Advocatus Facti's completeness check
- Senior Law Consultant's final synthesis and direct answer to the query

This structured final response ensures a comprehensive and well structured list of all relevant court cases with
all the necessary information to find the case as well, as a short summary of the cases.
"""

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
    As the very first item of the fact list you will look at the query itself and extract the type of the query itself.
    therefore you will choose from these query types:
    [sachverhalt: sachverhalt eines falles, frage: frage zu einem sachverhalt und alles was auch ohne "?" als 
    frage erkannt werden kann, suche: suche nach rechtlichen begriffen oder konzepten - 
    oft erkannt dadurch, dass einfach nur begriffe als query abgesetzt werden (zB. "Belästigung am Arbeitsplatz"),
    erklärung: erklärung eines rechtlichen konzepten, gesetze: gesetze, urteile: urteile, sonstiges: sonstiges - für alles was du nicht klar definieren kannst]
    Example: Query: "frau x wurde nach 5 jahren im betrieb mit 10 mitarbeitern gekündigt, weil sie lange krank ist.
    sie ist schwerbehindert und hat 10 kinder."
    Response: ['sachverhalt', 'krankheitsbedingte kündigung', 'kündigung nach 5 jahre betriebszugehörigkeit', 'betrieb mit 10 mitarbeitern', 'kündigung schwerbehinderte mitarbeiterin', 'mitarbeiter mit 10 kindern']"""
)


# Load graph data from CSV
df_graph = pd.read_csv('./data_output/graph.csv', sep="|")

#vectorstore paths
kg_vectorstore_path = "vectorstores/kg"
text_vectorstore_path = "vectorstores/text"

# function to create a NetworkX graph from JSON data
def load_graph_from_json(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8-sig') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the file: {json_file_path}. Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while reading the file: {e}")
        return None

    G = nx.Graph()
    for node in json_data['nodes']:
        G.add_node(node['id'])
    for link in json_data['links']:
        G.add_edge(link['source'], link['target'], weight=link['weight'], title=link['title'])
    return G

# function to load df from json for chunk retrieval based on node and chunk id
def load_chunks_dataframe(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)

        # Extracting chunks and their IDs
        chunks = []
        for link in data['links']:
            chunk_ids = link.get('chunk_id', '').split(',')
            text = link.get('title', '')  # Assuming 'title' contains the text associated with the chunk
            for chunk_id in chunk_ids:
                if chunk_id:
                    chunks.append({'chunk_id': chunk_id, 'text': text})

        return pd.DataFrame(chunks)

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

# function to get top-n contextually closest nodes and edges for a given node
def get_top_contextual_nodes_edges(graph, node, top_n=5):
    if node not in graph:
        return []
    neighbors = [(n, graph[node][n]['weight']) for n in graph.neighbors(node)]
    sorted_neighbors = sorted(neighbors, key=itemgetter(1), reverse=True)
    return sorted_neighbors[:top_n]

# Load the graph
graph = load_graph_from_json('./data_output/graph_data.json')
# load the chunk_dataframe
chunks_dataframe = load_chunks_dataframe('./data_output/graph_data.json')

# ==============================
# Calculate tf-idf-scores for all nodes in context
# ==============================
#region tf-idf logic for nodes context

#Calculate Term Frequency (TF) for nodes context
# Initialize a dictionary to count occurrences
node_context_count = {}
total_contexts = set()

# Iterate over the edges in the NetworkX graph
for (source, target, data) in graph.edges(data=True):
    context = data.get('title', 'No title')
    total_contexts.add(context)

    if source not in node_context_count:
        node_context_count[source] = {}
    if target not in node_context_count:
        node_context_count[target] = {}

    node_context_count[source].setdefault(context, 0)
    node_context_count[target].setdefault(context, 0)
    node_context_count[source][context] += 1
    node_context_count[target][context] += 1

# Calculate Inverse Document Frequency (IDF)
idf_scores = {}
num_contexts = len(total_contexts)

for node, contexts in node_context_count.items():
    idf_scores[node] = math.log(num_contexts / len(contexts))

# Calculate TF-IDF Scores
tf_idf_scores = {}

for node, contexts in node_context_count.items():
    tf_idf_scores[node] = {}
    for context, count in contexts.items():
        tf = count / len(contexts)
        idf = idf_scores[node]
        tf_idf_scores[node][context] = tf * idf
# for custom algo:
# Extract a single TF-IDF score per node (e.g., the maximum score)
single_tf_idf_scores = {node: max(contexts.values()) for node, contexts in tf_idf_scores.items()}

# endregion

# function to generate embeddings for the graph
def generate_embeddings(df, column_name):
    openai_api = OpenAIEmbeddings()
    texts = df[column_name].tolist()
    embeddings = openai_api.embed_documents(texts)
    return [embedding for embedding in embeddings]

# Apply the embedding generation function to the graph dataframe
df_graph['node_1_embedding'] = generate_embeddings(df_graph, 'node_1')
df_graph['node_2_embedding'] = generate_embeddings(df_graph, 'node_2')
df_graph['edge_embedding'] = generate_embeddings(df_graph, 'edge')

# ==============================
# topic clustering of indiv nodes
# ==============================

node_embeddings_sum = defaultdict(list)

# Iterate over the DataFrame and sum embeddings for each node
for _, row in df_graph.iterrows():
    node_embeddings_sum[row['node_1']].append(row['node_1_embedding'])
    node_embeddings_sum[row['node_2']].append(row['node_2_embedding'])

# Average the embeddings for each node
node_embeddings_avg = {node: np.mean(embeddings, axis=0) for node, embeddings in node_embeddings_sum.items()}

# Convert embeddings to a list for clustering
nodes, embeddings = zip(*node_embeddings_avg.items())
embeddings_list = list(embeddings)

# Perform K-Means clustering
n_clusters = 5  # Adjust this based on your data
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(embeddings_list)

# Map nodes to their clusters
node_cluster_mapping = dict(zip(nodes, clusters))

# function to find similar nodes and edges based on query
def find_similar_nodes_and_edges(query, df, top_n=2):
    print("start find_similar_nodes_and_edges")
    query_embedding = OpenAIEmbeddings().embed_query(query)
    print("query_embedding: ", query)
    df['similarity'] = df.apply(lambda row: cosine_similarity(
        [row['node_1_embedding']] + [row['node_2_embedding']] + [row['edge_embedding']],
        [query_embedding])[0][0], axis=1)

    # Retrieve the top_n similar rows and include chunk_ids
    similar_nodes_df = df.nlargest(top_n, 'similarity')
    return similar_nodes_df[['node_1', 'node_2', 'edge', 'chunk_id', 'similarity']]

# function to get text_chunks from chunk Id based on similar nodes with cosine similarity
def get_text_chunks(chunk_ids, chunks_dataframe):
    # Retrieve rows where chunk_id is in the given set of chunk_ids
    relevant_rows = chunks_dataframe[chunks_dataframe['chunk_id'].isin(chunk_ids)]

    # Filter out chunks containing specific strings
    filtered_chunks = relevant_rows[~relevant_rows['text'].str.contains("chunk contextual proximity|global contextual proximity", regex=True)]

    return filtered_chunks['text'].tolist()

# function to extract cosine scores for vector searched nodes based on query
def extract_cosine_scores(similar_nodes_df):
    cosine_scores = {}
    for _, row in similar_nodes_df.iterrows():
        nodes = [row['node_1'], row['node_2']]
        for node in nodes:
            cosine_scores[node] = max(row['similarity'], cosine_scores.get(node, 0))
    return cosine_scores

# function to apply Dijkstra on all node pairs from vector searched nodes
def find_all_shortest_paths(graph, nodes):
    all_paths_str = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            try:
                path = nx.shortest_path(graph, source=nodes[i], target=nodes[j], weight='weight')
                path_str = construct_path_string(path, graph)
                all_paths_str.append(path_str)
            except nx.NetworkXNoPath:
                continue  # If there's no path between a pair of nodes, just skip

    return all_paths_str

# function to make path_string for Dijkstra-paths
def construct_path_string(path, graph):
    path_str = ''
    for i in range(len(path)):
        node = path[i]
        if i > 0:
            prev_node = path[i - 1]
            edge_data = graph.get_edge_data(prev_node, node)
            edge_description = edge_data.get('title', 'Unnamed Edge')
            path_str += f" - {edge_description} - "
        path_str += f"{node}"
    return path_str

# function to apply traveling salesman problem (variation, no circle) on vector-searched nodes
def find_tsp_path(graph, start_node, nodes):
    path = [start_node]
    current_node = start_node
    visited = set([start_node])

    while len(visited) < len(nodes):
        neighbors = [(n, graph[current_node][n]['weight']) for n in graph.neighbors(current_node) if n not in visited]
        if not neighbors:
            break  # No unvisited neighbors, break the loop

        # Sort neighbors by weight
        neighbors.sort(key=lambda x: x[1])

        # Choose the nearest neighbor
        next_node = neighbors[0][0]
        path.append(next_node)
        visited.add(next_node)
        current_node = next_node

    return path

# function to get context_triplets from topic clustering, top n results based on weight
def get_context_triplets(graph, context_nodes, top_n):
    context_triplets = []

    for u, v, data in graph.edges(data=True):
        if u in context_nodes and v in context_nodes:
            edge_description = data.get('title', 'Unnamed Edge')
            edge_weight = data.get('weight', 0)
            triplet = (u, edge_description, v, edge_weight)
            context_triplets.append(triplet)

    # Sort the triplets based on weight and get the top-n results
    context_triplets.sort(key=lambda x: x[3], reverse=True)  # Sorting by edge weight
    return context_triplets[:top_n]

# function to get context_triplets from topic clustering, no weights, all of them
def get_context_triplets_no_weights(graph, context_nodes):
    context_triplets = []

    for u, v, data in graph.edges(data=True):
        if u in context_nodes and v in context_nodes:
            edge_description = data.get('title', 'Unnamed Edge')
            triplet = (u, edge_description, v)
            context_triplets.append(triplet)

    return context_triplets

# function for similarity_search for text
def find_similar_text_chunks(query, vectorstore, k=2):
    similar_chunks = vectorstore.similarity_search_with_score(query, k)
    return [(doc.page_content, score) for doc, score in similar_chunks]

# function to apply custom algo to find top_nodes
def generate_context(graph, df_graph, query, tf_idf_scores, top_n=5):
    # Initialize the scaler
    scaler = MinMaxScaler()
    combined_scores = {}  # Initialize the dictionary

    #   Centrality Measures: Highlights structurally significant nodes in the graph with PageRank

    # Calculate centrality measures in the graph
    centrality = nx.pagerank(graph)  # or other centrality measures
    # Normalize the centrality scores
    centrality_values = list(centrality.values())
    centrality_normalized = scaler.fit_transform(np.array(centrality_values).reshape(-1, 1)).flatten()

    # Map back the normalized scores to nodes
    normalized_centrality = dict(zip(centrality.keys(), centrality_normalized))

    #   TF-IDF Scores: Signifies the uniqueness and importance of nodes in various contexts

    tf_idf_values = list(single_tf_idf_scores.values())
    tf_idf_normalized = scaler.fit_transform(np.array(tf_idf_values).reshape(-1, 1)).flatten()

    # Map back the normalized scores to nodes
    normalized_tf_idf_scores = dict(zip(single_tf_idf_scores.keys(), tf_idf_normalized))

    #    Cosine Similarity: Determines the relevance of nodes to the query

    # Find similar nodes and edges
    similar_nodes_df = find_similar_nodes_and_edges(query, df_graph, top_n=10)

    # Extract cosine similarity scores for individual nodes
    cosine_scores = extract_cosine_scores(similar_nodes_df)

    #   combined scores for top nodes
    for node in graph.nodes():
        cosine_score = cosine_scores.get(node, 0)
        tf_idf_score = normalized_tf_idf_scores.get(node, 0)
        centrality_score = normalized_centrality.get(node, 0)
        combined_scores[node] = cosine_score + tf_idf_score + centrality_score

    # Select top scoring nodes
    top_nodes = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_n]
    return top_nodes

# function to get text chunks for top nodes
def get_text_chunks_for_top_nodes(top_nodes, df_graph, chunks_dataframe):
    # Retrieve chunk IDs associated with top nodes
    chunk_ids = set()
    for node in top_nodes:
        rows = df_graph[(df_graph['node_1'].isin(node)) | (df_graph['node_2'].isin(node))]
        for _, row in rows.iterrows():
            chunk_ids.update(row['chunk_id'].split(','))

    # Fetch corresponding text chunks
    return get_text_chunks(chunk_ids, chunks_dataframe)

# function to Generate Response with OpenAI API for fact extract
def generate_fact_extract(query):

    print("start fact extract")
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
    print("Got response from LLM: ")
    # Convert the string representation of a list to an actual list
    response_list = ast.literal_eval(response)


    return response_list

#function to determine the system_prompt based on the first entry in the list returned from fact_extract
def get_system_prompt(fact_extract):
    print("start get_system_prompt")
    query_type = fact_extract[0]
    print(fact_extract[0])
    print("query_type: ", query_type)
    if query_type.lower() == 'sachverhalt':
        return system_prompt_sachverhalt
    elif query_type.lower() == 'frage':
        return system_prompt_frage
    elif query_type.lower() == 'suche':
        return system_prompt_begriffliche_suche
    elif query_type.lower() == 'erklärung':
        return system_prompt_erklärung
    elif query_type.lower() == 'gesetze':
        return system_prompt_gesetze
    elif query_type.lower() == 'urteile':
        return system_prompt_urteile
    else:
        return system_prompt_sachverhalt

# function to Generate Response with OpenAI API
def generate_response(query, fact_extract):

    # ==============================
    # find similar_nodes with cos similarity
    # ==============================

    # Using list comprehension
    similar_nodes = [find_similar_nodes_and_edges(query, df_graph, top_n=2) for query in fact_extract]

    # Format graph information of similar nodes
    graph_info = "\n".join(
        [f"Node: {row['node_1']} ist mit Node: {row['node_2']} verbunden, durch die Edge: '{row['edge']}'" 
        for df in similar_nodes for _, row in df.iterrows()]
    )

    # Retrieve unique chunk_ids from the similar nodes
    chunk_ids = set()
    for df in similar_nodes:
        for _, row in df.iterrows():
            chunk_ids.update(row['chunk_id'].split(','))

    # Fetch the corresponding text chunks with the filtering applied
    cosine_nodes_chunks = get_text_chunks(chunk_ids, chunks_dataframe)
    # Outside the f-string
    cosine_nodes_chunks_str = '\n'.join(cosine_nodes_chunks)
    #print("chunk_id chunks for similar_nodes: ", cosine_nodes_chunks_str)

    # ==============================
    # get TSP(Variant) between vector searched nodes
    # ==============================

    # Get the list of similar nodes
    similar_nodes_list = [row['node_1'] for df in similar_nodes for _, row in df.iterrows()]

    # Find the TSP path
    tsp_path = find_tsp_path(graph, similar_nodes_list[0], similar_nodes_list)

    # Construct the string representation of the TSP path
    tsp_path_str = ''
    for i in range(len(tsp_path)):
        node = tsp_path[i]
        if i > 0:
            prev_node = tsp_path[i - 1]
            edge_data = graph.get_edge_data(prev_node, node)
            edge_description = edge_data['title']  # Adjust key as needed
            tsp_path_str += f" - {edge_description} - "
        tsp_path_str += f"{node}"
    #print(f"shortest path (tsp) that connects all nodes: {tsp_path_str}")

    # ==============================
    # get shortest_paths between all similar_nodes pairs (Dijkstra)
    # ==============================
    #toDo: evaluate relevance of Dijkstra approach
    # Find all shortest paths
    shortest_paths_strings = find_all_shortest_paths(graph, similar_nodes_list)

    # ==============================
    # get contextual_ nodes with tf-idf balanced with weights for similar_nodes
    # ==============================

    # Initialize a string to hold contextual node information
    contextual_nodes_info = ""

    #tf-idf neighbors with balanced weight between tf-idf-score and weight
    for df in similar_nodes:
        for index, row in df.iterrows():
            node = row['node_1']
            # Retrieve the top neighbors based on edge weight
            top_nodes_edges = get_top_contextual_nodes_edges(graph, node, top_n=2)

            contextual_nodes_info += f"Node: {node}\n"
            for neighbor, weight in top_nodes_edges:
                tf_idf_score = tf_idf_scores.get(node, {}).get(neighbor, 0)
                # Combine the scores: Adjust the '0.5' factors to tweak the balance
                combined_score = 0.5 * tf_idf_score + 0.4 * weight

                edge_data = graph.get_edge_data(node, neighbor) or {}
                edge_title = edge_data.get('title', '')
                contextual_info = f" - Contextual Neighbor: {neighbor}, Edge: {edge_title}, Combined Score: {combined_score}\n"
                contextual_nodes_info += contextual_info

    #print(index, "top nodes: ", contextual_nodes_info, "\n")

    # ==============================
    # topic clustering of nodes
    # ==============================

    # Get the list of similar nodes
    similar_nodes_list = [row['node_1'] for df in similar_nodes for _, row in df.iterrows()]

    # Find the clusters of the similar nodes
    similar_nodes_clusters = {node_cluster_mapping[node] for node in similar_nodes_list if node in node_cluster_mapping}

    # Get all nodes from these clusters
    context_nodes = [node for node, cluster in node_cluster_mapping.items() if cluster in similar_nodes_clusters]
    #print("topic clusters: ", context_nodes)

    # Get the top-n topic clustered triplets (weighted)
    context_triplets = get_context_triplets(graph, context_nodes, top_n=3)
    # or context_triplets without the weights
    context_triplets_all = get_context_triplets_no_weights(graph, context_nodes)

    # Format these triplets as a string
    topic_triplets_str = "\n".join([f"{u} - {edge} - {v}" for u, edge, v, _ in context_triplets])
    #print("context triplets_with_weights: ", triplets_str)

    # ==============================
    # custom algo for top nodes
    # ==============================

    # Generate context using the custom algorithm
    top_nodes = [generate_context(graph, df_graph, query, tf_idf_scores, top_n=1) for query in fact_extract]
    # Flattening the list of lists
    top_nodes = [node for sublist in top_nodes for node in sublist]
    print("top_nodes: ", top_nodes)

    # Extract node-edge-node triplets for these top nodes
    top_nodes_triplets = get_context_triplets(graph, top_nodes, top_n = 2)
    #print("top nodes_triplets: ", top_nodes_triplets)
    top_nodes_triplets_str = "\n".join([f"{u} - {edge} - {v}" for u, edge, v, _ in top_nodes_triplets])
    print("top nodes_triplets: ", top_nodes_triplets)

    # get top chunks based on top_nodes chunks_id
    #top_nodes_chunks = get_text_chunks_for_top_nodes(top_nodes, df_graph, chunks_dataframe)

    # Formatting the top nodes chunks for display
    #top_nodes_chunks_str = '\n'.join(top_nodes_chunks)

    # ==============================
    # get contextual_ nodes with tf-idf balanced with weights for top_nodes
    # ==============================

    # Initialize a string to hold contextual node information
    contextual_nodes_info_top_nodes = ""

    # tf-idf neighbors with balanced weight between tf-idf-score and weight
    for node in top_nodes:
        # Retrieve the top neighbors based on edge weight
        top_nodes_edges = get_top_contextual_nodes_edges(graph, node, top_n=2)

        contextual_nodes_info_top_nodes += f"Node: {node}\n"
        for neighbor, weight in top_nodes_edges:
            tf_idf_score = tf_idf_scores.get(node, {}).get(neighbor, 0)
            # Combine the scores: Adjust the '0.5' factors to tweak the balance
            combined_score = 0.5 * tf_idf_score + 0.4 * weight

            edge_data = graph.get_edge_data(node, neighbor) or {}
            edge_title = edge_data.get('title', '')
            contextual_info = f" - Contextual Neighbor: {neighbor}, Edge: {edge_title}, Combined Score: {combined_score}\n"
            contextual_nodes_info_top_nodes += contextual_info

    #print("top nodes contextual info with tfidf- balanced with weight: ", contextual_nodes_info_top_nodes)

    # ==============================
    # get text chunks with cos similarity
    # ==============================

    # Get referenced text chunks
    similar_chunk_for_query = find_similar_text_chunks(query, text_vectorstore, k=1)
    similar_chunks = [find_similar_text_chunks(query, text_vectorstore, k=1) for query in fact_extract]
    print("Apply similarity search for text chunks")

    # Format the referenced text chunks
    references_query = "\n".join([f"{content[0]} (Score: {content[1] if len(content) > 1 else 'N/A'})" for content in similar_chunk_for_query])
    references = "\n".join([f"{content[0]} (Score: {content[1] if len(content) > 1 else 'N/A'})" for content in similar_chunks])
    #print(references)

    # ==============================
    # combine all context
    # ==============================

    #toDO: for KG Contexts - use the triplets str or just topic clustered nodes for example?
    # Combine context from KG, references, and contextual nodes
    combined_context = (
                        f"User Query:\n {query}"
                        f"\n Facts:\n {fact_extract}"
                        f"\n\n Korpus aus Knowledge Graph (A) und referenzierten Textstellen (B): \n"
                        f"A. Kontext vom Knowledge Graphen - \n"
                        f"1. cosine similarity Vector-searched Nodes und Edges vom Knowledge Graphen :\n{graph_info}\n\n"
                        f"2. Kontextuelle Nachbarn der vorherigen Nodes :\n{contextual_nodes_info}\n\n"
                        f"3. Kürzester Weg durch den Graphen, der alle vector-searched nodes verbindet: \n{tsp_path_str}\n\n"
                        f"4. Topic Clusters im graphen basierend auf query:\n{topic_triplets_str}\n\n"
                        f"5. weitere top_nodes für das query:\n{top_nodes_triplets_str}\n\n"
                        f"6. kontextuelle Nachbarn der weiteren top_nodes:\n {contextual_nodes_info_top_nodes}\n\n"
                        f"B. Kontext durch referenzierte Textstellen aus dem Corpus:\n{references_query}\n{references}\n"
                        )
    #print(combined_context)

    # ==============================
    # make LLM call
    # ==============================

    system_prompt = get_system_prompt(fact_extract)

    print("start LLM call")
    client = OpenAI(api_key=api_key)
    #gpt-4-1106-preview #gpt-3.5-turbo-1106
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system",
             "content":
                 f"{system_prompt}"
             },
            {"role": "user", "content": combined_context}
        ]
    )

    # Generate response using combined context
    #response = completion.choices[0].message
    response = completion.choices[0].message.content #only the content
    print("Got response from LLM: ")
    # Combine context and response
    full_interaction = (
                        f"\n\nUser Query:\n {query}"
                        f"\n Facts:\n {fact_extract}"
                        f"\n Response des LLM:\n {response}"
                        f"\n\n Korpus aus Knowledge Graph (A) und referenzierten Textstellen aus dem Grundkorpus (B): \n"
                        f"A. Kontext vom Knowledge Graphen - \n"
                        f"1. cosine similarity Vector-searched Nodes und Edges vom Knowledge Graphen :\n{graph_info}\n\n"
                        f"2. Kontextuelle Nachbarn der vorherigen Nodes :\n{contextual_nodes_info}\n\n"
                        f"3. Kürzester Weg durch den Graphen, der alle vector-searched nodes verbindet: \n{tsp_path_str}\n\n"
                        f"4. Topic Clusters im graphen basierend auf query:\n{topic_triplets_str}\n\n"
                        f"5. weitere top_nodes für das query:\n{top_nodes_triplets_str}\n\n"
                        f"6. kontextuelle Nachbarn der weiteren top_nodes:\n {contextual_nodes_info_top_nodes}\n\n"
                        f"B. Kontext durch referenzierte Textstellen aus dem Corpus:\n{references_query}\n{references}\n"
                        )
    return full_interaction

# function to apply Custom HI Splitter Function
def custom_hi_text_splitter(directory):
    chunks = []

    for file_path in glob.glob(f"{directory}/*.txt"):
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            text = file.read()

            # Find all occurrences of 'HI' followed by numbers
            hi_indices = [match.start() for match in re.finditer(r'HI\d+', text)]
            if hi_indices:
                # Add the first chunk from the beginning of the text to the first 'HI' index
                chunks.append(text[:hi_indices[0]])

                # Add all subsequent chunks from one 'HI' index to the next
                for start, end in zip(hi_indices, hi_indices[1:] + [None]):
                    chunks.append(text[start:end])

    return chunks

#splitting by HI
#texts = custom_hi_text_splitter("./data_input")

#chunk embedding with langchain loader
loader = DirectoryLoader("./data_input")
documents = loader.load()
# Text Splitting using Langchain
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
docs = splitter.split_documents(documents)
#toDO: split by children (512) but also embed parent HI for retrieval
# maybe have a dictionary with parent children HI for -> embedded children to later retrieve

# Check if the text vector store exists, if not create a new one
if os.path.exists(text_vectorstore_path):
    print("Loading existing vectorstore for text chunks")
    text_vectorstore = Chroma(persist_directory=text_vectorstore_path, embedding_function=OpenAIEmbeddings())
else:
    print("Creating vectorstore for text chunks")
    #vectorstore for Directory Loader
    text_vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory=text_vectorstore_path)
    
    #vectorstore for custom splitting
    #text_vectorstore = Chroma.from_texts(texts, OpenAIEmbeddings(), persist_directory=text_vectorstore_path)


#toDO: embeddings get created every time - would that storage even work?
# Check if the knowledge graph vector store exists, if so load it, otherwise create a new one
if os.path.exists(kg_vectorstore_path):
    print("Loading existing vectorstore for KG")
    kg_vectorstore = Chroma(persist_directory=kg_vectorstore_path, embedding_function=OpenAIEmbeddings())
else:
    print("Creating vectorstore for KG")
    # Create and initialize Chroma vectorstore with node and edge embeddings
    kg_vectorstore = Chroma.from_texts(
        df_graph['node_1'].tolist() + df_graph['node_2'].tolist() + df_graph['edge'].tolist(), OpenAIEmbeddings(),
        persist_directory=kg_vectorstore_path)

# Main Loop for User Interaction
chat_history = []
query = None

while True:
    query = input("Prompt: ")
    if query.lower() in ['quit', 'q', 'exit']:
        break
    
    fact_extract = generate_fact_extract(query)
    print(fact_extract)
    full_interaction = generate_response(query, fact_extract)
    print(full_interaction)
    chat_history.append(full_interaction)

# At the end of your chat session
filename = './data_output/history/chat_history.txt'
with open(filename, 'a') as file:
    file.write("\n\n".join(chat_history) + "\n\n")