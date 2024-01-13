import pandas as pd
import numpy as np
import os
import glob
import re
import json
import requests
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from pathlib import Path
import random
import networkx as nx
import seaborn as sns
from pyvis.network import Network
import json
from yachalk import chalk
import uuid
from dotenv import load_dotenv
import logging
import io
import pickle

#semantic chunking
from llama_index.llama_pack import download_llama_pack
from llama_index import SimpleDirectoryReader
download_llama_pack(
    "SemanticChunkingQueryEnginePack",
    "./semantic_chunking_pack",
    skip_load=True,
    # leave the below line commented out if using the notebook on main
    # llama_hub_url="https://raw.githubusercontent.com/run-llama/llama-hub/jerry/add_semantic_chunker/llama_hub"
)
from semantic_chunking_pack.base import SemanticChunker
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings import OpenAIEmbedding


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()
logging.info(".env loaded")

api_key = os.getenv("OPENAI_API_KEY")

# function with OpenAI API integration for generating responses
def generate_openai(prompt, model_name="gpt-3.5-turbo-1106"):
    try:
        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system",
                 "content":
                     "You are a network graph maker who extracts terms and their relations from a given german context."
                     "The purpose of the Knowledge graph will be to structure german legal areas for later query purposes. "
                     "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
                     "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
                     "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
                     "\tTerms may include object, entity, location, organization, person, \n"
                     "\tcondition, acronym, documents, service, concept, etc.\n"
                     "\tTerms should be as atomistic as possible\n\n"
                     "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
                     "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
                     "\tTerms can be related to many other terms\n\n"
                     "Thought 3: Find out the relation between each such related pair of terms. \n\n"
                     "Format your output as a list of json. Each element of the list contains a pair of terms"
                     "and the relation between them, like the following: \n"
                     "[\n"
                     "   {\n"
                     '       "node_1": "A concept from extracted ontology",\n'
                     '       "node_2": "A related concept from extracted ontology",\n'
                     '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
                     "   }, {...}\n"
                     "]\n"
                     "Ensure that each node is distinct, with no duplicates or minor variations of another node."
                 },
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# function for generating graph data from a given text input
def graphPrompt(input: str, metadata={}, model="gpt-3.5-turbo-1106"):

    USER_PROMPT = f"context: {input} \n\n output: "
    full_prompt = USER_PROMPT

    response = generate_openai(full_prompt, model_name=model)

    try:
        if response:
            # Extract content from the ChatCompletionMessage object
            content = response.content

            # Remove triple backticks and any other non-JSON formatting
            content_cleaned = content.replace('```json\n', '').replace('\n```', '').strip()

            # Parse the cleaned content as JSON
            result = json.loads(content_cleaned)

            # Add additional metadata to each item in the result
            result = [dict(item, **metadata) for item in result]
        else:
            raise ValueError("Invalid response format or content missing")
    except json.JSONDecodeError as e:
        print("\n\nERROR ### JSON Parsing Error: ", e, "\nResponse Content: ", response, "\n\n")
        result = None
    except Exception as e:
        print("\n\nERROR ### Unexpected Error: ", e, "\nResponse: ", content, "\n\n")
        result = None

    return result

# function for converting a collection of document chunks into a structured pandas DataFrame
def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

# function used to transform text data from a DataFrame into a format suitable for graph-related operations
def df2Graph(dataframe: pd.DataFrame, model=None) -> list:
    # Counter for chunks sent to OpenAI
    chunks_sent_to_openai = 0

    # dataframe.reset_index(inplace=True)
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()

    # Increment the counter for each chunk processed
    chunks_sent_to_openai += len(concept_list)
    print(f"Total chunks sent to OpenAI: {chunks_sent_to_openai}")

    return concept_list

# function intended to process graph data and convert it into a pandas DataFrame format
def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())

    return graph_dataframe

# function to calculate contextual_proximity based on chunk alone
def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    # Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    # Group and count edges
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "chunk contextual proximity"
    return dfg2

# function to calculate global contextual proximity over the whole corpus into a third dfg
def global_contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting global_contextual_proximity function")

    # Melt the dataframe into a long format
    logging.info("Melting DataFrame")
    dfg_long = pd.melt(
        df, id_vars=[], value_vars=["node_1", "node_2"], var_name="variable", value_name="node"
    ).drop(columns=["variable"])

    logging.info(f"DataFrame after melting: {dfg_long.head()}")

    # Create a self-join using the node column
    logging.info("Performing self-join")
    dfg_wide = pd.merge(dfg_long.rename(columns={"node": "node_1"}),
                        dfg_long.rename(columns={"node": "node_2"}),
                        left_on="node_1",
                        right_on="node_2")

    logging.info(f"DataFrame after self-join: {dfg_wide.head()}")

    # Drop self loops where an edge starts and ends on the same node
    logging.info("Removing self loops")
    dfg_wide = dfg_wide[dfg_wide["node_1"] != dfg_wide["node_2"]]

    logging.info(f"DataFrame after removing self loops: {dfg_wide.head()}")

    # Group and count the number of co-occurrences of the node pairs
    logging.info("Grouping and counting co-occurrences")
    dfg2 = (
        dfg_wide.groupby(["node_1", "node_2"])
        .size()
        .reset_index(name="count")
    )

    logging.info(f"DataFrame after grouping and counting: {dfg2.head()}")

    # Filter out pairs with only one co-occurrence
    logging.info("Filtering out pairs with only one co-occurrence")
    dfg2 = dfg2[dfg2["count"] > 1]

    logging.info(f"DataFrame after filtering: {dfg2.head()}")

    # Add edge type
    dfg2["edge"] = "global contextual proximity"

    logging.info("Completed global_contextual_proximity function")

    return dfg2

# function to apply HI splitter
def custom_hi_text_splitter(directory):
    pages = []

    for file_path in glob.glob(f"{directory}/*.txt"):
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            text = file.read()

            # Find all occurrences of 'HI' followed by numbers
            hi_indices = [match.start() for match in re.finditer(r'HI\d+', text)]
            if hi_indices:
                logging.info(f"chunked {hi_indices}")
                # Add the first chunk as a Document object
                pages.append(Document(page_content=text[:hi_indices[0]]))

                # Add all subsequent chunks as Document objects
                for start, end in zip(hi_indices, hi_indices[1:] + [None]):
                    pages.append(Document(page_content=text[start:end]))

    return pages

# ====================
# Splitting into document objects with normal splitting techniques
# ====================

# Input data directory
data_dir = "data_input"
inputdirectory = Path(f"./data_input")

# This is where the output csv files will be written
outputdirectory = Path(f"./data_output")

# Directory Loader for langchain splitter
# loader = DirectoryLoader(inputdirectory, show_progress=True)
# documents = loader.load()
logging.info("Starting document loading...")

#toDO: either use HI splitting or normal splitting -> to get more context in this script a bigger chunk is better
# either HI splitting or normal langchain splitting
#pages = custom_hi_text_splitter("./data_input") #HI splitted chunks

# Text Splitting using Langchain
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=20000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )
#pages = splitter.split_documents(documents) #normal langchain splitter

# ====================
# semantic chunking for splitting - experimental
# ==================== 
# Define the path to the directory
dir_path = "data_input"

# Initialize an empty string to store the contents of all files
all_content = ""

# Iterate over all .txt files in the directory
for file_path in glob.glob(os.path.join(dir_path, "*.txt")):
    # Open the file and read its content
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()

    # Append the content to the all_content string
    all_content += content

embed_model = OpenAIEmbedding()
splitter = SemanticChunker(
    buffer_size=1, breakpoint_percentile_threshold=85, embed_model=embed_model
)
# Process Documents
docs = splitter.split_text(all_content)
pages = [Document(page_content=chunk) for chunk in docs]
# ====================

print("Number of chunks = ", len(pages))
#print(pages[0].page_content)
logging.info("Document splitted")

# Convert documents to DataFrame
df = documents2Dataframe(pages)
print(df.shape)
df.head()

# Regenerate the graph with LLM if graph.csv not already made
# Define the path to the graph.csv file
graph_csv_path = "data_output/graph.csv"

# Check if the graph.csv file exists
if os.path.exists(graph_csv_path):
    regenerate = False
else:
    regenerate = True

if regenerate:
    # Using df2Graph to generate graph data from the DataFrame
    logging.info("Generate graph data")
    graph_data = df2Graph(df, model='gpt-3.5-turbo-1106')  # Using the OpenAI model name
    logging.info("Finished API Calls")

    # toDO: Preprocess and Optimize Nodes and Edges
    #optimized_graph_data = optimize_graph_data(graph_data)

    logging.info("Writing graph data and converting to DataFrame")
    dfg1 = graph2Df(graph_data)  # Converting the graph data to a DataFrame

    # Creating the output directory if it doesn't exist
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)

    # Writing the graph data and the original DataFrame to CSV files
    dfg1.to_csv(outputdirectory / "graph.csv", sep="|", index=False)
    df.to_csv(outputdirectory / "chunks.csv", sep="|", index=False)
    logging.info("Starting writing graph data and DataFrame to CSV files")
else:
    logging.info("write from graph.csv")
    # Reading the graph data from a CSV file if not regenerating
    dfg1 = pd.read_csv(outputdirectory / "graph.csv", sep="|")


dfg1.replace("", np.nan, inplace=True)
dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
dfg1['count'] = 4
# initializing the weight of the direct-relation to 4
# will assign the weight of 1 when later the contextual proximity will be calculated
print(dfg1.shape)
#print(dfg1.head())

dfg2 = contextual_proximity(dfg1)
#print(dfg2.tail())
logging.info("Melt Dataframe into list of Nodes")


#toDO: concatenate third Dataframe (global contextual similarity) with only the first dataframe
global_context_df = global_contextual_proximity(dfg1)
# Concatenate the three DataFrames / maybe only conc the dfg1 and the global?
dfg_combined = pd.concat([dfg1, dfg2, global_context_df])
# Group by node pairs. For 'chunk_id' and 'edge', concatenate unique values; sum the 'count'
dfg_final = (
    dfg_combined.groupby(["node_1", "node_2"])
    .agg({
        "chunk_id": lambda x: ','.join(set([str(i) for i in x if pd.notna(i)])),  # Convert to string and concatenate unique, non-null chunk_ids
        "edge": lambda x: ','.join(set([str(i) for i in x if pd.notna(i)])),      # Convert to string and concatenate unique, non-null edge descriptions
        'count': 'sum'                                                            # Sum the weights (counts)
    })
    .reset_index()
)

#toDO: weight adjustement for glaobal proximit to 0.5 - evaluate
# Update the weights for global contextual proximity to 0.5
dfg_final['count'] = np.where(dfg_final['edge'] == 'global contextual proximity', 0.5 * dfg_final['count'], dfg_final['count'])

#concatenate only dfg1 and dfg2 (contextual proximity within chunks)
dfg = pd.concat([dfg1, dfg2], axis=0)
dfg = (
    dfg.groupby(["node_1", "node_2"])
    .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
    .reset_index()
)
dfg
print(dfg.head())

#toDO: either use "dfg" for concatenad dfg1 and dfg2 or "dfg_final" for global_context_df as well
nodes = pd.concat([dfg_final['node_1'], dfg_final['node_2']], axis=0).unique()
nodes.shape

G = nx.Graph()

## Add nodes to the graph
for node in nodes:
    G.add_node(
        str(node)
    )

## Add edges to the graph
for index, row in dfg.iterrows():
    G.add_edge(
        str(row["node_1"]),
        str(row["node_2"]),
        title=row["edge"],
        weight=row['count']/4,
        chunk_id=row['chunk_id'] #toDO check if this actually worked
    )
logging.info("adding nodes and edges to graph")

# Convert the NetworkX graph to a dictionary for JSON
graph_data = nx.node_link_data(G)

# Specify the file path where you want to save the JSON
json_file_path = "./data_output/graph_data.json"

# Write the graph data to a JSON file
with open(json_file_path, "w", encoding="utf-8-sig") as json_file:
    json.dump(graph_data, json_file)
logging.info("Graph Data as Json")

#toDO: redundant - delete
#Save as pickle file

# Define the file path for saving the pickled graph
pickle_file_path = "./data_output/graph_data.pkl"

# Save the graph to a file using pickle
with open(pickle_file_path, "wb") as pickle_file:
    pickle.dump(G, pickle_file)

logging.info("Graph saved as pickle")

# detect communities with GirvanNewman Algo
communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
communities = sorted(map(sorted, next_level_communities))
print("Number of Communities = ", len(communities))
#print(communities)

palette = "hls"

# function to add colors to communities and make another dataframe
def colors2Community(communities) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors
logging.info("adding colors and making another dataframe")

colors = colors2Community(communities)
colors
print(colors.head())

for index, row in colors.iterrows():
    G.nodes[row['node']]['group'] = row['group']
    G.nodes[row['node']]['color'] = row['color']
    G.nodes[row['node']]['size'] = G.degree[row['node']]

print("Number of nodes in G:", len(G.nodes()))
print("Number of edges in G:", len(G.edges()))

# Print node attributes for a few nodes to verify
for node in list(G.nodes)[:1]:  # Adjust the number of nodes to print as needed
    print(node, G.nodes[node])

# Define the output file path for the HTML content and write to
graph_output_directory = "./docs/index.html"

net = Network(
    notebook=False,
    bgcolor="#1a1a1a",
    cdn_resources="remote",
    height="900px",
    width="100%",
    select_menu=True,
    font_color="#cccccc",
    filter_menu=False,
)

net.from_nx(G)
# net.repulsion(node_distance=150, spring_length=400)
net.force_atlas_2based(central_gravity=0.015, gravity=-31)
# net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380) #alternative physics algo
net.show_buttons(filter_=["physics"])

html = net.generate_html()
with open(graph_output_directory, mode='w', encoding='utf-8-sig') as fp:
        fp.write(html)

#net.show(graph_output_directory, notebook=False)