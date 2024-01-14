import pandas as pd
import os
import json
import networkx as nx
import pandas as pd
import os
import json
import networkx as nx
from py2neo import Graph, Node, Relationship
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
AURA_INSTANCEID = os.getenv('AURA_INSTANCEID')
api_key = os.getenv("OPENAI_API_KEY")

# Set this to True if you want to write to Neo4j
WRITE_TO_NEO4J = os.getenv('WRITE_TO_NEO4J', 'false').lower() == 'true'


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

# function to create a Neo4j graph from NetworkX graph
def convert_to_neo4j(graph):
    neo4j_graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Create nodes
    for node in graph.nodes:
        neo4j_node = Node("Node", id=node)
        neo4j_graph.create(neo4j_node)

    # Create relationships
    for edge in graph.edges:
        source, target = edge
        weight = graph.edges[edge]['weight']
        title = graph.edges[edge]['title']
        source_node = neo4j_graph.nodes.match("Node", id=source).first()
        target_node = neo4j_graph.nodes.match("Node", id=target).first()
        if source_node and target_node:
            neo4j_relationship = Relationship(source_node, "CONNECTED_TO", target_node, weight=weight, title=title)
            neo4j_graph.create(neo4j_relationship)

    return neo4j_graph

def translate_query_to_cypher(prompt, model_name="gpt-3.5-turbo-1106"):
    try:
        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": system_prompt
                 },
                {"role": "user", "content": prompt}
            ]
        )

        cypher_query = completion.choices[0].message.content
        # Remove markdown formatting
        cypher_query = cypher_query.replace("```cypher\n", "").replace("```", "").strip()
        return cypher_query
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to execute a Cypher query in Neo4j and return the result
def execute_cypher_query(query):
    neo4j_graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    result = neo4j_graph.run(query)
    return result

# Load your NetworkX graph and DataFrame
graph = load_graph_from_json('./data_output/graph_data.json')
chunks_dataframe = load_chunks_dataframe('./data_output/graph_data.json')

# Convert NetworkX graph to Neo4j graph only if WRITE_TO_NEO4J is true
if WRITE_TO_NEO4J:
    neo4j_graph = convert_to_neo4j(graph)

system_prompt = """
Translate the following natural language query which is german into a cypher query:
return only the cypher query so that i can directly be used in neo4j.
The neo4j knowledge graph is about "kündigungsrecht im deutschen arbeitsrecht"
it deals with "voraussetzungen von personen- verhaltens- und betriebsbedingten kündigungen"
"""
prompt = input("Enter your query: ")
cypher_query = translate_query_to_cypher(prompt, model_name="gpt-4-1106-preview")
print("Cypher Query after Openai API: " , cypher_query)
# Example Cypher query (modify as needed)
query_results = execute_cypher_query(cypher_query)
print("Query results: " ,query_results)