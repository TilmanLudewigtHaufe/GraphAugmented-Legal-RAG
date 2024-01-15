import os
import json
import networkx as nx
import logging
from py2neo import Graph, Node, Relationship
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info('Environment variables loaded.')

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# function to create a NetworkX graph from JSON data
def load_graph_from_json(json_file_path):
    logger.info(f'Loading graph from JSON file: {json_file_path}')
    try:
        with open(json_file_path, 'r', encoding='utf-8-sig') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from the file: {json_file_path}. Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while reading the file: {e}")
        return None

    G = nx.Graph()
    for node in json_data['nodes']:
        G.add_node(node['id'])
    for link in json_data['links']:
        G.add_edge(link['source'], link['target'], weight=link['weight'], title=link['title'])
    logger.info('Graph loaded successfully.')
    return G

# function to create a Neo4j graph from NetworkX graph
def convert_to_neo4j(graph):
    logger.info('Converting NetworkX graph to Neo4j graph.')
    neo4j_graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Create nodes
    for node in graph.nodes:
        neo4j_node = Node("Node", id=node)
        neo4j_graph.create(neo4j_node)
    logger.info('Nodes created in Neo4j graph.')

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
    logger.info('Relationships created in Neo4j graph.')
    logger.info('Conversion to Neo4j graph completed.')
    return neo4j_graph

# Load your NetworkX graph
graph = load_graph_from_json('./data_output/graph_data.json')

# Convert NetworkX graph to Neo4j graph
neo4j_graph = convert_to_neo4j(graph)
logger.info('Process completed.')