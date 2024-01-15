import os
import logging
import sys
import openai
from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.graph_stores import Neo4jGraphStore
from llama_index.query_engine import KnowledgeGraphQueryEngine
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import LangChainLLM
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up your environment variables
logger.info('Setting up environment variables.')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
AURA_INSTANCEID = os.getenv('AURA_INSTANCEID')
api_key = os.getenv("OPENAI_API_KEY")

# Configure OpenAI settings
logger.info('Configuring OpenAI settings.')
llm = OpenAI(
    api_key=openai.api_key,
    temperature=0,
    model="gpt-3.5-turbo-1106"
)

embedding = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key=openai.api_key
)

# Create a service context for LlamaIndex
logger.info('Creating service context for LlamaIndex.')
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding
)

# Set up the Neo4j Graph Store
logger.info('Setting up the Neo4j Graph Store.')
neo4j_store = Neo4jGraphStore(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    aura_instance_id=AURA_INSTANCEID
)
storage_context = StorageContext.from_defaults(graph_store=neo4j_store)

# Initialize the KnowledgeGraphQueryEngine with the Neo4j Graph Store
logger.info('Initializing the KnowledgeGraphQueryEngine with the Neo4j Graph Store.')
query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)

# Example query
query = input("Enter your query: ")
logger.info(f'User query: {query}')

# Generate and print the Cypher query
cypher_query = query_engine.generate_query(query)
formatted_cypher_query = cypher_query.replace("WHERE", "\n  WHERE").replace("RETURN", "\nRETURN")
print("Generated Cypher Query:\n")
print(formatted_cypher_query)

# Execute the query
logger.info('Executing the query.')
response = query_engine.query(query)
logger.info('Query executed successfully.')

# Print the response
print("\nResponse:")
print(response)
logger.info('Process completed.')
