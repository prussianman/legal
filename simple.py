import os
import logging
import sys
import time
from typing import List, Optional

# Set up logging to see the inner workings of LlamaIndex
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    Document,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
)
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential

# --- 1. Configuration ---
# Fill in your Azure credentials and deployment details.
# It's recommended to use environment variables for security.
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Your Azure OpenAI model deployment names
LLM_DEPLOYMENT_NAME = "gpt-4o"
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-large"

# --- Name for your Azure AI Search Index ---
CHILD_INDEX_NAME = "llama-child-chunks-vector"

# Check for missing configuration
if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY]):
    raise ValueError(
        "One or more Azure credentials are not set. Please set the environment variables: "
        "AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY"
    )

# --- 2. Setup LLM and Embedding Models ---
# Initialize the LLM and embedding models to be used by LlamaIndex
llm = AzureOpenAI(
    deployment_name=LLM_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

embed_model = AzureOpenAIEmbedding(
    deployment_name=EMBEDDING_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# Set the global settings for LlamaIndex
Settings.llm = llm
Settings.embed_model = embed_model

# --- 3. Main Execution Logic ---
def main():
    """
    Main function to set up the index, ingest data, and run queries.
    """
    print("--- Starting Simple RAG Demo ---")

    # --- 3.1. Prepare Sample Data ---
    # In a real application, you would load your documents here.
    docs = [
        Document(
            text=(
                "Project Apollo: A Retrospective. The Apollo program, also known as Project Apollo, "
                "was the third United States human spaceflight program carried out by NASA, which "
                "succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. "
                "The key challenge was developing the Saturn V rocket."
            ),
            metadata={"doc_id": "apollo-summary", "source": "NASA History Archives"},
        ),
        Document(
            text=(
                "Saturn V Rocket Specifications. The Saturn V was an American super heavy-lift launch "
                "vehicle developed by NASA under the Apollo program for human exploration of the Moon. "
                "It was a three-stage liquid-fueled rocket. The first stage was the most powerful, "
                "using five F-1 engines."
            ),
            metadata={"doc_id": "saturn-v-specs", "source": "Rocketry Engineering Journal"},
        ),
        Document(
            text=(
                "The F-1 Engine. The Rocketdyne F-1 is a gas-generator cycle rocket engine that was "
                "developed in the United States by Rocketdyne in the late 1950s and was used in the "
                "Saturn V rocket. Five F-1 engines were used in the S-IC first stage of each Saturn V, "
                "which launched the Apollo missions."
            ),
            metadata={"doc_id": "f1-engine-details", "source": "Rocketdyne Technical Brief"},
        ),
    ]
    print(f"\n--- Prepared {len(docs)} sample documents ---")

    # --- 3.2. Setup Azure AI Search Vector Store ---
    # This is where the small, embedded chunks will be stored and searched.
    credential = AzureKeyCredential(AZURE_SEARCH_KEY)
    
    # Define the schema for the chunk index
    embedding_dimensions = len(embed_model.get_text_embedding("test"))
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name="my-vector-profile",
        ),
        # Metadata fields that LlamaIndex uses
        SearchField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="ref_doc_id", type=SearchFieldDataType.String, filterable=True),
    ]
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-profile", algorithm_configuration_name="my-hnsw-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-config")],
    )

    vector_store = AzureAISearchVectorStore(
        search_or_index_client=AZURE_SEARCH_ENDPOINT,
        credential=credential,
        index_name=CHILD_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
    )
    print(f"\n--- Azure AI Search Vector Store for index '{CHILD_INDEX_NAME}' is ready ---")

    # --- 3.3. Ingest Data ---
    # The StorageContext defines where the data is stored.
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # The node parser creates the child nodes (chunks) from the documents.
    node_parser = SentenceSplitter(chunk_size=64, chunk_overlap=4)
    
    # The service context bundles the components LlamaIndex uses.
    service_context = ServiceContext.from_defaults(node_parser=node_parser)

    # This command parses docs, generates embeddings, and stores them in Azure AI Search.
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, service_context=service_context,
        show_progress=True
    )
    print(f"\n--- Ingestion complete. Chunks stored in Azure AI Search. ---")
    
    # Give Azure a moment to finish indexing
    time.sleep(5)

    # --- 3.4. Create a Simple Query Engine ---
    # This engine queries the vector store directly.
    query_engine = index.as_query_engine(similarity_top_k=3)
    print("\n--- Simple Query Engine is ready ---")

    # --- 3.5. Run a Query ---
    query = "What was the most powerful part of the Saturn V rocket?"
    print(f"\n--- Executing Query: '{query}' ---")
    
    response = query_engine.query(query)

    print("\n\n--- LLM Final Answer ---")
    print(response)
    print("\n--- Retrieved Source Chunks ---")
    for node in response.source_nodes:
        print(f"Source Document ID: {node.metadata['doc_id']}, Score: {node.score:.4f}")
        print(f"Chunk Content: {node.text}")
        print("-" * 20)

if __name__ == "__main__":
    main()
