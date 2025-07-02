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
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.stores import SimpleDocumentStore
from llama_index.storage.docstore.azure_blob import AzureBlobStorageKVStore
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
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")


# Your Azure OpenAI model deployment names
LLM_DEPLOYMENT_NAME = "gpt-4o"
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-large"

# --- Names for your Azure AI Search Indexes and Storage Container ---
CHILD_INDEX_NAME = "llama-child-chunks-vector"
PARENT_DOC_CONTAINER_NAME = "parent-doc-store" # Container for persistent parent docs

# Check for missing configuration
if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_STORAGE_CONNECTION_STRING]):
    raise ValueError(
        "One or more Azure credentials are not set. Please set the environment variables: "
        "AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_STORAGE_CONNECTION_STRING"
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

# --- 3. Custom Parent-Child Retriever ---
class ParentChildRetriever(BaseRetriever):
    """
    A custom retriever that implements the parent-child retrieval strategy.
    It first fetches smaller child nodes from a vector store and then retrieves
    their larger parent documents from a document store.
    """
    def __init__(self, child_retriever: BaseRetriever, docstore):
        """
        Args:
            child_retriever (BaseRetriever): A retriever for the child nodes (e.g., VectorIndexRetriever).
            docstore: The document store containing the parent documents.
        """
        self.child_retriever = child_retriever
        self.docstore = docstore
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        The core retrieval logic.
        """
        print(f"--> Retrieving child nodes for query: '{query_bundle.query_str}'")
        # 1. Retrieve the top-k most relevant child nodes
        child_nodes_with_score = self.child_retriever.retrieve(query_bundle)
        
        child_node_ids = [n.node.node_id for n in child_nodes_with_score]
        print(f"--> Found {len(child_node_ids)} relevant child nodes: {child_node_ids}")

        # 2. Get the parent document IDs from the child nodes
        # LlamaIndex automatically links child nodes to their parent via `ref_doc_id`
        parent_doc_ids = []
        for node in child_nodes_with_score:
            if node.node.ref_doc_id and node.node.ref_doc_id not in parent_doc_ids:
                parent_doc_ids.append(node.node.ref_doc_id)

        if not parent_doc_ids:
            return []
            
        print(f"--> De-duplicated to {len(parent_doc_ids)} unique parent document IDs: {parent_doc_ids}")

        # 3. Retrieve the full parent documents from the docstore
        parent_docs = self.docstore.get_documents(parent_doc_ids)
        
        # We need to return NodeWithScore objects. We'll use the highest score
        # from a child node belonging to that parent as the parent's score.
        retrieved_nodes = []
        for parent_doc in parent_docs:
            # Find the best score among child nodes for this parent
            best_score = 0.0
            for child_node in child_nodes_with_score:
                if child_node.node.ref_doc_id == parent_doc.doc_id:
                    best_score = max(best_score, child_node.score)
            retrieved_nodes.append(NodeWithScore(node=parent_doc, score=best_score))

        print(f"--> Retrieved {len(retrieved_nodes)} full parent documents.")
        return retrieved_nodes

# --- 4. Main Execution Logic ---
def main():
    """
    Main function to set up indexes, ingest data, and run queries.
    """
    print("--- Starting Parent-Child RAG Demo ---")

    # --- 4.1. Prepare Sample Data ---
    # In a real application, you would load your documents here.
    # We'll create some sample documents for this demo.
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
    print(f"\n--- Prepared {len(docs)} sample parent documents ---")

    # --- 4.2. Setup Azure AI Search Vector Store for Child Chunks ---
    # This is where the small, embedded chunks will be stored and searched.
    credential = AzureKeyCredential(AZURE_SEARCH_KEY)
    
    # Define the schema for the child chunk index
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

    child_vector_store = AzureAISearchVectorStore(
        search_or_index_client=AZURE_SEARCH_ENDPOINT,
        credential=credential,
        index_name=CHILD_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
    )
    print(f"\n--- Azure AI Search Vector Store for child index '{CHILD_INDEX_NAME}' is ready ---")

    # --- 4.3. Ingest Data with Persistent Docstore ---
    # For a persistent solution, we store parent documents in Azure Blob Storage.
    print(f"\n--- Setting up persistent docstore in Azure Blob Storage container '{PARENT_DOC_CONTAINER_NAME}' ---")

    # Create a key-value store backed by Azure Blob Storage
    azure_kv_store = AzureBlobStorageKVStore(
        connection_string=AZURE_STORAGE_CONNECTION_STRING,
        container_name=PARENT_DOC_CONTAINER_NAME,
    )

    # Create a docstore that uses the Azure KV store for persistence
    persistent_docstore = SimpleDocumentStore.from_kvstore(azure_kv_store)
    
    # We now combine our persistent docstore with our vector store in the StorageContext
    storage_context = StorageContext.from_defaults(
        vector_store=child_vector_store,
        docstore=persistent_docstore
    )
    
    # The node parser creates the child nodes from the parent documents.
    node_parser = SentenceSplitter(chunk_size=64, chunk_overlap=4)
    
    # The service context bundles the components LlamaIndex uses for ingestion and querying
    service_context = ServiceContext.from_defaults(node_parser=node_parser)

    # This single command now does the heavy lifting with persistence:
    # 1. Parses parent docs into child nodes.
    # 2. Generates embeddings for child nodes.
    # 3. Stores child nodes in the Azure AI Search child index.
    # 4. Stores parent docs in the Azure Blob Storage container.
    child_index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, service_context=service_context,
        show_progress=True
    )
    print(f"\n--- Ingestion complete. Child nodes in Azure AI Search, Parent docs in Azure Blob Storage. ---")
    
    # Give Azure a moment to finish indexing
    time.sleep(5)

    # --- 4.4. Create the Query Engine with our Custom Retriever ---
    # 1. Create a standard retriever for the child index
    child_retriever = child_index.as_retriever(similarity_top_k=3)
    
    # 2. Get the docstore where the parent docs are held (now persistent)
    docstore = child_index.docstore

    # 3. Instantiate our custom retriever
    parent_child_retriever = ParentChildRetriever(child_retriever, docstore)

    # 4. Create the query engine that uses our custom retriever
    from llama_index.core.query_engine import RetrieverQueryEngine

    query_engine = RetrieverQueryEngine.from_args(
        retriever=parent_child_retriever,
    )
    print("\n--- Parent-Child Query Engine is ready ---")

    # --- 4.5. Run a Query ---
    query = "What was the most powerful part of the Saturn V rocket?"
    print(f"\n--- Executing Query: '{query}' ---")
    
    response = query_engine.query(query)

    print("\n\n--- LLM Final Answer ---")
    print(response)
    print("\n--- Retrieved Parent Source Documents ---")
    for node in response.source_nodes:
        print(f"Source: {node.metadata['source']}, Score: {node.score:.4f}")
        print(f"Content: {node.text[:150]}...") # Print snippet of the parent doc
        print("-" * 20)

if __name__ == "__main__":
    main()
