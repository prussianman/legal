import numpy as np
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# --- Helper function to convert NumPy types to native Python types ---
def convert_numpy_types(obj):
    """
    Recursively converts NumPy types in a dictionary or list to native Python types
    to make them JSON serializable.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

# --- Your Azure AI Search Credentials ---
service_endpoint = "https://your-search-service-name.search.windows.net"
index_name = "my-chunk-index"
api_key = "your-admin-api-key"

# Assume 'final_df' is the DataFrame you have prepared
# ...

# 1. Convert the DataFrame to a list of dictionaries
documents_to_upload = final_df.to_dict('records')

# 2. **FIX: Apply the robust conversion function to the entire list**
# This will iterate through every value in every dictionary and fix any NumPy types.
cleaned_documents = [convert_numpy_types(doc) for doc in documents_to_upload]


# 3. Instantiate the SearchClient
credential = AzureKeyCredential(api_key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

# 4. Upload the cleaned documents
try:
    # Use the 'cleaned_documents' list for the upload
    result = search_client.upload_documents(documents=cleaned_documents)
    print(f"Successfully uploaded {len(cleaned_documents)} documents.")
except Exception as e:
    print(f"An error occurred during upload: {e}")
