from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import pandas as pd

# --- Your Azure AI Search Credentials ---
service_endpoint = "https://your-search-service-name.search.windows.net"
index_name = "my-chunk-index"  # The name of the index you just defined
api_key = "your-admin-api-key" # Get this from the "Keys" section in the Azure portal

# Assume 'final_df' is the DataFrame created by your script in the Canvas
# For example:
# final_df = embed_sub_chunks_from_df(original_df, text_column='text_chunk', openai_client=client)

# 1. Convert the DataFrame to a list of dictionaries
# This is the format the SDK expects.
documents_to_upload = final_df.to_dict('records')

# 2. Instantiate the SearchClient
credential = AzureKeyCredential(api_key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

# 3. Upload the documents in batches
try:
    result = search_client.upload_documents(documents=documents_to_upload)
    print(f"Successfully uploaded {len(documents_to_upload)} documents.")
except Exception as e:
    print(f"An error occurred during upload: {e}")


#########

# Assume 'final_df' is the DataFrame created by your script

# --- FIX: Add this line before converting to dictionary ---
# This converts each numpy array in the 'embedding' column into a standard Python list.
final_df['embedding'] = final_df['embedding'].apply(list)

# 1. Convert the DataFrame to a list of dictionaries
# Now, this will work correctly because all data types are JSON-compatible.
documents_to_upload = final_df.to_dict('records')

# 2. Instantiate the SearchClient
credential = AzureKeyCredential(api_key)
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

# 3. Upload the documents in batches
try:
    result = search_client.upload_documents(documents=documents_to_upload)
    print(f"Successfully uploaded {len(documents_to_upload)} documents.")
except Exception as e:
    print(f"An error occurred during upload: {e}")
