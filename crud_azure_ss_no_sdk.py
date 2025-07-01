import requests
import json
import uuid # Used to generate a unique ID for the new document

# --- Configuration: Replace with your Azure AI Search details ---
service_endpoint = "https://your-search-service-name.search.windows.net"
index_name = "sentence-child-index" # The name of your index
api_key = "your-admin-api-key"       # An admin key for your search service
api_version = "2024-07-01"           # The latest stable REST API version

# --- Construct the URL for the Index Documents operation ---
# The URL format is: {endpoint}/indexes/{index-name}/docs/index
url = f"{service_endpoint}/indexes/{index_name}/docs/index?api-version={api_version}"

# --- Set up the required headers ---
# The api-key is required for authentication.
# The Content-Type tells the service we are sending JSON data.
headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}

# --- 1. Prepare the Document Payload ---
# This is the document you want to add to your index.
# The field names ('sentence_id', 'title', 'embedding', etc.) MUST exactly match
# the field names you defined in your Azure AI Search index schema.

new_document = {
    # @search.action tells Azure what to do with this document.
    # "upload": Adds a new document. Fails if a document with the same key exists.
    # "merge": Updates an existing document. Fails if the key doesn't exist.
    # "mergeOrUpload": A combination - updates if the key exists, adds if it's new. (Recommended)
    # "delete": Removes the document with this key from the index.
    "@search.action": "mergeOrUpload",

    # --- Your Fields ---
    # The 'key' field (sentence_id) must be a unique string.
    "sentence_id": str(uuid.uuid4()), # Generate a new unique ID for this sentence
    "parent_chunk_id": "parent_doc_abc_123",
    "title": "Case of the Missing Signature",
    "summary": "This case revolves around a contract dispute where the defendant claims the final page was never signed.",
    "plaintiff": ["Global Innovations Inc."],
    "defendant": ["John Smith"],
    "court_decision": "The motion to dismiss was denied by the presiding judge.",
    "compensation": 50000.00,
    "type_of_legal_case": ["Contract Dispute", "Corporate Law"],
    "footnotes": ["See appendix A for signature examples.", "Refer to testimony from 2024-06-15."],
    "decision_date": "2025-07-01T12:00:00Z", # Must be in ISO 8601 format
    "country": "USA",
    "jurisdiction": "State of California",
    "sentiment": 1,
    "sentence_text": "The court found that sufficient evidence exists to proceed to trial.",
    
    # The embedding vector must be a list of floats.
    # This is a truncated example with 1536 dimensions.
    "embedding": [-0.01, 0.02, -0.03] + [0.0] * 1533 
}

# --- 2. Wrap the document(s) in the required 'value' array ---
# The API expects a JSON object with a single key "value",
# which contains a list of one or more documents to process.
payload = {
    "value": [
        new_document
        # You can add more document dictionaries here to upload in a batch
        # , { "@search.action": "mergeOrUpload", "sentence_id": "...", ... }
    ]
}

# --- 3. Make the POST request to the API ---
try:
    print("Sending document to Azure AI Search...")
    
    # Use requests.post to send the data.
    # json.dumps(payload) is not strictly necessary as the `json=` parameter handles it,
    # but it's good practice to be explicit.
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # This will raise an exception if the request returned an error status code (4xx or 5xx)
    response.raise_for_status()

    # The response for a successful batch upload will be 200 OK if all documents succeeded,
    # or 207 Multi-Status if some succeeded and some failed.
    # We check the status of the first document in the response.
    response_json = response.json()
    if response_json['value'][0]['status']:
        print("Successfully uploaded document.")
        print(f"Status code for document 1: {response_json['value'][0]['statusCode']}")
    else:
        print("Upload failed for document 1:")
        print(f"Error: {response_json['value'][0]['errorMessage']}")

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    print(f"Response body: {response.text}")
except requests.exceptions.RequestException as req_err:
    # This will catch network-level errors, including SSL/TLS issues
    print(f"A network request error occurred: {req_err}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

