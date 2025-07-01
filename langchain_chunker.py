import os
import pandas as pd
from openai import OpenAI
import nltk
from nltk.tokenize import sent_tokenize
# New imports for token-based splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# --- Configuration ---
# It's best practice to set your API key as an environment variable.
# You can get a key from https://platform.openai.com/api-keys
# In your terminal: export OPENAI_API_KEY='your-api-key-here'
# If you can't set an environment variable, you can uncomment the next line:
# os.environ["OPENAI_API_KEY"] = "sk-..."

# --- NLTK Setup ---
# The sentence tokenizer needs the 'punkt' dataset.
# This downloads it if you don't have it already.
try:
    sent_tokenize("test sentence.")
except LookupError:
    print("NLTK 'punkt' dataset not found. Downloading...")
    nltk.download('punkt')
    print("Download complete.")

def get_openai_client():
    """Initializes and returns the OpenAI client, checking for the API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set the key to use the OpenAI API."
        )
    return OpenAI(api_key=api_key)

def get_embedding(text: str, client: OpenAI, model: str = "text-embedding-3-small") -> list[float]:
    """
    Generates a vector embedding for a given text sentence by calling the OpenAI API.
    """
    normalized_text = ' '.join(text.strip().split())
    try:
        response = client.embeddings.create(input=[normalized_text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return []

def split_text_by_tokens(text: str, chunk_size: int = 1024, chunk_overlap_ratio: float = 0.1) -> list[str]:
    """
    Splits a large text into smaller sub-chunks based on token count, but only if the
    original text exceeds the chunk_size.

    Args:
        text: The text to split.
        chunk_size: The maximum number of tokens for each chunk.
        chunk_overlap_ratio: The percentage of overlap between chunks.

    Returns:
        A list of smaller text sub-chunks, or a list with the original text if it's
        within the size limit.
    """
    # Get the tokenizer for the model we are likely using (e.g., text-embedding models)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = tiktoken.get_encoding("p50k_base") # Fallback for older models

    num_tokens = len(encoding.encode(text))

    # If the text is within the limit, no need to split. Return it as a single-item list.
    if num_tokens <= chunk_size:
        return [text]

    # If the text exceeds the limit, split it using Langchain's token-based splitter.
    print(f"Text with {num_tokens} tokens exceeds the {chunk_size} limit. Splitting...")
    
    # Calculate the actual overlap in tokens
    chunk_overlap = int(chunk_size * chunk_overlap_ratio)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(encoding.encode(x)), # Use token count for length
        is_separator_regex=False,
    )
    
    return text_splitter.split_text(text)


def process_chunk_row(row: pd.Series, text_column: str, openai_client: OpenAI) -> list[dict]:
    """
    (Helper Function) Takes a DataFrame row, splits very large text into sub-chunks if needed,
    then gets embeddings for each sentence, and combines them with metadata.
    """
    text_chunk = row.get(text_column, "")
    if not isinstance(text_chunk, str) or not text_chunk.strip():
        return []

    # Pre-process the large chunk into smaller, manageable sub-chunks using token logic
    sub_chunks = split_text_by_tokens(text_chunk)
    
    all_sentence_records = []
    metadata = row.drop(text_column).to_dict()

    for i, sub_chunk in enumerate(sub_chunks):
        sentences = sent_tokenize(sub_chunk)
        
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue
                
            embedding_vector = get_embedding(clean_sentence, openai_client)
            
            if embedding_vector:
                new_record = metadata.copy()
                # Add a sub_chunk_id to track which part of the original chunk it came from
                new_record['sub_chunk_id'] = i 
                new_record['sentence'] = clean_sentence
                new_record['embedding'] = embedding_vector
                all_sentence_records.append(new_record)
            
    return all_sentence_records

def embed_sentences_from_df(df: pd.DataFrame, text_column: str, openai_client: OpenAI) -> pd.DataFrame:
    """
    Applies the sentence embedding process to a DataFrame using a more modular approach.
    """
    processed_series = df.apply(
        lambda row: process_chunk_row(row, text_column, openai_client),
        axis=1
    )
    exploded_series = processed_series.explode().dropna()
    if exploded_series.empty:
        return pd.DataFrame()
    final_df = pd.DataFrame(exploded_series.tolist())
    return final_df


# --- Main Execution Example ---
if __name__ == "__main__":
    # 1. Create a sample DataFrame with metadata, including a very large chunk
    # This long_text will have well over 1024 tokens
    long_text = "This is a sample sentence for testing token-based splitting. " * 300
    long_text += "\n\nThis is a new paragraph. " * 300
    
    data = {
        'chunk_id': [101, 102, 103],
        'source_doc': ['report_alpha.pdf', 'report_alpha.pdf', 'large_report.pdf'],
        'text_chunk': [
            "The quick brown fox jumps over the lazy dog. This is the first sentence.", # Will not be split
            "A second chunk begins here. It also contains two sentences.", # Will not be split
            long_text # Example of a chunk that needs splitting
        ]
    }
    original_df = pd.DataFrame(data)
    print("--- Original DataFrame with Metadata ---")
    print(original_df)
    print("\n" + "="*40 + "\n")

    print("--- Processing DataFrame with Modular Function ---")
    
    try:
        # Initialize the client once
        client = get_openai_client()

        # 2. Call the main modular function
        final_df = embed_sentences_from_df(original_df, text_column='text_chunk', openai_client=client)

        # 3. Display the results
        if not final_df.empty:
            print(f"\nSuccessfully processed {len(original_df)} chunks into {len(final_df)} sentences.")
            
            print("\n--- New DataFrame with Metadata Preserved ---")
            # Reordering columns for clarity
            cols = ['chunk_id', 'sub_chunk_id', 'source_doc', 'sentence', 'embedding']
            # Ensure all desired columns are present before reordering
            final_cols = [col for col in cols if col in final_df.columns]
            print(final_df[final_cols].head().to_string())
            print("...")
            print(final_df[final_cols].tail().to_string())

        else:
            print("No data was processed. The output DataFrame is empty.")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
