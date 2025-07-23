import os
import openai
import tiktoken
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math

# --- 1. Azure OpenAI Configuration ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2024-03-01-preview" # Ensure this API version supports logprobs for chat completions
AZURE_OPENAI_DEPLOYMENT_NAME = "your-gpt-deployment-name" # e.g., "gpt-35-turbo-0125" or "gpt-4"

# Validate environment variables are set
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
    raise ValueError("Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME environment variables.")

client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# --- 2. Define your Classification Classes and their Token IDs ---
class_labels = ["Positive", "Negative", "Neutral"]

# Use tiktoken to get token IDs for your labels.
# Use the encoding that matches your deployed GPT model (e.g., cl100k_base for gpt-3.5-turbo/gpt-4).
enc = tiktoken.encoding_for_model("gpt-3.5-turbo") # Or "gpt-4" if that's your model

# Map your class labels to their corresponding single token IDs.
# LLMs often tokenize words with a leading space differently than words at the start of a sentence.
# We'll include both possibilities to be robust.
# Store as a dictionary {token_id: label_index} for quick lookup
label_to_id_mapping = {}
id_to_label_index = {} # Map token_id to its index in class_labels list

for i, label in enumerate(class_labels):
    # Try tokenizing with a leading space (common for words in a sequence)
    tokens_with_space = enc.encode(" " + label)
    if tokens_with_space and len(tokens_with_space) == 1:
        token_id = tokens_with_space[0]
        label_to_id_mapping[label] = token_id
        id_to_label_index[token_id] = i
        continue # Found a single token, move to next label

    # If not single token with space, try without leading space (for words at start of output)
    tokens_without_space = enc.encode(label)
    if tokens_without_space and len(tokens_without_space) == 1:
        token_id = tokens_without_space[0]
        label_to_id_mapping[label] = token_id
        id_to_label_index[token_id] = i
        continue

    print(f"Warning: Class label '{label}' does not tokenize to a single token. This might complicate logprob extraction.")
    # For multi-token labels, logprob extraction becomes much harder and less reliable for single-token output.
    # You might need to change your prompt strategy to ask for the class index (e.g., 0, 1, 2)
    # or handle multi-token logprobs which is significantly more complex.

if len(id_to_label_index) != len(class_labels):
    raise ValueError("Not all class labels could be mapped to unique single token IDs. Please review your class labels or tokenization strategy.")

print(f"Mapped Class Labels to Token IDs for lookup: {label_to_id_mapping}")
print(f"Mapped Token IDs to Class Indices: {id_to_label_index}")


# --- 3. Function to Get Soft Logits from Azure OpenAI LLM ---
def get_llm_soft_logits(text, class_labels, distillation_temperature=1.0):
    """
    Gets softened log probabilities (logits) for specified class labels from an Azure OpenAI LLM.
    Args:
        text (str): The input text to classify.
        class_labels (list): A list of possible classification labels (e.g., ["Positive", "Negative"]).
        distillation_temperature (float): The temperature to apply to the teacher's logits for softening.
                                          Higher values produce softer probabilities.
    Returns:
        torch.Tensor: Softened log probabilities for the class_labels (shape: [num_classes]).
                      Returns None if logprobs cannot be extracted or processed.
    """
    messages = [
        {"role": "system", "content": f"You are a text classifier. Classify the following text into one of these exact categories: {', '.join(class_labels)}. Output only the category name."},
        {"role": "user", "content": text}
    ]

    # Use logit_bias to significantly increase the likelihood of only our class tokens
    # and strongly suppress all other tokens for the first generated token.
    # This helps ensure the model outputs one of our desired class labels.
    logit_bias = {}
    # Bias for our target class tokens
    for label_id in id_to_label_index.keys():
        logit_bias[str(label_id)] = 100 # High positive bias

    # To be extremely strict, you could try to bias *against* all other tokens.
    # However, this is usually not necessary with a max_tokens=1 and a strong positive bias.
    # If you have a list of ALL possible token IDs (vocab_size), you could:
    # for token_id in range(enc.n_vocab):
    #     if token_id not in id_to_label_index:
    #         logit_bias[str(token_id)] = -100 # Strongly negative bias

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=1,  # We only need the first token's logprobs
            logprobs=True, # Request log probabilities
            top_logprobs=len(id_to_label_index), # Request only as many top logprobs as we have classes.
                                                 # Or even 5-10 if that's sufficient, it reduces payload size.
            logit_bias=logit_bias # Apply logit bias
        )

        if not response.choices or not response.choices[0].logprobs or not response.choices[0].logprobs.content:
            print(f"Warning: No logprobs content received for text: '{text}'. Response: {response}")
            return None

        # The logprobs are returned for each token position. We care about the first one (index 0).
        first_token_logprob_info = response.choices[0].logprobs.content[0]

        # Initialize a tensor to store logits for our specific classes
        teacher_raw_logits = torch.full((len(class_labels),), float('-inf'), dtype=torch.float32)

        # Iterate through the top_logprobs for the first token
        # These are usually the tokens with the highest probabilities, hopefully including our classes
        for top_lp in first_token_logprob_info.top_logprobs:
            # Re-tokenize the top_lp.token to get its ID, accounting for potential leading spaces
            # The API often includes a leading space if the token is not the first in the completion
            token_to_match = top_lp.token
            token_id_without_space = enc.encode(token_to_match.replace(' ', ''))
            token_id_with_space = enc.encode(token_to_match)

            matched_id = None
            if token_id_without_space and len(token_id_without_space) == 1 and token_id_without_space[0] in id_to_label_index:
                matched_id = token_id_without_space[0]
            elif token_id_with_space and len(token_id_with_space) == 1 and token_id_with_space[0] in id_to_label_index:
                matched_id = token_id_with_space[0]

            if matched_id is not None:
                class_index = id_to_label_index[matched_id]
                teacher_raw_logits[class_index] = torch.tensor(top_lp.logprob, dtype=torch.float32)

        # Apply temperature to soften the logits for distillation
        # F.log_softmax is good here because KLDiv expects log-probabilities for input
        soft_log_probabilities = F.log_softmax(teacher_raw_logits / distillation_temperature, dim=-1)

        return soft_log_probabilities

    except openai.APIStatusError as e:
        print(f"OpenAI API Error (Status {e.status_code}): {e.response}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- 4. Example Usage: Generate Soft Labels for a Sample Dataset ---
sample_texts = [
    "This product is amazing, I love it!",
    "It was okay, nothing special.",
    "Absolutely terrible experience, completely ruined my day.",
    "Neutral on this one, neither good nor bad.",
    "The service was fast and efficient!"
]

# Store generated soft labels and corresponding hard labels (if available)
data_for_bert_training = []

print("--- Generating soft labels with GPT (Teacher Model) ---")
for text in tqdm(sample_texts, desc="Processing texts"):
    soft_log_probs_from_gpt = get_llm_soft_logits(text, class_labels, distillation_temperature=5.0) # Use T > 1 for more softening
    if soft_log_probs_from_gpt is not None:
        # For demonstration, let's assume a dummy hard label for now.
        # In a real scenario, you'd use your true labels if available.
        # Or, if no true labels, you could derive a pseudo-label from GPT's prediction:
        # dummy_hard_label = torch.argmax(soft_log_probs_from_gpt).item()
        dummy_hard_label = np.random.randint(0, len(class_labels)) # Replace with actual logic

        data_for_bert_training.append({
            "text": text,
            "teacher_soft_log_probs": soft_log_probs_from_gpt,
            "hard_label": dummy_hard_label
        })
    else:
        print(f"Skipping text due to failed logprob extraction: '{text}'")


print("\n--- Sample Generated Data for BERT Training ---")
for i, item in enumerate(data_for_bert_training):
    if i >= 2: break # Just show first two
    print(f"Text: {item['text']}")
    print(f"Teacher Soft Log Probs (log_softmaxed): {item['teacher_soft_log_probs']}")
    print(f"Hard Label (for student): {item['hard_label']}")
    print("-" * 20)

# --- 5. Define BERT Student Model and Training Setup ---

class BertStudentDataset(Dataset):
    def __init__(self, data, bert_tokenizer, max_len):
        self.data = data
        self.tokenizer = bert_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten() if 'token_type_ids' in encoding else torch.empty(0), # Handle models that don't use token_type_ids
            'teacher_soft_log_probs': item['teacher_soft_log_probs'], # This is already log_softmaxed
            'labels': torch.tensor(item['hard_label'], dtype=torch.long)
        }

# Initialize BERT tokenizer and model
bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=len(class_labels))

# Set up training parameters
max_seq_len = 128
batch_size = 8
num_epochs = 5
learning_rate = 2e-5 # For BERT fine-tuning

# Create DataLoader
dataset = BertStudentDataset(data_for_bert_training, bert_tokenizer, max_seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = AdamW(bert_model.parameters(), lr=learning_rate)

# --- 6. Define Custom Knowledge Distillation Loss ---
def distillation_loss_fn(student_logits, teacher_soft_log_probs, student_labels, T, alpha):
    """
    Combines hard label loss and distillation loss.
    Args:
        student_logits (torch.Tensor): Raw logits from the BERT student model.
        teacher_soft_log_probs (torch.Tensor): Softened log probabilities from the GPT teacher model.
                                               (Assumed to be already log_softmaxed with T).
        student_labels (torch.Tensor): True hard labels for the student.
        T (float): Distillation temperature.
        alpha (float): Weight for the hard label loss. (1-alpha) for distillation loss.
    Returns:
        torch.Tensor: Total combined loss.
    """
    # Hard label loss (Cross-Entropy)
    # F.cross_entropy expects raw logits for the first arg and integer labels for the second.
    hard_loss = F.cross_entropy(student_logits, student_labels)

    # Distillation loss (KL Divergence)
    # F.kl_div expects log-probabilities for the input and probabilities for the target.
    # We provide log_softmax(student_logits / T) for input, and teacher_soft_log_probs (already log-softmaxed) for target.
    # The T*T factor is from Hinton's original paper.
    distillation_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        teacher_soft_log_probs.exp(), # Convert teacher's log-probs back to probs for KLDiv target
        reduction='batchmean' # Using batchmean ensures sum over batch and mean over samples
    ) * (T * T)

    # Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * distillation_loss
    return total_loss

# --- 7. BERT Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

distillation_temperature = 5.0 # T for distillation loss (should match or be similar to T used for teacher)
alpha_weight = 0.5 # Weight for hard label loss vs. distillation loss

print("\n--- Starting BERT Student Model Training ---")
for epoch in range(num_epochs):
    bert_model.train()
    total_train_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_type_ids'].to(device) if batch['token_type_ids'].numel() > 0 else None
        teacher_soft_log_probs = batch['teacher_soft_log_probs'].to(device)
        student_labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Get logits from BERT student model
        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids # Pass None if not used by model
        )
        student_raw_logits = outputs.logits

        # Calculate combined distillation loss
        loss = distillation_loss_fn(
            student_raw_logits,
            teacher_soft_log_probs,
            student_labels,
            distillation_temperature,
            alpha_weight
        )

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished. Average Train Loss: {avg_train_loss:.4f}")

print("\nBERT Student Model Training Complete!")
