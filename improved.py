import os
import openai
import tiktoken
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import List, Dict, Any

# --- 1. Configuration Dataclass ---
@dataclass
class DistillationConfig:
    """Holds all configuration parameters for the distillation process."""
    # Azure OpenAI Settings
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
    api_version: str = "2024-03-01-preview"
    gpt_deployment_name: str = "your-gpt-4-deployment" # e.g., "gpt-4"

    # Model and Tokenizer Settings
    class_labels: List[str] = field(default_factory=lambda: ["Positive", "Negative", "Neutral"])
    student_model_name: str = "bert-base-uncased"
    gpt_tokenizer_model: str = "gpt-4"
    max_seq_len: int = 128

    # Training Hyperparameters
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    teacher_temperature: float = 4.0 # Temperature for softening teacher logits
    student_temperature: float = 4.0 # Temperature for student logits in distillation loss
    alpha: float = 0.3 # Weight for hard label loss (CrossEntropy). (1-alpha) is for distillation loss.
    validation_split_size: float = 0.2

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Helper Functions ---
def get_label_token_maps(cfg: DistillationConfig) -> (Dict[str, int], Dict[int, int]):
    """Creates mappings between class labels and their single token IDs."""
    enc = tiktoken.encoding_for_model(cfg.gpt_tokenizer_model)
    label_to_id = {}
    id_to_label_index = {}
    print("Mapping class labels to single token IDs...")
    for i, label in enumerate(cfg.class_labels):
        # Check for token with a leading space, which is common
        tokens_with_space = enc.encode(" " + label)
        if len(tokens_with_space) == 1:
            token_id = tokens_with_space[0]
            label_to_id[label] = token_id
            id_to_label_index[token_id] = i
            continue
        # Fallback to token without a leading space
        tokens_without_space = enc.encode(label)
        if len(tokens_without_space) == 1:
            token_id = tokens_without_space[0]
            label_to_id[label] = token_id
            id_to_label_index[token_id] = i
            continue
        print(f"Warning: Label '{label}' does not map to a single token. Consider changing the label.")

    if len(id_to_label_index) != len(cfg.class_labels):
        raise ValueError("Could not map all class labels to unique single token IDs.")
    
    print(f"Token ID to Class Index Mapping: {id_to_label_index}")
    return label_to_id, id_to_label_index

def generate_teacher_data(texts: List[str], cfg: DistillationConfig, id_to_label_index: Dict) -> List[Dict[str, Any]]:
    """Generates soft logits from the teacher LLM for a list of texts."""
    client = openai.AzureOpenAI(
        azure_endpoint=cfg.azure_endpoint, api_key=cfg.api_key, api_version=cfg.api_version
    )
    logit_bias = {str(token_id): 100 for token_id in id_to_label_index.keys()}
    
    teacher_dataset = []
    print("--- Generating soft labels with GPT Teacher Model ---")
    for text in tqdm(texts, desc="Querying Teacher"):
        try:
            messages = [
                {"role": "system", "content": f"You are a text classifier. Classify the text into one of these exact categories: {', '.join(cfg.class_labels)}. Output only the category name."},
                {"role": "user", "content": text},
            ]
            response = client.chat.completions.create(
                model=cfg.gpt_deployment_name,
                messages=messages,
                max_tokens=1,
                temperature=0.0, # Low temperature for predictability
                logprobs=True,
                top_logprobs=10, # Get enough top logprobs to find our labels
                logit_bias=logit_bias,
            )
            
            logprob_content = response.choices[0].logprobs.content[0]
            raw_logits = torch.full((len(cfg.class_labels),), -1e9, dtype=torch.float32)

            for top_lp in logprob_content.top_logprobs:
                # The API returns string tokens, we must re-encode to find the ID
                # This is a known complexity when working with logprobs.
                token_id = tiktoken.encoding_for_model(cfg.gpt_tokenizer_model).encode(top_lp.token)
                if len(token_id) == 1 and token_id[0] in id_to_label_index:
                    class_index = id_to_label_index[token_id[0]]
                    raw_logits[class_index] = top_lp.logprob

            # Use argmax on the raw logits from the teacher as the "hard label" for the student
            hard_label = torch.argmax(raw_logits).item()

            teacher_dataset.append({
                "text": text,
                "teacher_logits": raw_logits,
                "hard_label": hard_label,
            })
        except Exception as e:
            print(f"Skipping text '{text[:50]}...' due to error: {e}")

    return teacher_dataset

# --- 3. Dataset & Loss Function ---
class StudentDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
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
            'teacher_logits': item['teacher_logits'],
            'labels': torch.tensor(item['hard_label'], dtype=torch.long)
        }

def distillation_loss(student_logits, teacher_logits, hard_labels, T, alpha):
    """Calculates the total distillation loss."""
    # Distillation Loss (KL Divergence)
    # Measures how the softened student predictions match the softened teacher predictions.
    soft_teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    soft_student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    distill_loss = F.kl_div(soft_student_log_probs, soft_teacher_probs, reduction='batchmean') * (T * T)

    # Student Loss (Cross-Entropy with hard labels)
    hard_loss = F.cross_entropy(student_logits, hard_labels)

    # Combine losses
    return alpha * hard_loss + (1.0 - alpha) * distill_loss

# --- 4. Training & Evaluation Functions ---
def train_epoch(model, dataloader, optimizer, cfg):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(cfg.device)
        attention_mask = batch['attention_mask'].to(cfg.device)
        teacher_logits = batch['teacher_logits'].to(cfg.device)
        labels = batch['labels'].to(cfg.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = outputs.logits
        
        loss = distillation_loss(student_logits, teacher_logits, labels, cfg.student_temperature, cfg.alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, cfg):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            teacher_logits = batch['teacher_logits'].to(cfg.device)
            labels = batch['labels'].to(cfg.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = outputs.logits
            
            # Use the same loss function for consistency, but you could also just use CrossEntropy
            loss = distillation_loss(student_logits, teacher_logits, labels, cfg.student_temperature, cfg.alpha)
            total_loss += loss.item()
            
            preds = torch.argmax(student_logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
            
    accuracy = correct_predictions / total_predictions
    return total_loss / len(dataloader), accuracy

# --- 5. Main Execution ---
if __name__ == "__main__":
    # Initialize configuration
    config = DistillationConfig()
    
    if not all([config.azure_endpoint, config.api_key, config.gpt_deployment_name]):
        raise ValueError("Please set Azure OpenAI environment variables.")

    # Get token mappings
    _, id_to_label_index_map = get_label_token_maps(config)
    
    # Define sample dataset
    sample_texts_for_distillation = [
        "This product is amazing, I love it!",
        "It was okay, nothing special.",
        "Absolutely terrible experience, completely ruined my day.",
        "Neutral on this one, neither good nor bad.",
        "The service was fast and efficient!",
        "I have no strong feelings about this whatsoever.",
        "A true masterpiece of modern cinema.",
        "Completely and utterly disappointing from start to finish."
    ]
    
    # Generate teacher data (or load from cache if you have it)
    teacher_data = generate_teacher_data(sample_texts_for_distillation, config, id_to_label_index_map)
    
    if not teacher_data:
        raise SystemExit("No data generated from teacher model. Exiting.")

    # Split data into training and validation sets
    train_data, val_data = train_test_split(
        teacher_data,
        test_size=config.validation_split_size,
        random_state=42
    )

    # Initialize student model and tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    student_model = BertForSequenceClassification.from_pretrained(
        config.student_model_name, num_labels=len(config.class_labels)
    ).to(config.device)
    
    # Create Datasets and DataLoaders
    train_dataset = StudentDataset(train_data, student_tokenizer, config.max_seq_len)
    val_dataset = StudentDataset(val_data, student_tokenizer, config.max_seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Optimizer
    optimizer = AdamW(student_model.parameters(), lr=config.learning_rate)

    # --- Training & Evaluation Loop ---
    print("\n--- Starting BERT Student Model Training ---")
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")
        avg_train_loss = train_epoch(student_model, train_dataloader, optimizer, config)
        avg_val_loss, val_accuracy = evaluate(student_model, val_dataloader, config)
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"\tAverage Train Loss: {avg_train_loss:.4f}")
        print(f"\tAverage Validation Loss: {avg_val_loss:.4f}")
        print(f"\tValidation Accuracy: {val_accuracy:.4f}")

    print("\nBERT Student Model Training Complete!")
