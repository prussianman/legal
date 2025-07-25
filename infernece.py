import tensorflow as tf
from transformers import AutoTokenizer, TFBertForSequenceClassification
import numpy as np

# --- 1. Load Saved Model and Tokenizer ---
# Replace this with the actual path to your saved student model
model_path = "./distill_trained_model/" 

print(f"Loading model from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(model_path)

# Optional: Define your class labels if you know them.
# The order should match what the model was trained on. 
# For example: 0 -> Negative, 1 -> Neutral, 2 -> Positive
class_labels = ["Negative", "Neutral", "Positive"] 

# --- 2. Prepare Input Text ---
text_samples = [
    "The customer service was outstanding and the product exceeded all my expectations.",
    "I am incredibly disappointed with this purchase and will be returning it immediately.",
    "The package is scheduled for delivery on Friday, according to the tracking information.",
    "Oh, fantastic. Another software update that slows down my computer. Just what I needed."
]

print("\n--- Starting Inference ---")

# --- 3. Tokenize the Text ---
# The tokenizer converts text to numbers (token IDs).
# padding=True ensures all sentences have the same length.
# truncation=True cuts off sentences that are too long.
# return_tensors="tf" returns TensorFlow tensors.
inputs = tokenizer(text_samples, padding=True, truncation=True, return_tensors="tf")

# --- 4. Make Prediction ---
# The model outputs a dictionary containing the 'logits'.
outputs = model(inputs)
logits = outputs.logits

# --- 5. Process Output ---
# Apply softmax to convert logits to probabilities
probabilities = tf.nn.softmax(logits, axis=1)

# Get the predicted class index for each sample
predicted_class_ids = np.argmax(probabilities, axis=1)

print("\n--- Inference Results ---")
for i, text in enumerate(text_samples):
    predicted_label = class_labels[predicted_class_ids[i]]
    confidence_score = probabilities[i][predicted_class_ids[i]]
    
    print(f"\nText: '{text}'")
    print(f"--> Predicted Label: {predicted_label} (Confidence: {confidence_score:.4f})")

