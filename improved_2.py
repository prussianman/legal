@tf.function
def train_step(x_batch, y_batch, model, optimizer, loss_fns, metrics_map, temperature, alpha):
    """Performs a single, compiled training step."""
    kl_loss_fn, ce_loss_fn = loss_fns
    train_loss_metric = metrics_map['train_loss']

    with tf.GradientTape() as tape:
        student_logits = model(x_batch, training=True).logits
        teacher_logits = y_batch['teacher_logits']
        hard_labels = y_batch['hard_labels']

        soft_teacher_probs = tf.nn.softmax(teacher_logits / temperature)
        soft_student_probs = tf.nn.softmax(student_logits / temperature)
        distill_loss = kl_loss_fn(soft_teacher_probs, soft_student_probs) * (temperature**2)
        
        hard_loss = ce_loss_fn(hard_labels, student_logits)
        total_loss = alpha * hard_loss + (1.0 - alpha) * distill_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_metric.update_state(total_loss)

#####################

import numpy as np
import tensorflow as tf

# The logit string you have
logit_string = 'tensor([-inf,-0.2344,-1.234])'

# --- 1. Parse the string ---
cleaned_string = logit_string.replace('tensor(', '').replace(')', '').replace('[', '').replace(']', '')
string_values = cleaned_string.split(',')
parsed_values = [float(v) for v in string_values]

# --- 2. Handle -inf by replacing it with a more moderate number ---
replacement_value = -100.0
handled_values = [v if v != -np.inf else replacement_value for v in parsed_values]

# --- 3. Convert to the final TensorFlow tensor ---
final_tensor = tf.constant(handled_values, dtype=tf.float32)

print(f"Original String: {logit_string}")
print(f"Handled List: {handled_values}")
print(f"Final Usable Tensor: {final_tensor}")

##########


import tensorflow as tf
import tf_keras as keras
from tf_keras import optimizers, losses, metrics
from transformers import AutoTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
from dataclasses import dataclass

# --- 1. Centralized Configuration ---
@dataclass
class Config:
    """Holds all hyperparameters and settings."""
    # Paths and Models
    local_model_path: str = "/base_model"
    # Data and Splitting
    test_split_size: float = 0.2
    validation_split_size: float = 0.25 # 0.25 of (1-test_size) = 20% of total
    random_state: int = 42
    # Training Hyperparameters
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_seq_len: int = 128
    # Distillation Parameters
    alpha: float = 0.3  # Weight for hard loss
    temperature: float = 4.0

# --- 2. Data Preparation and Pipeline ---
def create_tf_dataset(data, tokenizer, config):
    """Creates an efficient, batched, and prefetched tf.data.Dataset."""
    encodings = tokenizer([d['text'] for d in data], truncation=True, padding=True, max_length=config.max_seq_len)
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        {
            'teacher_logits': np.array([d['teacher_logits'] for d in data], dtype=np.float32),
            'hard_labels': np.array([d['hard_label'] for d in data], dtype=np.int32)
        }
    ))
    return dataset.cache().batch(config.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 3. Compiled Training and Evaluation Steps ---
@tf.function
def train_step(x_batch, y_batch, model, optimizer, loss_fns, metrics_map, config):
    """Performs a single, compiled training step."""
    kl_loss_fn, ce_loss_fn = loss_fns
    train_loss_metric = metrics_map['train_loss']

    with tf.GradientTape() as tape:
        student_logits = model(x_batch, training=True).logits
        teacher_logits = y_batch['teacher_logits']
        hard_labels = y_batch['hard_labels']

        soft_teacher_probs = tf.nn.softmax(teacher_logits / config.temperature)
        soft_student_probs = tf.nn.softmax(student_logits / config.temperature)
        distill_loss = kl_loss_fn(soft_teacher_probs, soft_student_probs) * (config.temperature**2)
        hard_loss = ce_loss_fn(hard_labels, student_logits)
        total_loss = config.alpha * hard_loss + (1.0 - config.alpha) * distill_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_metric.update_state(total_loss)

@tf.function
def eval_step(x_batch, y_batch, model, loss_fn, metrics_map):
    """Performs a single, compiled evaluation step."""
    val_loss_metric, val_accuracy_metric = metrics_map['val_loss'], metrics_map['val_accuracy']
    
    student_logits = model(x_batch, training=False).logits
    hard_labels = y_batch['hard_labels']
    
    val_loss = loss_fn(hard_labels, student_logits)
    val_loss_metric.update_state(val_loss)
    val_accuracy_metric.update_state(hard_labels, student_logits)

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Assume 'teacher_data' is your full list of dictionaries
    # For demonstration, we'll create dummy data
    num_samples = 100
    num_classes = 3
    teacher_data = [{
        'text': f'Sample text entry number {i}.',
        'teacher_logits': np.random.randn(num_classes).astype(np.float32),
        'hard_label': np.random.randint(0, num_classes)
    } for i in range(num_samples)]
    
    # Split data
    train_val_data, test_data = train_test_split(teacher_data, test_size=config.test_split_size, random_state=config.random_state)
    train_data, val_data = train_test_split(train_val_data, test_size=config.validation_split_size, random_state=config.random_state)

    # Load model and tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(config.local_model_path)
    student_model = TFBertForSequenceClassification.from_pretrained(config.local_model_path, num_labels=num_classes)

    # Create data pipelines
    train_dataset = create_tf_dataset(train_data, student_tokenizer, config)
    val_dataset = create_tf_dataset(val_data, student_tokenizer, config)
    test_dataset = create_tf_dataset(test_data, student_tokenizer, config)

    # Setup optimizers, losses, and metrics
    optimizer = optimizers.Adam(learning_rate=config.learning_rate)
    loss_functions = [losses.KLDivergence(), losses.SparseCategoricalCrossentropy(from_logits=True)]
    metrics_map = {
        'train_loss': metrics.Mean(name='train_loss'),
        'val_loss': metrics.Mean(name='val_loss'),
        'val_accuracy': metrics.SparseCategoricalAccuracy(name='val_accuracy')
    }
    
    # Training loop
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")
        
        # Reset metrics at the start of each epoch
        for metric in metrics_map.values():
            metric.reset_states()

        # Training
        for x_batch, y_batch in train_dataset:
            train_step(x_batch, y_batch, student_model, optimizer, loss_functions, metrics_map, config)

        # Validation
        for x_batch_val, y_batch_val in val_dataset:
            eval_step(x_batch_val, y_batch_val, student_model, loss_functions[1], metrics_map)

        print(f"Train Loss: {metrics_map['train_loss'].result():.4f} | "
              f"Val Loss: {metrics_map['val_loss'].result():.4f} | "
              f"Val Accuracy: {metrics_map['val_accuracy'].result():.4f}")

    # Final evaluation on the test set
    print("\n--- Final Evaluation on Test Set ---")
    test_accuracy_metric = metrics.SparseCategoricalAccuracy(name='test_accuracy')
    for x_batch_test, y_batch_test in test_dataset:
        student_logits_test = student_model(x_batch_test, training=False).logits
        test_accuracy_metric.update_state(y_batch_test['hard_labels'], student_logits_test)
    
    print(f"Final Test Accuracy: {test_accuracy_metric.result():.4f}")
