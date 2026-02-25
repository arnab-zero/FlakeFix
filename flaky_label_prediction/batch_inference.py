"""
Batch Inference Script for Flaky Label Prediction
Runs inference on all test files in dataset/test_files_v12 and saves results to CSV.
"""

import os
import re
import csv
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import warnings

# Import from existing modules
from siamese_dataset import SiameseDataset
from siamese_network import SiameseNetwork
from utils import get_class_rep, get_closest_cluster

warnings.filterwarnings("ignore")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths - Change path here
TEST_FILES_DIR = r'G:\Flaky Tests\SPL-3\dataset\FlakyCat_data\test_files_v12'
MODEL_PATH = r'G:\Flaky Tests\SPL-3\models\model_snapshot_FlakyCat.pth'
EMBEDDING_CACHE_PATH = r'G:\Flaky Tests\SPL-3\embeddings\train_embeddings_FlakyCat.pt'
OUTPUT_CSV = r'G:\Flaky Tests\SPL-3\flaky-label-prediction\inference_results.csv'

# Label mappings
label_to_int = {
    'async wait': 0,
    'unordered collections': 1,
    'concurrency': 2,
    'time': 3,
    'test order dependency': 4
}

int_to_label = {v: k for k, v in label_to_int.items()}

# Hyperparameters
m_len = 3402

# Initialize CodeBERT
print("Loading CodeBERT model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model_codebert = AutoModel.from_pretrained(
    "microsoft/codebert-base",
    trust_remote_code=True,
    use_safetensors=True
).to(device)
model_codebert.eval()

# Load Siamese network
print("Loading Siamese network...")
siamese_network = SiameseNetwork(m_len).to(device)
siamese_network.load_state_dict(torch.load(MODEL_PATH, map_location=device))
siamese_network.eval()

# Load training embeddings for class representatives
print("Loading training embeddings...")
if os.path.exists(EMBEDDING_CACHE_PATH):
    saved = torch.load(EMBEDDING_CACHE_PATH, map_location=device)
    post_train_embed = saved['embeddings']
    post_train_label = saved['labels']
    print(f"Loaded {len(post_train_embed)} training embeddings")
else:
    print(f"ERROR: Training embeddings not found at {EMBEDDING_CACHE_PATH}")
    exit(1)

# Get class representatives
print("Computing class representatives...")
representatives = get_class_rep(post_train_embed, post_train_label)


def extract_label_from_filename(filename):
    """
    Extract actual label from filename.
    Label is between '@' and '.txt'
    
    Example: 'v1_activemq.BrokerTest.testConsumerClose@Async wait.txt'
    Returns: 'async wait'
    """
    match = re.search(r'@(.+?)\.txt$', filename)
    if match:
        label = match.group(1).lower()
        # Normalize label
        if 'async' in label and 'wait' in label:
            return 'async wait'
        elif 'unordered' in label or 'collection' in label:
            return 'unordered collections'
        elif 'concurrency' in label:
            return 'concurrency'
        elif 'time' in label:
            return 'time'
        elif 'order' in label and 'dependency' in label:
            return 'test order dependency'
        else:
            return label
    return 'unknown'


def preprocess_code(code_text):
    """
    Preprocess code and generate embedding.
    
    Args:
        code_text: Java code as string
        
    Returns:
        Embedding tensor
    """
    # Tokenize
    encoded = tokenizer(
        code_text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    # Get CodeBERT embedding
    with torch.no_grad():
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        outputs = model_codebert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embedding_vector = cls_embedding.squeeze(0)
    
    # Pad to m_len
    if embedding_vector.shape[0] < m_len:
        padded_vector = torch.zeros(m_len, device=device)
        padded_vector[:embedding_vector.shape[0]] = embedding_vector
        return padded_vector
    else:
        return embedding_vector[:m_len]


def predict(input_vector):
    """
    Predict flaky test category.
    
    Args:
        input_vector: Input embedding
        
    Returns:
        Predicted category string
    """
    with torch.no_grad():
        modified_vector = siamese_network(input_vector.to(device))
        return get_closest_cluster(representatives, modified_vector, int_to_label)


def main():
    """Main batch inference function."""
    print("\n" + "="*70)
    print("BATCH INFERENCE - FLAKY LABEL PREDICTION")
    print("="*70)
    
    # Get all test files
    print(f"\nScanning directory: {TEST_FILES_DIR}")
    test_files = [f for f in os.listdir(TEST_FILES_DIR) if f.endswith('.txt')]
    print(f"Found {len(test_files)} test files")
    
    # Prepare CSV output
    results = []
    
    # Process each file
    print("\nRunning inference...")
    for filename in tqdm(test_files, desc="Processing files"):
        try:
            # Read file
            filepath = os.path.join(TEST_FILES_DIR, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                code_text = f.read()
            
            # Extract actual label
            actual_label = extract_label_from_filename(filename)
            
            # Preprocess and predict
            embedding = preprocess_code(code_text)
            predicted_label = predict(embedding)
            
            # Store result
            results.append({
                'test_file_name': filename,
                'actual_label': actual_label,
                'predicted_label': predicted_label
            })
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            results.append({
                'test_file_name': filename,
                'actual_label': 'error',
                'predicted_label': 'error'
            })
    
    # Write to CSV
    print(f"\nWriting results to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['test_file_name', 'actual_label', 'predicted_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"✓ Saved {len(results)} results to {OUTPUT_CSV}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    correct = sum(1 for r in results if r['actual_label'] == r['predicted_label'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    print(f"Total files processed: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Per-category accuracy
    print("\nPer-category breakdown:")
    for label in label_to_int.keys():
        label_results = [r for r in results if r['actual_label'] == label]
        if label_results:
            label_correct = sum(1 for r in label_results if r['predicted_label'] == label)
            label_total = len(label_results)
            label_acc = label_correct / label_total if label_total > 0 else 0
            print(f"  {label:25s}: {label_correct:3d}/{label_total:3d} ({label_acc:.2%})")
    
    print("="*70)


if __name__ == "__main__":
    main()
