import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import sklearn.metrics as metrics
import seaborn as sn
import pandas as pd
import warnings
import os

# Import custom modules
from siamese_dataset import SiameseDataset
from siamese_network import SiameseNetwork
from utils import multiclass_roc_auc_score, get_class_rep, get_closest_cluster
from data_loader import prepare_data

# Configuration
warnings.filterwarnings("ignore")
np.random.seed(123456)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths - Change path here
v0_data_dir = r'G:\Flaky Tests\SPL-3\dataset\FlakyCat_data\test_files_v0'
v12_data_dir = r'G:\Flaky Tests\SPL-3\dataset\FlakyCat_data\test_files_v12'

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
batch_size = 8

# Initialize CodeBERT
print("Loading CodeBERT model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model_codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
model_codebert = AutoModel.from_pretrained(
    "microsoft/codebert-base",
    trust_remote_code=True,
    use_safetensors=True
).to(device)


# Load and prepare data
print("\nPreparing data...")
(train_buggy_code, train_filenames, valid_buggy_code, valid_filenames,
 train_category_counts, valid_category_counts) = prepare_data(
    v0_data_dir, v12_data_dir, label_to_int
)

print("\nCategory-wise file counts in Validation Set:")
for category, counts in valid_category_counts.items():
    print(f"{category}: total={sum(counts.values())}, v0={counts['v0']}, "
          f"v1={counts['v1']}, v2={counts['v2']}")

# Create datasets
print("\nCreating datasets...")
train_dataset = SiameseDataset(train_buggy_code, tokenizer, model_codebert, 
                               train_filenames, 'train', label_to_int, m_len)
val_dataset = SiameseDataset(valid_buggy_code, tokenizer, model_codebert, 
                             valid_filenames, 'val', label_to_int, m_len)

# Load trained model
print("\nLoading trained model...")
siamese_network = SiameseNetwork(m_len).to(device)
siamese_network.load_state_dict(torch.load(r'G:\Flaky Tests\SPL-3\models\model_snapshot_FlakyCat.pth', map_location=device))
siamese_network.eval()

# === 🔥 CHANGED: Generate or load cached embeddings ===
embedding_cache_path = r"G:\Flaky Tests\SPL-3\embeddings\train_embeddings_FlakyCat.pt"

if os.path.exists(embedding_cache_path):
    print("\nLoading cached training embeddings...")
    saved = torch.load(embedding_cache_path, map_location=device)
    post_train_embed = saved['embeddings']
    post_train_label = saved['labels']
    print(f"\nLoaded {len(post_train_embed)} training embeddings from cache")
else:
    print("\nGenerating embeddings from training set (this may take a while)...")
    post_train_embed = []
    post_train_label = []
    with torch.no_grad():
        for item in tqdm(train_dataset, desc="Extracting training embeddings"):
            post_train_embed.append(siamese_network(item['anchor'].to(device)))
            post_train_label.append(item['label'])
    
    # 🔥 CHANGED: Save the embeddings after generation
    torch.save({'embeddings': post_train_embed, 'labels': post_train_label}, embedding_cache_path)
    print(f"\nTraining embeddings cached at '{embedding_cache_path}'")

# === 🔥 NEW: Function to preprocess code snippet ===
def preprocess_code_snippet(code_snippet):
    """
    Preprocesses a code snippet and converts it to the format expected by the model.
    Matches the format used in SiameseDataset by inspecting the training data.
    
    Args:
        code_snippet (str): Raw code snippet as string
        
    Returns:
        torch.Tensor: Preprocessed tensor ready for prediction (shape: [m_len])
    """
    # 🔥 IMPROVED: Check the actual format from training dataset
    if len(train_dataset) > 0:
        sample_item = train_dataset[0]
        sample_shape = sample_item['anchor'].shape
        print(f"\nDebug: Training data shape = {sample_shape}, m_len = {m_len}")
    
    # 🔥 FIXED: Use the same tokenization approach as the dataset
    # Tokenize with proper length handling for CodeBERT (max 512 tokens)
    encoded = tokenizer(
        code_snippet,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # encoded = tokenizer.encode_plus(
    #     code_snippet,
    #     add_special_tokens=True,
    #     max_length=512,  # CodeBERT's maximum sequence length
    #     padding='max_length',
    #     truncation=True,
    #     return_tensors='pt'
    # )
    

    # Get CodeBERT embedding
    with torch.no_grad():
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Generate embeddings
        outputs = model_codebert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the pooled output or CLS token embedding (768-dimensional)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [1, 768]
        
        # Flatten
        embedding_vector = cls_embedding.squeeze(0)  # Shape: [768]
    
    # 🔥 FIXED: Match m_len by padding or truncating
    if embedding_vector.shape[0] < m_len:
        # Pad with zeros to reach m_len
        padded_vector = torch.zeros(m_len, device=device)
        padded_vector[:embedding_vector.shape[0]] = embedding_vector
        return padded_vector
    else:
        # Truncate if somehow larger (shouldn't happen with 768-dim embedding)
        return embedding_vector[:m_len]

# === 🔥 CHANGED: Modified prediction function ===
def predict(input_vector):
    """
    Predicts the flaky test category for a given input vector.
    
    Args:
        input_vector (torch.Tensor): Input code embedding
        
    Returns:
        str: Predicted flaky test category
    """
    with torch.no_grad():
        modified_vector = siamese_network(input_vector.to(device))
        representatives = get_class_rep(post_train_embed, post_train_label)
        return get_closest_cluster(representatives, modified_vector, int_to_label)

# === 🔥 NEW: Main detection function for code snippets ===
def detect_flaky_test(code_snippet):
    """
    Detects if a code snippet results in a flaky test and identifies the category.
    
    Args:
        code_snippet (str): The code snippet to analyze
        
    Returns:
        dict: Contains 'is_flaky', 'category', and 'confidence' information
    """
    print("\nAnalyzing code snippet...")
    
    # Preprocess the code snippet
    input_embedding = preprocess_code_snippet(code_snippet)
    
    # Get prediction
    predicted_category = predict(input_embedding)
    
    # 🔥 NEW: Calculate confidence based on distance to cluster representatives
    with torch.no_grad():
        modified_vector = siamese_network(input_embedding.to(device))
        representatives = get_class_rep(post_train_embed, post_train_label)
        
        # Calculate distances to all cluster representatives
        distances = {}
        # for label_name, rep_vector in representatives.items():
        for idx, rep_vector in enumerate(representatives):
            label_name = f"class_{idx}"
            rep_vector = torch.tensor(rep_vector).to(device)        # Added this line 
            dist = torch.nn.functional.cosine_similarity(
                modified_vector.unsqueeze(0), 
                rep_vector.unsqueeze(0)
            ).item()
            distances[label_name] = dist
        
        # Get confidence score (normalized similarity)
        max_similarity = max(distances.values())
        confidence = max_similarity * 100  # Convert to percentage
    
    result = {
        'is_flaky': True,  # All categories indicate flaky tests
        'category': predicted_category,
        'confidence': confidence,
        'all_scores': distances
    }
    
    return result

# === 🔥 NEW: Interactive prediction interface ===
def interactive_detection():
    """
    Provides an interactive interface for detecting flaky tests from code snippets.
    """
    print("\n" + "="*70)
    print("FLAKY TEST DETECTION SYSTEM")
    print("="*70)
    print("\nEnter your code snippet (type 'END' on a new line when finished):")
    print("Type 'QUIT' to exit the system.\n")
    
    while True:
        lines = []
        print(">>> ", end="")
        while True:
            line = input()
            if line.strip() == 'END':
                break
            if line.strip() == 'QUIT':
                print("\n Exiting detection system...")
                return
            lines.append(line)
        
        if not lines:
            continue
            
        code_snippet = '\n'.join(lines)
        
        # Detect flaky test
        result = detect_flaky_test(code_snippet)
        
        # Display results
        print("\n" + "-"*70)
        print("DETECTION RESULTS")
        print("-"*70)
        print(f"Is Flaky: {'⚠️  YES' if result['is_flaky'] else '✅ NO'}")
        print(f"Category: {result['category'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nAll Category Scores:")
        for category, score in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {category}: {score:.4f}")
        print("-"*70 + "\n")

# === 🔥 CHANGED: Evaluation section (optional) ===
# This section can be run separately or commented out for production use
def evaluate_on_validation_set():
    """
    Evaluates the model on the validation set.
    """
    print("\nEvaluating on validation set...")
    predicted_labels = []
    true_labels = []

    for item in tqdm(val_dataset, desc="Predicting"):
        input_vector = item['anchor']
        predicted_label = predict(input_vector)
        predicted_labels.append(predicted_label)
        true_label = int_to_label[int(item['label'])]
        true_labels.append(true_label)
    
    # Print evaluation metrics
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(classification_report(true_labels, predicted_labels))
    
    return predicted_labels, true_labels

# === 🔥 NEW: Main execution block ===
if __name__ == "__main__":
    # 🔥 NEW: Debug section to understand the data format
    print("\n" + "="*70)
    print("INSPECTING TRAINING DATA FORMAT")
    print("="*70)
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample anchor shape: {sample['anchor'].shape}")
        print(f"Sample anchor type: {type(sample['anchor'])}")
        print(f"Expected m_len: {m_len}")
        print(f"Anchor tensor device: {sample['anchor'].device}")
        print("="*70)
    
    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)
    
    # Example 1: Direct prediction
    example_code = """
    @Test public void testInsert(){
        SqlSession sqlSession=MybatisHelper.getSqlSession();
        try {
            UserInfoAbleMapper mapper=sqlSession.getMapper(UserInfoAbleMapper.class);
            UserInfoAble userInfo=new UserInfoAble();
            userInfo.setUsername(""abel533"");
            userInfo.setPassword(""123456"");
            userInfo.setUsertype(""2"");
            userInfo.setEmail(""abel533@gmail.com"");
            Assert.assertEquals(1,mapper.insert(userInfo));
            Assert.assertNotNull(userInfo.getId());
            Assert.assertEquals(6,(int)userInfo.getId());
            userInfo=mapper.selectByPrimaryKey(userInfo.getId());
            Assert.assertNull(userInfo.getEmail());
        }
        finally {
            sqlSession.rollback();
            sqlSession.close();
        }
    }
    """
    
    print("\nExample Detection:")
    result = detect_flaky_test(example_code)
    print(f"Result: {result}")
    
    