"""
Flaky Test Detection Model - Inference Module
A modular Python script for predicting whether Java test methods are flaky or not.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, classification_report,
    confusion_matrix
)
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_name': 'microsoft/codebert-base',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_length': 218,
    'embedding_size': 768,  # CodeBERT hidden size
    'checkpoint_path': r'G:\Flaky Tests\SPL-3\models\model_snapshot_IDoFT.pth',  # Update with your .pth file path
}

LABEL_MAPPING = {
    0: 'notFlaky',
    1: 'Flaky'
}

REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


# ============================================================================
# SIAMESE NETWORK DEFINITION
# ============================================================================

class SiameseNetwork(nn.Module):
    """
    Siamese Network for code embedding transformation.
    Maps CodeBERT embeddings to a modified embedding space.
    """
    def __init__(self, embedding_size: int = 768):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, int(embedding_size / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embedding_size / 2), int(embedding_size / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embedding_size / 4), embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        output = self.fc(x)
        return output


# ============================================================================
# EMBEDDING AND PREPROCESSING
# ============================================================================

class CodeEmbedder:
    """Handles code snippet tokenization and embedding generation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the CodeEmbedder with CodeBERT model and tokenizer.
        
        Args:
            config: Configuration dictionary
        """
        self.device = config['device']
        self.max_length = config['max_length']
        self.model_name = config['model_name']
        
        # Load CodeBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.codebert_model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True
        ).to(self.device)
        self.codebert_model.eval()
    
    def embed_code(self, code_snippet: str) -> torch.Tensor:
        """
        Generate embedding for a code snippet using CodeBERT.
        
        Args:
            code_snippet: Java test method as string
            
        Returns:
            CLS token embedding (768-dim vector)
        """
        # Tokenize the input
        inputs = self.tokenizer(
            code_snippet,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embedding without gradient computation
        with torch.no_grad():
            outputs = self.codebert_model(**inputs)
            # Extract CLS token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach()
        
        return cls_embedding
    
    def embed_long_code(self, code_snippet: str) -> List[torch.Tensor]:
        """
        Handle long code snippets by splitting into chunks.
        
        Args:
            code_snippet: Java test method (potentially long)
            
        Returns:
            List of embeddings for each chunk
        """
        inputs = self.tokenizer(code_snippet, return_tensors='pt')
        embeddings = []
        
        i = 0
        while i < len(inputs) - 200:
            chunk = code_snippet[i:i + 250]
            input_chunk = self.tokenizer(
                chunk,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.codebert_model(**input_chunk)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach()
                embeddings.append(cls_embedding)
            
            i += 100
        
        return embeddings


# ============================================================================
# DISTANCE CALCULATIONS
# ============================================================================

class DistanceCalculator:
    """Handles distance calculations between embeddings."""
    
    @staticmethod
    def calculate_normalized_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate normalized (L2) distance between two vectors.
        
        Args:
            vec1: First vector (numpy array or torch tensor)
            vec2: Second vector (numpy array or torch tensor)
            
        Returns:
            Normalized Euclidean distance
        """
        # Convert to numpy if needed
        if not isinstance(vec1, np.ndarray):
            vec1 = vec1.cpu().detach().numpy()
        if not isinstance(vec2, np.ndarray):
            vec2 = vec2.cpu().detach().numpy()
        
        # Normalize vectors
        norm_vec1 = vec1 / np.linalg.norm(vec1)
        norm_vec2 = vec2 / np.linalg.norm(vec2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(norm_vec1 - norm_vec2)
        
        return distance
    
    @staticmethod
    def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0 to 1, where 1 is identical)
        """
        if not isinstance(vec1, np.ndarray):
            vec1 = vec1.cpu().detach().numpy()
        if not isinstance(vec2, np.ndarray):
            vec2 = vec2.cpu().detach().numpy()
        
        # Normalize and compute dot product
        norm_vec1 = vec1 / np.linalg.norm(vec1)
        norm_vec2 = vec2 / np.linalg.norm(vec2)
        
        cosine_sim = np.dot(norm_vec1, norm_vec2)
        return cosine_sim


# ============================================================================
# CLASS REPRESENTATIVE CALCULATION
# ============================================================================

class ClassRepresentativeCalculator:
    """Calculates mean embeddings for each class (for inference with pre-trained representatives)."""
    
    @staticmethod
    def get_class_representatives(
        embeddings: List[torch.Tensor],
        labels: List[int]
    ) -> Dict[int, np.ndarray]:
        """
        Calculate mean embedding for each class.
        
        Args:
            embeddings: List of embedding tensors
            labels: List of corresponding labels
            
        Returns:
            Dictionary mapping label -> mean embedding vector
        """
        representatives = {}
        
        for label in set(labels):
            # Get indices for this label
            indices = [i for i, l in enumerate(labels) if l == label]
            
            # Get embeddings for this label
            class_embeddings = [embeddings[i] for i in indices]
            class_embeddings = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e 
                               for e in class_embeddings]
            
            # Calculate mean
            representatives[label] = np.mean(class_embeddings, axis=0)
        
        return representatives


# ============================================================================
# FLAKY TEST PREDICTOR
# ============================================================================

class FlakyTestPredictor:
    """Main predictor class combining all components for inference."""
    
    def __init__(self, config: Dict, checkpoint_path: str):
        """
        Initialize the predictor with trained models.
        
        Args:
            config: Configuration dictionary
            checkpoint_path: Path to Siamese network checkpoint (.pth file)
        """
        self.config = config
        self.device = config['device']
        
        # Initialize components
        self.embedder = CodeEmbedder(config)
        self.distance_calc = DistanceCalculator()
        
        # Load Siamese network
        self.siamese_network = SiameseNetwork(config['embedding_size']).to(self.device)
        self.siamese_network.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.siamese_network.eval()
        
        # Placeholder for class representatives (to be set via calibration)
        self.class_representatives = None
    
    def set_class_representatives(self, calibration_embeddings: List[torch.Tensor], 
                                   calibration_labels: List[int]):
        """
        Set class representatives from calibration data.
        
        Args:
            calibration_embeddings: List of embeddings from training data
            calibration_labels: List of corresponding labels
        """
        self.class_representatives = ClassRepresentativeCalculator.get_class_representatives(
            calibration_embeddings, calibration_labels
        )
    
    def predict(self, java_test_code: str) -> Dict:
        """
        Predict if a Java test method is flaky or not.
        
        Args:
            java_test_code: Java test method as string
            
        Returns:
            Dictionary with prediction, confidence, and per-label scores
        """
        if self.class_representatives is None:
            raise ValueError("Class representatives not set. Call set_class_representatives() first.")
        
        # Step 1: Get CodeBERT embedding
        codebert_embedding = self.embedder.embed_code(java_test_code)
        
        # Step 2: Transform through Siamese network
        with torch.no_grad():
            transformed_embedding = self.siamese_network(codebert_embedding.unsqueeze(0)).squeeze()
        
        # Step 3: Calculate distances to class representatives
        distances = {}
        for label, rep in self.class_representatives.items():
            distance = self.distance_calc.calculate_normalized_distance(rep, transformed_embedding)
            distances[label] = distance
        
        # Step 4: Convert distances to confidence scores (inverse relationship)
        # Normalize distances to [0, 1] range
        min_dist = min(distances.values())
        max_dist = max(distances.values())
        
        confidence_scores = {}
        for label, dist in distances.items():
            # Invert: smaller distance = higher confidence
            if max_dist == min_dist:
                confidence = 0.5
            else:
                confidence = 1 - ((dist - min_dist) / (max_dist - min_dist))
            confidence_scores[label] = confidence
        
        # Step 5: Determine predicted label
        predicted_label = min(distances, key=distances.get)
        predicted_label_str = LABEL_MAPPING[predicted_label]
        
        return {
            'prediction': predicted_label_str,
            'predicted_label_id': predicted_label,
            'confidence': confidence_scores[predicted_label],
            'all_confidences': {LABEL_MAPPING[k]: v for k, v in confidence_scores.items()},
            'distances': {LABEL_MAPPING[k]: float(v) for k, v in distances.items()},
            'embedding_shape': codebert_embedding.shape,
            'transformed_embedding_shape': transformed_embedding.shape
        }
    
    def predict_batch(self, java_test_codes: List[str]) -> List[Dict]:
        """
        Predict for multiple Java test methods.
        
        Args:
            java_test_codes: List of Java test method strings
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for i, code in enumerate(java_test_codes):
            try:
                prediction = self.predict(code)
                predictions.append(prediction)
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'code_index': i
                })
        
        return predictions


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""
    
    @staticmethod
    def calculate_multiclass_roc_auc(true_labels: List[int], 
                                     pred_labels: List[int], 
                                     average: str = 'weighted') -> float:
        """
        Calculate ROC AUC score for multiclass classification.
        
        Args:
            true_labels: True labels
            pred_labels: Predicted labels
            average: Averaging method
            
        Returns:
            ROC AUC score
        """
        from sklearn.preprocessing import LabelBinarizer
        
        lb = LabelBinarizer()
        lb.fit(true_labels)
        true_bin = lb.transform(true_labels)
        pred_bin = lb.transform(pred_labels)
        
        return roc_auc_score(true_bin, pred_bin, average=average)
    
    @staticmethod
    def evaluate(true_labels: List[int], pred_labels: List[int]) -> Dict:
        """
        Calculate all metrics.
        
        Args:
            true_labels: True labels
            pred_labels: Predicted labels
            
        Returns:
            Dictionary with all metrics
        """
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(true_labels, pred_labels)
        
        try:
            auc = MetricsCalculator.calculate_multiclass_roc_auc(true_labels, pred_labels)
        except:
            auc = None
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'auc': auc,
            'classification_report': classification_report(true_labels, pred_labels, 
                                                          target_names=[LABEL_MAPPING[i] for i in sorted(set(true_labels))],
                                                          zero_division=0),
            'confusion_matrix': confusion_matrix(true_labels, pred_labels)
        }
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """Pretty print metrics."""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1 Score  : {metrics['f1_score']:.4f}")
        print(f"MCC       : {metrics['mcc']:.4f}")
        if metrics['auc'] is not None:
            print(f"AUC       : {metrics['auc']:.4f}")
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(metrics['classification_report'])
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print(metrics['confusion_matrix'])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_java_test() -> str:
    """Create a sample Java test method for demonstration."""
    return """
    @Test
    public void testAsyncWait() throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        Future<Integer> future = executor.submit(() -> {
            Thread.sleep(100);
            return 42;
        });
        
        int result = future.get(5, TimeUnit.SECONDS);
        assertEquals(42, result);
        executor.shutdown();
    }
    """


# ============================================================================
# MAIN INFERENCE EXAMPLE
# ============================================================================

def main():
    """Main function demonstrating inference workflow."""
    
    print("\n" + "="*60)
    print("FLAKY TEST DETECTOR - INFERENCE MODE")
    print("="*60)
    
    # Initialize predictor
    try:
        predictor = FlakyTestPredictor(CONFIG, CONFIG['checkpoint_path'])
        print(f"\n✓ Model loaded from: {CONFIG['checkpoint_path']}")
        print(f"✓ Device: {CONFIG['device']}")
    except FileNotFoundError:
        print(f"\n✗ Error: Checkpoint not found at {CONFIG['checkpoint_path']}")
        print("Please update CONFIG['checkpoint_path'] with your .pth file location")
        return
    
    # IMPORTANT: You need to calibrate with training data first
    print("\n" + "-"*60)
    print("NOTE: To use the predictor, you must first:")
    print("1. Load your training embeddings and labels")
    print("2. Call: predictor.set_class_representatives(embeddings, labels)")
    print("-"*60)
    
    # Example: Single test inference
    print("\n" + "="*60)
    print("EXAMPLE SINGLE TEST INFERENCE")
    print("="*60)
    
    sample_test = create_sample_java_test()
    print(f"\nTest method:\n{sample_test}")
    
    # Uncomment once class representatives are set:
    # result = predictor.predict(sample_test)
    # print("\nPrediction Result:")
    # print(f"  Label: {result['prediction']}")
    # print(f"  Confidence: {result['confidence']:.4f}")
    # print(f"\n  Confidence Scores:")
    # for label, conf in result['all_confidences'].items():
    #     print(f"    - {label}: {conf:.4f}")
    # print(f"\n  Distances:")
    # for label, dist in result['distances'].items():
    #     print(f"    - {label}: {dist:.6f}")


if __name__ == "__main__":
    main()