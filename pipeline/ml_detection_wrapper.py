"""
ML-based Detection Wrapper
Wraps flaky_test_prediction and flaky_label_prediction modules
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any
import logging

from .config import (
    BINARY_PREDICTION_FILE,
    LABEL_PREDICTION_FILE,
    ML_TO_STANDARD,
    map_confidence
)
from .utils import save_json


def detect_with_ml(
    code: str,
    output_dir: str,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    ML-based detection (binary + label)
    
    Args:
        code: Java code to analyze
        output_dir: Directory for output files
        logger: Optional logger
    
    Returns:
        {
            "status": "success" | "error" | "fallback_to_llm",
            "binary_file": "path/to/04a_binary_prediction.json",
            "label_file": "path/to/04a_label_prediction.json" | None,
            "is_flaky": True | False,
            "label": "ASYNC_AWAIT" | None,  # Standardized
            "confidence": "HIGH" | "MEDIUM" | "LOW",
            "raw_confidence": 0.95,
            "error": None | "error message",
            "execution_time_seconds": 2.1
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    binary_file = Path(output_dir) / BINARY_PREDICTION_FILE
    label_file = Path(output_dir) / LABEL_PREDICTION_FILE
    
    logger.info(f"Running ML-based detection...")
    
    try:
        # Add flaky_test_prediction to path
        sys.path.insert(0, "flaky_test_prediction")
        
        # Import prediction function
        from prediction import predict_test
        
        # Step 1: Binary prediction
        logger.info("  Step 1: Binary flaky detection...")
        binary_result = predict_test(code)
        
        is_flaky = binary_result.get("is_flaky", False)
        binary_confidence = binary_result.get("confidence", 0.0)
        
        # Save binary result
        binary_output = {
            "is_flaky": is_flaky,
            "confidence": binary_confidence,
            "confidence_text": map_confidence(binary_confidence),
            "model": "IDoFT"
        }
        save_json(binary_output, str(binary_file))
        
        logger.info(f"    Binary result: {'FLAKY' if is_flaky else 'NOT_FLAKY'} (confidence: {binary_confidence:.3f})")
        
        # If not flaky, stop here
        if not is_flaky:
            execution_time = time.time() - start_time
            logger.info(f"✓ ML detection complete: NOT_FLAKY ({execution_time:.1f}s)")
            
            return {
                "status": "success",
                "binary_file": str(binary_file),
                "label_file": None,
                "is_flaky": False,
                "label": "NOT_FLAKY_OR_UNKNOWN",
                "confidence": map_confidence(binary_confidence),
                "raw_confidence": binary_confidence,
                "error": None,
                "execution_time_seconds": execution_time
            }
        
        # Step 2: Label prediction (only if flaky)
        logger.info("  Step 2: Flakiness label prediction...")
        
        # Add flaky_label_prediction to path
        sys.path.insert(0, "flaky_label_prediction")
        
        # Import necessary modules
        from siamese_network import SiameseNetwork
        from prediction_utils import get_class_rep, calculate_normalized_distance
        import torch
        from transformers import RobertaTokenizer, RobertaModel
        
        # Load model and embeddings
        from config.default_config import CONFIG
        
        model_path = "models/model_snapshot_FlakyCat.pth"
        embeddings_path = "embeddings/train_embeddings_FlakyCat.pt"
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Load training embeddings
        train_data = torch.load(embeddings_path, map_location=device)
        train_embeddings = train_data["embeddings"]
        train_labels = train_data["labels"]
        
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.tolist()
        
        # Get class representatives
        class_reps = get_class_rep(train_embeddings, train_labels)
        
        # Generate embedding for test code
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        codebert = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
        
        tokens = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            codebert_output = codebert(**tokens)
            test_embedding = model(codebert_output.last_hidden_state[:, 0, :])
            test_embedding = test_embedding.cpu().numpy()[0]
        
        # Calculate distances to each class
        distances = [calculate_normalized_distance(test_embedding, rep) for rep in class_reps]
        
        # Predict label (closest class)
        label_id = distances.index(min(distances))
        min_distance = min(distances)
        
        # Convert distance to confidence (inverse relationship)
        label_confidence = max(0.0, 1.0 - min_distance)
        
        # Convert to standard label
        standard_label = ML_TO_STANDARD.get(label_id, "NOT_FLAKY_OR_UNKNOWN")
        
        # Save label result
        label_output = {
            "label_id": label_id,
            "label_name": standard_label,
            "confidence": label_confidence,
            "confidence_text": map_confidence(label_confidence),
            "distances": distances,
            "model": "FlakyCat"
        }
        save_json(label_output, str(label_file))
        
        logger.info(f"    Label: {standard_label} (confidence: {label_confidence:.3f})")
        
        execution_time = time.time() - start_time
        logger.info(f"✓ ML detection complete: {standard_label} ({execution_time:.1f}s)")
        
        return {
            "status": "success",
            "binary_file": str(binary_file),
            "label_file": str(label_file),
            "is_flaky": True,
            "label": standard_label,
            "confidence": map_confidence(label_confidence),
            "raw_confidence": label_confidence,
            "error": None,
            "execution_time_seconds": execution_time
        }
    
    except Exception as e:
        error_msg = f"ML detection failed: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        logger.warning("Will fall back to LLM detection")
        
        return {
            "status": "fallback_to_llm",
            "binary_file": None,
            "label_file": None,
            "is_flaky": None,
            "label": None,
            "confidence": None,
            "raw_confidence": None,
            "error": error_msg,
            "execution_time_seconds": time.time() - start_time
        }
