"""
Pipeline Configuration
Contains all constants, paths, and thresholds
"""

import os
from pathlib import Path

# ============================================================================
# Label Mappings
# ============================================================================

# Standardized labels across entire pipeline
STANDARD_LABELS = {
    "ASYNC_AWAIT": 0,
    "UNORDERED_COLLECTION": 1,
    "CONCURRENCY": 2,
    "TIME_DEPENDENT": 3,
    "TEST_ORDER_DEPENDENCY": 4,
    "NOT_FLAKY_OR_UNKNOWN": -1
}

# Reverse mapping for ML model output (numeric to standard label)
ML_TO_STANDARD = {
    0: "ASYNC_AWAIT",
    1: "UNORDERED_COLLECTION",
    2: "CONCURRENCY",
    3: "TIME_DEPENDENT",
    4: "TEST_ORDER_DEPENDENCY"
}

# ============================================================================
# Thresholds
# ============================================================================

# Token count threshold for ML vs LLM detection
TOKEN_THRESHOLD = 512

# Confidence mapping thresholds (numeric to text)
CONFIDENCE_HIGH_THRESHOLD = 0.8
CONFIDENCE_MEDIUM_THRESHOLD = 0.5

# ============================================================================
# Model Paths
# ============================================================================

# IDoFT model (binary flaky detection)
IDOFT_MODEL_PATH = os.getenv(
    'IDOFT_MODEL_PATH',
    'models/model_snapshot_IDoFT.pth'
)
IDOFT_EMBEDDINGS_PATH = os.getenv(
    'IDOFT_EMBEDDINGS_PATH',
    'embeddings/train_embeddings_IDoFT.pt'
)

# FlakyCat model (label prediction)
FLAKYCAT_MODEL_PATH = os.getenv(
    'FLAKYCAT_MODEL_PATH',
    'models/model_snapshot_FlakyCat.pth'
)
FLAKYCAT_EMBEDDINGS_PATH = os.getenv(
    'FLAKYCAT_EMBEDDINGS_PATH',
    'embeddings/train_embeddings_FlakyCat.pt'
)

# ============================================================================
# Ollama Configuration
# ============================================================================

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
INLINER_MODEL = os.getenv('INLINER_MODEL', 'qwen3-coder:480b-cloud')
DETECTION_MODEL = os.getenv('DETECTION_MODEL', 'gpt-oss:20b-cloud')
PATCH_MODEL = os.getenv('PATCH_MODEL', 'gpt-oss:20b-cloud')

# ============================================================================
# Timeouts (seconds)
# ============================================================================

INLINER_TIMEOUT = 300  # 5 minutes
LLM_DETECTION_TIMEOUT = 300  # 5 minutes
PATCH_GENERATION_TIMEOUT = 300  # 5 minutes

# ============================================================================
# File Names
# ============================================================================

CALL_GRAPH_FILE = "01_call_graph.json"
INLINER_OUTPUT_FILE = "02_inliner_output.json"
TOKEN_COUNT_FILE = "03_token_count.txt"
BINARY_PREDICTION_FILE = "04a_binary_prediction.json"
LABEL_PREDICTION_FILE = "04a_label_prediction.json"
ZERO_SHOT_FILE = "04b_zero_shot.json"
ONE_SHOT_FILE = "04b_one_shot.json"
RESOLVED_LABEL_FILE = "04b_resolved_label.json"
PATCH_FILE = "05_patch.txt"

RESULTS_FILE = "results.json"
SUMMARY_FILE = "summary.md"

# ============================================================================
# Helper Functions
# ============================================================================

def map_confidence(score: float) -> str:
    """
    Convert numeric confidence score to text label
    
    Args:
        score: Confidence score between 0.0 and 1.0
    
    Returns:
        "HIGH", "MEDIUM", or "LOW"
    """
    if score > CONFIDENCE_HIGH_THRESHOLD:
        return "HIGH"
    elif score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def get_intermediate_dir(project_name: str, test_method: str, output_dir: str) -> Path:
    """
    Get path to intermediate files directory
    
    Args:
        project_name: Name of the project
        test_method: Name of the test method
        output_dir: Base output directory
    
    Returns:
        Path to intermediate directory
    """
    dir_name = f"{project_name}_{test_method}_intermediate"
    return Path(output_dir) / dir_name


def get_results_dir(project_name: str, test_method: str, output_dir: str) -> Path:
    """
    Get path to results directory
    
    Args:
        project_name: Name of the project
        test_method: Name of the test method
        output_dir: Base output directory
    
    Returns:
        Path to results directory
    """
    dir_name = f"{project_name}_{test_method}_results"
    return Path(output_dir) / dir_name
