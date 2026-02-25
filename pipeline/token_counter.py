"""
Token Counter using CodeBERT Tokenizer
"""

import time
from pathlib import Path
from typing import Dict, Any
import logging

from .config import TOKEN_COUNT_FILE, TOKEN_THRESHOLD


def count_tokens(
    code: str,
    output_dir: str,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Count tokens in code using CodeBERT tokenizer
    
    Args:
        code: Java code to count tokens
        output_dir: Directory for output files
        logger: Optional logger
    
    Returns:
        {
            "status": "success" | "error",
            "output_file": "path/to/03_token_count.txt",
            "token_count": 450,
            "threshold": 512,
            "use_ml_path": True,  # < 512
            "use_llm_path": False,  # >= 512
            "error": None | "error message"
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    output_file = Path(output_dir) / TOKEN_COUNT_FILE
    
    logger.info(f"Counting tokens...")
    
    try:
        # Import CodeBERT tokenizer
        from transformers import RobertaTokenizer
        
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Tokenize code
        tokens = tokenizer.encode(code, add_special_tokens=True)
        token_count = len(tokens)
        
        # Determine path
        use_ml_path = token_count < TOKEN_THRESHOLD
        use_llm_path = token_count >= TOKEN_THRESHOLD
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Token Count: {token_count}\n")
            f.write(f"Threshold: {TOKEN_THRESHOLD}\n")
            f.write(f"Path: {'ML' if use_ml_path else 'LLM'}\n")
        
        execution_time = time.time() - start_time
        
        logger.info(f"✓ Token count: {token_count} ({execution_time:.1f}s)")
        logger.info(f"  Detection path: {'ML (< 512 tokens)' if use_ml_path else 'LLM (≥ 512 tokens)'}")
        
        return {
            "status": "success",
            "output_file": str(output_file),
            "token_count": token_count,
            "threshold": TOKEN_THRESHOLD,
            "use_ml_path": use_ml_path,
            "use_llm_path": use_llm_path,
            "error": None
        }
    
    except ImportError as e:
        error_msg = f"CodeBERT tokenizer not available: {str(e)}. Install with: pip install transformers"
        logger.error(error_msg)
        
        # Fallback to simple whitespace tokenization
        logger.warning("Falling back to whitespace tokenization")
        token_count = len(code.split())
        use_ml_path = token_count < TOKEN_THRESHOLD
        use_llm_path = token_count >= TOKEN_THRESHOLD
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Token Count (whitespace): {token_count}\n")
            f.write(f"Threshold: {TOKEN_THRESHOLD}\n")
            f.write(f"Path: {'ML' if use_ml_path else 'LLM'}\n")
            f.write(f"Note: Using fallback whitespace tokenization\n")
        
        logger.warning(f"⚠ Token count (fallback): {token_count}")
        logger.warning(f"  Detection path: {'ML' if use_ml_path else 'LLM'}")
        
        return {
            "status": "success",
            "output_file": str(output_file),
            "token_count": token_count,
            "threshold": TOKEN_THRESHOLD,
            "use_ml_path": use_ml_path,
            "use_llm_path": use_llm_path,
            "error": f"Using fallback tokenization: {error_msg}"
        }
    
    except Exception as e:
        error_msg = f"Token counting failed: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        return {
            "status": "error",
            "output_file": None,
            "token_count": None,
            "threshold": TOKEN_THRESHOLD,
            "use_ml_path": False,
            "use_llm_path": True,  # Default to LLM on error
            "error": error_msg
        }
