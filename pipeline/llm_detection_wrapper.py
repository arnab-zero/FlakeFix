"""
LLM-based Detection Wrapper
Wraps zero_shot_prompt and one_shot_prompt modules
"""

import sys
import time
import requests
from pathlib import Path
from typing import Dict, Any
import logging

from .config import (
    ZERO_SHOT_FILE,
    ONE_SHOT_FILE,
    RESOLVED_LABEL_FILE,
    OLLAMA_BASE_URL,
    DETECTION_MODEL,
    LLM_DETECTION_TIMEOUT
)
from .utils import save_json, parse_llm_json_response


def call_ollama_api(prompt: str, model: str, base_url: str, timeout: int) -> str:
    """
    Call Ollama API with the given prompt
    
    Args:
        prompt: The prompt to send
        model: Model name
        base_url: Ollama base URL
        timeout: Request timeout in seconds
    
    Returns:
        Response text from the model
    
    Raises:
        Exception: If API call fails
    """
    # Try chat endpoint first, then generate endpoint
    endpoints = [
        (f"{base_url}/api/chat", "chat"),
        (f"{base_url}/api/generate", "generate")
    ]
    
    for endpoint_url, endpoint_type in endpoints:
        try:
            if endpoint_type == "chat":
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a software testing analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 2048
                    }
                }
            else:
                full_prompt = "You are a software testing analyst.\n\n" + prompt
                payload = {
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 2048
                    }
                }
            
            response = requests.post(endpoint_url, json=payload, timeout=timeout)
            
            if response.status_code == 404:
                continue
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content
            if endpoint_type == "chat" and 'message' in result and 'content' in result['message']:
                return result['message']['content']
            elif endpoint_type == "generate" and 'response' in result:
                return result['response']
            else:
                raise ValueError(f"Unexpected response format: {result}")
        
        except requests.exceptions.ConnectionError:
            if endpoint_type == endpoints[-1][1]:
                raise ConnectionError(f"Cannot connect to Ollama at {base_url}")
            continue
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {endpoint_url} timed out")
        except Exception as e:
            if response.status_code != 404:
                raise Exception(f"Error calling Ollama API: {e}")
            continue
    
    raise ValueError(f"Could not find working Ollama endpoint at {base_url}")


def resolve_label(
    zero_shot_result: Dict[str, Any],
    one_shot_result: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Resolve final label from zero-shot and one-shot results
    
    Resolution logic:
    1. If 1-shot = "NOT_FLAKY_OR_UNKNOWN" + "LOW" → use 0-shot
    2. If 0-shot == 1-shot → use that label
    3. Else → use 1-shot
    
    Args:
        zero_shot_result: Zero-shot detection result
        one_shot_result: One-shot detection result
        logger: Logger instance
    
    Returns:
        {
            "final_label": "ASYNC_AWAIT",
            "final_confidence": "HIGH",
            "resolution_reason": "explanation",
            "zero_shot_label": "...",
            "one_shot_label": "..."
        }
    """
    zero_label = zero_shot_result.get("label", "NOT_FLAKY_OR_UNKNOWN")
    zero_conf = zero_shot_result.get("confidence", "LOW")
    
    one_label = one_shot_result.get("label", "NOT_FLAKY_OR_UNKNOWN")
    one_conf = one_shot_result.get("confidence", "LOW")
    
    # Rule 1: If 1-shot = "NOT_FLAKY_OR_UNKNOWN" + "LOW" → use 0-shot
    if one_label == "NOT_FLAKY_OR_UNKNOWN" and one_conf == "LOW":
        logger.info(f"  Resolution: Using 0-shot (1-shot returned NOT_FLAKY_OR_UNKNOWN with LOW confidence)")
        return {
            "final_label": zero_label,
            "final_confidence": zero_conf,
            "resolution_reason": "1-shot returned NOT_FLAKY_OR_UNKNOWN with LOW confidence, using 0-shot result",
            "zero_shot_label": zero_label,
            "one_shot_label": one_label
        }
    
    # Rule 2: If 0-shot == 1-shot → use that label
    if zero_label == one_label:
        logger.info(f"  Resolution: Both agree on {zero_label}")
        # Use higher confidence
        final_conf = zero_conf if zero_conf == "HIGH" or one_conf == "LOW" else one_conf
        return {
            "final_label": zero_label,
            "final_confidence": final_conf,
            "resolution_reason": "Both 0-shot and 1-shot agree on the same label",
            "zero_shot_label": zero_label,
            "one_shot_label": one_label
        }
    
    # Rule 3: Else → use 1-shot
    logger.info(f"  Resolution: Using 1-shot (0-shot={zero_label}, 1-shot={one_label})")
    return {
        "final_label": one_label,
        "final_confidence": one_conf,
        "resolution_reason": "0-shot and 1-shot disagree, using 1-shot result",
        "zero_shot_label": zero_label,
        "one_shot_label": one_label
    }


def detect_with_llm(
    code: str,
    output_dir: str,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    LLM-based detection (zero-shot + one-shot + resolution)
    
    Args:
        code: Java code to analyze
        output_dir: Directory for output files
        logger: Optional logger
    
    Returns:
        {
            "status": "success" | "error",
            "zero_shot_file": "path/to/04b_zero_shot.json",
            "one_shot_file": "path/to/04b_one_shot.json",
            "resolved_file": "path/to/04b_resolved_label.json",
            "final_label": "ASYNC_AWAIT",
            "final_confidence": "HIGH",
            "resolution_reason": "explanation",
            "zero_shot_result": {...},
            "one_shot_result": {...},
            "error": None | "error message",
            "execution_time_seconds": 5.2
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    zero_shot_file = Path(output_dir) / ZERO_SHOT_FILE
    one_shot_file = Path(output_dir) / ONE_SHOT_FILE
    resolved_file = Path(output_dir) / RESOLVED_LABEL_FILE
    
    logger.info(f"Running LLM-based detection...")
    
    try:
        # Add llm_based_detection to path
        sys.path.insert(0, "llm_based_detection")
        
        from zero_shot_prompt import build_flaky_prompt as build_zero_shot
        from one_shot_prompt import build_flaky_prompt as build_one_shot
        
        # Step 1: Zero-shot detection
        logger.info("  Step 1: Zero-shot detection...")
        zero_shot_prompt = build_zero_shot(code)
        zero_shot_response = call_ollama_api(
            zero_shot_prompt,
            DETECTION_MODEL,
            OLLAMA_BASE_URL,
            LLM_DETECTION_TIMEOUT
        )
        
        # Parse JSON response
        zero_shot_result = parse_llm_json_response(zero_shot_response)
        
        # Save zero-shot result
        save_json(zero_shot_result, str(zero_shot_file))
        logger.info(f"    0-shot: {zero_shot_result.get('label')} ({zero_shot_result.get('confidence')})")
        
        # Step 2: One-shot detection
        logger.info("  Step 2: One-shot detection...")
        one_shot_prompt = build_one_shot(code)
        one_shot_response = call_ollama_api(
            one_shot_prompt,
            DETECTION_MODEL,
            OLLAMA_BASE_URL,
            LLM_DETECTION_TIMEOUT
        )
        
        # Parse JSON response
        one_shot_result = parse_llm_json_response(one_shot_response)
        
        # Save one-shot result
        save_json(one_shot_result, str(one_shot_file))
        logger.info(f"    1-shot: {one_shot_result.get('label')} ({one_shot_result.get('confidence')})")
        
        # Step 3: Resolve label
        logger.info("  Step 3: Resolving final label...")
        resolved = resolve_label(zero_shot_result, one_shot_result, logger)
        
        # Save resolved result
        resolved_output = {
            **resolved,
            "zero_shot_full": zero_shot_result,
            "one_shot_full": one_shot_result
        }
        save_json(resolved_output, str(resolved_file))
        
        execution_time = time.time() - start_time
        logger.info(f"✓ LLM detection complete: {resolved['final_label']} ({execution_time:.1f}s)")
        
        return {
            "status": "success",
            "zero_shot_file": str(zero_shot_file),
            "one_shot_file": str(one_shot_file),
            "resolved_file": str(resolved_file),
            "final_label": resolved["final_label"],
            "final_confidence": resolved["final_confidence"],
            "resolution_reason": resolved["resolution_reason"],
            "zero_shot_result": zero_shot_result,
            "one_shot_result": one_shot_result,
            "error": None,
            "execution_time_seconds": execution_time
        }
    
    except Exception as e:
        error_msg = f"LLM detection failed: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        
        return {
            "status": "error",
            "zero_shot_file": None,
            "one_shot_file": None,
            "resolved_file": None,
            "final_label": None,
            "final_confidence": None,
            "resolution_reason": None,
            "zero_shot_result": None,
            "one_shot_result": None,
            "error": error_msg,
            "execution_time_seconds": time.time() - start_time
        }
