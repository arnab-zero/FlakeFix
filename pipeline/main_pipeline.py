"""
Main Pipeline Orchestrator
Coordinates all pipeline steps
"""

import time
from pathlib import Path
from typing import Dict, Any
import logging

from .config import get_intermediate_dir, get_results_dir, TOKEN_THRESHOLD
from .utils import validate_file_exists, validate_directory_exists, get_timestamp
from .call_graph_wrapper import generate_call_graph
from .inliner_wrapper import inline_test_method
from .token_counter import count_tokens
from .ml_detection_wrapper import detect_with_ml
from .llm_detection_wrapper import detect_with_llm
from .patch_wrapper import generate_patch


def run_pipeline(
    project_name: str,
    project_root: str,
    test_file: str,
    test_method: str,
    output_dir: str = "output",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run complete flaky test detection and patch generation pipeline
    
    Args:
        project_name: Name of the project
        project_root: Root directory of the project
        test_file: Path to test file (relative to project_root)
        test_method: Name of the test method
        output_dir: Base output directory
        verbose: Enable verbose logging
    
    Returns:
        Complete results dictionary with all step outputs
    """
    # Setup logging (simple console logger first)
    import logging
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Remove existing handlers
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    start_time = time.time()
    logger.info("="*70)
    logger.info(f"FLAKY TEST DETECTION AND PATCH GENERATION PIPELINE")
    logger.info("="*70)
    logger.info(f"Project: {project_name}")
    logger.info(f"Test Method: {test_method}")
    logger.info(f"Test File: {test_file}")
    logger.info("")
    
    # Validate inputs
    try:
        validate_directory_exists(project_root, "Project root")
        
        # Handle both absolute and relative paths for test_file
        test_file_path = Path(test_file)
        if test_file_path.is_absolute():
            # Use absolute path directly
            test_file_full = test_file_path
            # Calculate relative path for later use
            try:
                test_file_relative = test_file_path.relative_to(Path(project_root))
                test_file = str(test_file_relative)
            except ValueError:
                # If test_file is not under project_root, just use the filename
                test_file = test_file_path.name
        else:
            # Use relative path
            test_file_full = Path(project_root) / test_file
        
        validate_file_exists(str(test_file_full), "Test file")
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error(f"✗ Validation failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": get_timestamp()
        }
    
    # Create output directories
    intermediate_dir = get_intermediate_dir(project_name, test_method, output_dir)
    results_dir = get_results_dir(project_name, test_method, output_dir)
    
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Intermediate files: {intermediate_dir}")
    logger.info(f"Results: {results_dir}")
    logger.info("")
    
    # Initialize results
    results = {
        "project_name": project_name,
        "test_method": test_method,
        "test_file": test_file,
        "timestamp": get_timestamp(),
        "steps": {}
    }
    
    # ========================================================================
    # STEP 1: Generate Call Graph
    # ========================================================================
    logger.info("STEP 1: Generating call graph...")
    logger.info("-" * 70)
    
    call_graph_result = generate_call_graph(
        project_root=project_root,
        test_file=str(test_file_full),  # Use full path
        method_name=test_method,  # Fixed: parameter name is method_name, not test_method
        output_dir=str(intermediate_dir),
        logger=logger
    )
    results["steps"]["call_graph"] = call_graph_result
    
    if call_graph_result["status"] != "success":
        logger.error(f"✗ Pipeline failed at Step 1: {call_graph_result['error']}")
        results["status"] = "error"
        results["failed_at_step"] = "call_graph"
        return results
    
    call_graph_path = call_graph_result["output_file"]
    logger.info("")
    
    # ========================================================================
    # STEP 2: Inline Test Method
    # ========================================================================
    logger.info("STEP 2: Inlining test method...")
    logger.info("-" * 70)
    
    inliner_result = inline_test_method(
        call_graph_path=call_graph_path,
        output_dir=str(intermediate_dir),
        logger=logger
    )
    results["steps"]["inliner"] = inliner_result
    
    # Handle inliner failure: use original body and continue
    if inliner_result["status"] == "error":
        logger.warning(f"⚠ Inliner failed: {inliner_result['error']}")
        logger.warning("  Using original test method body and continuing...")
        test_code = inliner_result.get("original_body", "")
        if not test_code:
            logger.error("✗ Cannot continue: No original test method body available")
            results["status"] = "error"
            results["failed_at_step"] = "inliner"
            return results
    else:
        test_code = inliner_result["updated_body"]
    
    logger.info("")
    
    # ========================================================================
    # STEP 3: Count Tokens
    # ========================================================================
    logger.info("STEP 3: Counting tokens...")
    logger.info("-" * 70)
    
    token_result = count_tokens(
        code=test_code,
        output_dir=str(intermediate_dir),
        logger=logger
    )
    results["steps"]["token_count"] = token_result
    
    if token_result["status"] != "success":
        logger.error(f"✗ Pipeline failed at Step 3: {token_result['error']}")
        results["status"] = "error"
        results["failed_at_step"] = "token_count"
        return results
    
    token_count = token_result["token_count"]
    detection_method = "ML" if token_count < TOKEN_THRESHOLD else "LLM"
    logger.info(f"  Detection method: {detection_method} (tokens: {token_count}, threshold: {TOKEN_THRESHOLD})")
    logger.info("")
    
    # ========================================================================
    # STEP 4: Flaky Detection (ML or LLM)
    # ========================================================================
    logger.info(f"STEP 4: Flaky detection ({detection_method})...")
    logger.info("-" * 70)
    
    if detection_method == "ML":
        # Try ML detection first
        detection_result = detect_with_ml(
            code=test_code,
            output_dir=str(intermediate_dir),
            logger=logger
        )
        results["steps"]["detection"] = detection_result
        
        # If ML fails, fall back to LLM
        if detection_result["status"] == "fallback_to_llm":
            logger.warning("⚠ ML detection failed, falling back to LLM detection...")
            logger.info("")
            
            llm_result = detect_with_llm(
                code=test_code,
                output_dir=str(intermediate_dir),
                logger=logger
            )
            results["steps"]["detection_fallback"] = llm_result
            
            if llm_result["status"] != "success":
                logger.error(f"✗ Pipeline failed at Step 4: {llm_result['error']}")
                results["status"] = "error"
                results["failed_at_step"] = "detection"
                return results
            
            final_label = llm_result["final_label"]
            final_confidence = llm_result["final_confidence"]
            is_flaky = final_label != "NOT_FLAKY_OR_UNKNOWN"
            detection_method_used = "LLM (fallback)"
        else:
            if detection_result["status"] != "success":
                logger.error(f"✗ Pipeline failed at Step 4: {detection_result['error']}")
                results["status"] = "error"
                results["failed_at_step"] = "detection"
                return results
            
            final_label = detection_result["label"]
            final_confidence = detection_result["confidence"]
            is_flaky = detection_result["is_flaky"]
            detection_method_used = "ML"
    else:
        # Use LLM detection
        detection_result = detect_with_llm(
            code=test_code,
            output_dir=str(intermediate_dir),
            logger=logger
        )
        results["steps"]["detection"] = detection_result
        
        if detection_result["status"] != "success":
            logger.error(f"✗ Pipeline failed at Step 4: {detection_result['error']}")
            results["status"] = "error"
            results["failed_at_step"] = "detection"
            return results
        
        final_label = detection_result["final_label"]
        final_confidence = detection_result["final_confidence"]
        is_flaky = final_label != "NOT_FLAKY_OR_UNKNOWN"
        detection_method_used = "LLM"
    
    logger.info("")
    
    # ========================================================================
    # STEP 5: Patch Generation (if ASYNC_AWAIT)
    # ========================================================================
    logger.info("STEP 5: Patch generation...")
    logger.info("-" * 70)
    
    patch_result = generate_patch(
        final_label=final_label,
        call_graph_path=call_graph_path,
        output_dir=str(intermediate_dir),
        logger=logger
    )
    results["steps"]["patch"] = patch_result
    
    logger.info("")
    
    # ========================================================================
    # Pipeline Complete
    # ========================================================================
    execution_time = time.time() - start_time
    
    logger.info("="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Status: SUCCESS")
    logger.info(f"Is Flaky: {is_flaky}")
    logger.info(f"Label: {final_label}")
    logger.info(f"Confidence: {final_confidence}")
    logger.info(f"Detection Method: {detection_method_used}")
    logger.info(f"Patch Generated: {patch_result['generated']}")
    logger.info(f"Total Time: {execution_time:.1f}s")
    logger.info("="*70)
    
    # Add summary to results
    results["status"] = "success"
    results["summary"] = {
        "is_flaky": is_flaky,
        "label": final_label,
        "confidence": final_confidence,
        "detection_method": detection_method_used,
        "patch_generated": patch_result["generated"],
        "execution_time_seconds": execution_time
    }
    
    return results
