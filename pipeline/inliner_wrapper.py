"""
Method Inliner Wrapper
Wraps llm_based_inliner module
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any
import logging

from .config import INLINER_OUTPUT_FILE
from .utils import save_json, load_json, get_timestamp


def inline_test_method(
    call_graph_path: str,
    output_dir: str,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Inline test method using llm_based_inliner
    
    Args:
        call_graph_path: Path to call graph JSON
        output_dir: Directory for output files
        logger: Optional logger
    
    Returns:
        {
            "status": "success" | "error",
            "output_file": "path/to/02_inliner_output.json",
            "inlined_body": "...",  # updated_test_method_body
            "original_body": "...",  # original_test_method_body
            "used_original": False,  # True if inlining failed
            "error": None | "error message",
            "execution_time_seconds": 45.2
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    output_file = Path(output_dir) / INLINER_OUTPUT_FILE
    
    logger.info(f"Inlining test method...")
    logger.debug(f"Call graph: {call_graph_path}")
    
    try:
        # Call run_inliner.py as subprocess
        cmd = [
            sys.executable,
            "llm_based_inliner/run_inliner.py",
            call_graph_path
        ]
        
        logger.debug(f"Running command: {' '.join(cmd)}")
        logger.info("This may take 30-120 seconds...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.warning(f"Inliner returned non-zero exit code: {result.returncode}")
            logger.debug(f"Stderr: {result.stderr}")
        
        # The inliner should output JSON to stdout or create a file
        # Try to parse from stdout first
        inliner_result = None
        
        if result.stdout:
            try:
                # Try to find JSON in stdout
                import json
                import re
                
                # Look for JSON object in output
                json_match = re.search(r'\{.*"status".*\}', result.stdout, re.DOTALL)
                if json_match:
                    inliner_result = json.loads(json_match.group(0))
            except Exception as e:
                logger.debug(f"Could not parse JSON from stdout: {e}")
        
        # If not in stdout, look for output file
        if inliner_result is None:
            # Check for inliner output files
            possible_files = [
                "inliner_output.json",
                "llm_based_inliner/inliner_output.json",
                "output/inliner_output.json"
            ]
            
            for pf in possible_files:
                if Path(pf).exists():
                    inliner_result = load_json(pf)
                    Path(pf).unlink()  # Clean up
                    break
        
        if inliner_result is None:
            error_msg = "Could not find inliner output"
            logger.error(error_msg)
            logger.debug(f"Stdout: {result.stdout[:500]}")
            
            # Try to extract original body from call graph JSON as fallback
            try:
                call_graph_data = load_json(call_graph_path)
                body_lines = call_graph_data.get("test_method", {}).get("body", {}).get("lines", [])
                if body_lines:
                    original_body = "\n".join(body_lines)
                    logger.info("Extracted original test method body from call graph")
                    
                    # Save a minimal result with original body
                    fallback_result = {
                        "status": "error",
                        "original_test_method_body": original_body,
                        "updated_test_method_body": "",
                        "error": error_msg
                    }
                    save_json(fallback_result, str(output_file))
                    
                    return {
                        "status": "error",
                        "output_file": str(output_file),
                        "inlined_body": None,
                        "original_body": original_body,
                        "used_original": False,
                        "error": error_msg,
                        "execution_time_seconds": time.time() - start_time
                    }
            except Exception as e:
                logger.debug(f"Could not extract original body from call graph: {e}")
            
            return {
                "status": "error",
                "output_file": None,
                "inlined_body": None,
                "original_body": None,
                "used_original": False,
                "error": error_msg,
                "execution_time_seconds": time.time() - start_time
            }
        
        # Save to our output location
        save_json(inliner_result, str(output_file))
        
        # Extract bodies
        original_body = inliner_result.get("original_test_method_body", "")
        updated_body = inliner_result.get("updated_test_method_body", "")
        status = inliner_result.get("status", "unknown")
        inliner_error = inliner_result.get("error")
        
        # Determine which body to use
        used_original = False
        if status != "success" or not updated_body or inliner_error:
            logger.warning("Inlining failed or returned empty, using original body")
            inlined_body = original_body
            used_original = True
        else:
            inlined_body = updated_body
        
        execution_time = time.time() - start_time
        
        if used_original:
            logger.info(f"⚠ Inlining failed, using original body ({execution_time:.1f}s)")
        else:
            logger.info(f"✓ Test method inlined successfully ({execution_time:.1f}s)")
        
        return {
            "status": "success",
            "output_file": str(output_file),
            "inlined_body": inlined_body,
            "original_body": original_body,
            "used_original": used_original,
            "error": inliner_error,
            "execution_time_seconds": execution_time
        }
    
    except subprocess.TimeoutExpired:
        error_msg = "Inlining timed out (300s)"
        logger.error(error_msg)
        
        # Try to extract original body from call graph JSON as fallback
        try:
            call_graph_data = load_json(call_graph_path)
            body_lines = call_graph_data.get("test_method", {}).get("body", {}).get("lines", [])
            if body_lines:
                original_body = "\n".join(body_lines)
                logger.info("Extracted original test method body from call graph")
                
                # Save a minimal result with original body
                fallback_result = {
                    "status": "error",
                    "original_test_method_body": original_body,
                    "updated_test_method_body": "",
                    "error": error_msg
                }
                save_json(fallback_result, str(output_file))
                
                return {
                    "status": "error",
                    "output_file": str(output_file),
                    "inlined_body": None,
                    "original_body": original_body,
                    "used_original": False,
                    "error": error_msg,
                    "execution_time_seconds": time.time() - start_time
                }
        except Exception as e:
            logger.debug(f"Could not extract original body from call graph: {e}")
        
        return {
            "status": "error",
            "output_file": None,
            "inlined_body": None,
            "original_body": None,
            "used_original": False,
            "error": error_msg,
            "execution_time_seconds": time.time() - start_time
        }
    
    except Exception as e:
        error_msg = f"Inlining failed: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        
        # Try to extract original body from call graph JSON as fallback
        try:
            call_graph_data = load_json(call_graph_path)
            body_lines = call_graph_data.get("test_method", {}).get("body", {}).get("lines", [])
            if body_lines:
                original_body = "\n".join(body_lines)
                logger.info("Extracted original test method body from call graph")
                
                # Save a minimal result with original body
                fallback_result = {
                    "status": "error",
                    "original_test_method_body": original_body,
                    "updated_test_method_body": "",
                    "error": error_msg
                }
                save_json(fallback_result, str(output_file))
                
                return {
                    "status": "error",
                    "output_file": str(output_file),
                    "inlined_body": None,
                    "original_body": original_body,
                    "used_original": False,
                    "error": error_msg,
                    "execution_time_seconds": time.time() - start_time
                }
        except Exception as ex:
            logger.debug(f"Could not extract original body from call graph: {ex}")
        
        return {
            "status": "error",
            "output_file": None,
            "inlined_body": None,
            "original_body": None,
            "used_original": False,
            "error": error_msg,
            "execution_time_seconds": time.time() - start_time
        }
