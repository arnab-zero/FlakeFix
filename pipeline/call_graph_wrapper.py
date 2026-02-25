"""
Call Graph Generation Wrapper
Wraps java_method_analysis module
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any
import logging

from .config import CALL_GRAPH_FILE
from .utils import save_json, load_json, get_timestamp


def generate_call_graph(
    project_root: str,
    test_file: str,
    method_name: str,
    output_dir: str,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Generate call graph for test method using java_method_analysis
    
    Args:
        project_root: Path to Java project root
        test_file: Path to test file (relative to project root)
        method_name: Name of test method
        output_dir: Directory for output files
        logger: Optional logger
    
    Returns:
        {
            "status": "success" | "error",
            "output_file": "path/to/01_call_graph.json",
            "metadata": {
                "method_count": 10,
                "max_depth": 2,
                "execution_time_seconds": 2.5
            },
            "error": None | "error message"
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    output_file = Path(output_dir) / CALL_GRAPH_FILE
    
    logger.info(f"Generating call graph for {method_name}...")
    logger.debug(f"Project root: {project_root}")
    logger.debug(f"Test file: {test_file}")
    
    try:
        # Construct absolute test file path
        test_file_path = Path(test_file)
        if test_file_path.is_absolute():
            test_file_abs = test_file_path
        else:
            test_file_abs = Path(project_root) / test_file
        
        # Call java_method_analyzer.py as subprocess
        cmd = [
            sys.executable,
            "java_method_analysis/java_method_analyzer.py",
            project_root,
            str(test_file_abs),
            method_name
        ]
        
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = f"Call graph generation failed:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}\nReturn code: {result.returncode}"
            logger.error(error_msg)
            return {
                "status": "error",
                "output_file": None,
                "metadata": None,
                "error": error_msg
            }
        
        # The analyzer creates JSON file in method-analysis-output directory
        # Look for the most recent JSON file there
        
        # Try to find generated JSON file
        analysis_dir = Path("method-analysis-output")
        if analysis_dir.exists():
            json_files = list(analysis_dir.glob("*_invoked_methods_analysis.json"))
        else:
            json_files = []
        
        if json_files:
            # Use the most recent one
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            
            # Load and save to our output location
            call_graph_data = load_json(str(latest_json))
            save_json(call_graph_data, str(output_file))
            
            # Clean up original file
            latest_json.unlink()
            
            # Try to remove the directory if empty
            try:
                analysis_dir.rmdir()
            except:
                pass  # Directory not empty or other error, ignore
        else:
            # If no file found, check if output is in stdout
            if result.stdout:
                logger.warning("No JSON file found, attempting to parse from stdout")
                # This is a fallback - the analyzer should create a file
                error_msg = "Call graph JSON file not found"
                return {
                    "status": "error",
                    "output_file": None,
                    "metadata": None,
                    "error": error_msg
                }
        
        # Extract metadata
        call_graph_data = load_json(str(output_file))
        method_count = len(call_graph_data.get("test_method", {}).get("calls", []))
        max_depth = call_graph_data.get("analysis_metadata", {}).get("max_hops", 2)
        
        execution_time = time.time() - start_time
        
        logger.info(f"✓ Call graph generated successfully ({execution_time:.1f}s)")
        logger.info(f"  Methods traced: {method_count}")
        logger.info(f"  Max depth: {max_depth}")
        
        return {
            "status": "success",
            "output_file": str(output_file),
            "metadata": {
                "method_count": method_count,
                "max_depth": max_depth,
                "execution_time_seconds": execution_time
            },
            "error": None
        }
    
    except subprocess.TimeoutExpired:
        error_msg = "Call graph generation timed out (60s)"
        logger.error(error_msg)
        return {
            "status": "error",
            "output_file": None,
            "metadata": None,
            "error": error_msg
        }
    
    except Exception as e:
        error_msg = f"Call graph generation failed: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        return {
            "status": "error",
            "output_file": None,
            "metadata": None,
            "error": error_msg
        }
