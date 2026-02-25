"""
Patch Generation Wrapper
Wraps patch_generation module
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any
import logging

from .config import PATCH_FILE


def generate_patch(
    final_label: str,
    call_graph_path: str,
    output_dir: str,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Generate patch if label is ASYNC_AWAIT
    
    Args:
        final_label: Final flakiness label
        call_graph_path: Path to call graph JSON file
        output_dir: Directory for output files
        logger: Optional logger
    
    Returns:
        {
            "status": "success" | "skipped" | "error",
            "output_file": "path/to/05_patch.txt" | None,
            "generated": True | False,
            "skip_reason": None | "reason",
            "error": None | "error message",
            "execution_time_seconds": 3.5
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    patch_file = Path(output_dir) / PATCH_FILE
    
    # Check if label is ASYNC_AWAIT
    if final_label != "ASYNC_AWAIT":
        logger.info(f"Skipping patch generation (label is {final_label}, not ASYNC_AWAIT)")
        return {
            "status": "skipped",
            "output_file": None,
            "generated": False,
            "skip_reason": f"Patch generation only for ASYNC_AWAIT label, got {final_label}",
            "error": None,
            "execution_time_seconds": time.time() - start_time
        }
    
    logger.info(f"Generating patch for ASYNC_AWAIT flaky test...")
    
    try:
        # Add patch_generation to path
        sys.path.insert(0, "patch_generation")
        
        from generate_patches import (
            load_prompt_template,
            load_test_analysis,
            generate_patch_with_gpt_oss,
            save_patch_response
        )
        
        # Load prompt template
        prompt_template = load_prompt_template()
        
        # Load call graph (test analysis)
        test_analysis = load_test_analysis(call_graph_path)
        
        # Create complete prompt
        import json
        test_json_str = json.dumps(test_analysis, indent=2)
        complete_prompt = prompt_template.replace("<PASTE JSON HERE>", test_json_str)
        
        # Generate patch
        logger.info("  Calling GPT-OSS model...")
        patch_response = generate_patch_with_gpt_oss(complete_prompt)
        
        # Save patch
        save_patch_response(patch_response, str(patch_file))
        
        execution_time = time.time() - start_time
        logger.info(f"✓ Patch generated successfully ({execution_time:.1f}s)")
        
        return {
            "status": "success",
            "output_file": str(patch_file),
            "generated": True,
            "skip_reason": None,
            "error": None,
            "execution_time_seconds": execution_time
        }
    
    except Exception as e:
        error_msg = f"Patch generation failed: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        
        return {
            "status": "error",
            "output_file": None,
            "generated": False,
            "skip_reason": None,
            "error": error_msg,
            "execution_time_seconds": time.time() - start_time
        }
