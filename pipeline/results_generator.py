"""
Results Generator
Creates comprehensive results.json file
"""

from pathlib import Path
from typing import Dict, Any

from .config import RESULTS_FILE, get_results_dir
from .utils import save_json


def generate_results(
    pipeline_results: Dict[str, Any],
    project_name: str,
    test_method: str,
    output_dir: str
) -> str:
    """
    Generate comprehensive results.json file
    
    Args:
        pipeline_results: Complete pipeline results dictionary
        project_name: Name of the project
        test_method: Name of the test method
        output_dir: Base output directory
    
    Returns:
        Path to generated results.json file
    """
    results_dir = get_results_dir(project_name, test_method, output_dir)
    results_file = results_dir / RESULTS_FILE
    
    # Save results
    save_json(pipeline_results, str(results_file))
    
    return str(results_file)
