"""
Summary Generator
Creates concise markdown summary
"""

from pathlib import Path
from typing import Dict, Any

from .config import SUMMARY_FILE, get_results_dir
from .utils import format_execution_time


def generate_summary(
    pipeline_results: Dict[str, Any],
    project_name: str,
    test_method: str,
    output_dir: str
) -> str:
    """
    Generate concise markdown summary
    
    Args:
        pipeline_results: Complete pipeline results dictionary
        project_name: Name of the project
        test_method: Name of the test method
        output_dir: Base output directory
    
    Returns:
        Path to generated summary.md file
    """
    results_dir = get_results_dir(project_name, test_method, output_dir)
    summary_file = results_dir / SUMMARY_FILE
    
    # Extract key information
    status = pipeline_results.get("status", "unknown")
    summary = pipeline_results.get("summary", {})
    
    is_flaky = summary.get("is_flaky", False)
    label = summary.get("label", "UNKNOWN")
    confidence = summary.get("confidence", "UNKNOWN")
    detection_method = summary.get("detection_method", "UNKNOWN")
    patch_generated = summary.get("patch_generated", False)
    execution_time = summary.get("execution_time_seconds", 0)
    
    test_file = pipeline_results.get("test_file", "unknown")
    
    # Get recommendation from detection results
    recommendation = "N/A"
    if "detection" in pipeline_results.get("steps", {}):
        detection = pipeline_results["steps"]["detection"]
        if detection.get("status") == "success":
            # Try to get from LLM results
            if "zero_shot_result" in detection:
                recommendation = detection["zero_shot_result"].get("recommendation", "N/A")
            elif "one_shot_result" in detection:
                recommendation = detection["one_shot_result"].get("recommendation", "N/A")
    
    # Build markdown content
    lines = [
        "# Flaky Test Detection Summary",
        "",
        "## Test Information",
        f"- **Project**: {project_name}",
        f"- **Test File**: {test_file}",
        f"- **Test Method**: {test_method}",
        "",
        "## Detection Results",
        f"- **Status**: {status.upper()}",
        f"- **Is Flaky**: {'Yes' if is_flaky else 'No'}",
        f"- **Flakiness Type**: {label}",
        f"- **Confidence**: {confidence}",
        f"- **Detection Method**: {detection_method}",
        "",
        "## Patch Generation",
        f"- **Patch Generated**: {'Yes' if patch_generated else 'No'}",
    ]
    
    if patch_generated:
        patch_file = pipeline_results.get("steps", {}).get("patch", {}).get("output_file")
        if patch_file:
            lines.append(f"- **Patch File**: {patch_file}")
    elif is_flaky and label == "ASYNC_AWAIT":
        patch_error = pipeline_results.get("steps", {}).get("patch", {}).get("error")
        if patch_error:
            lines.append(f"- **Patch Error**: {patch_error}")
    
    lines.extend([
        "",
        "## Recommendation",
        f"{recommendation}",
        "",
        "## Execution Time",
        f"{format_execution_time(execution_time)}",
        ""
    ])
    
    # Write to file
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return str(summary_file)
