"""
Pipeline Utility Functions
"""

import json
import re
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to output file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from JSON file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_llm_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response, extracting JSON even if wrapped in markdown
    
    Args:
        response_text: Raw LLM response text
    
    Returns:
        Parsed JSON data
    
    Raises:
        ValueError: If JSON cannot be parsed
    """
    # Try direct JSON parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try extracting any JSON object
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not parse JSON from response: {response_text[:200]}...")


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format
    
    Returns:
        ISO-8601 formatted timestamp
    """
    return datetime.now(timezone.utc).isoformat()


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in human-readable format
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "2m 30s" or "45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.1f}s"


def validate_file_exists(file_path: str, description: str = "File") -> None:
    """
    Validate that a file exists
    
    Args:
        file_path: Path to file
        description: Description for error message
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")


def validate_directory_exists(dir_path: str, description: str = "Directory") -> None:
    """
    Validate that a directory exists
    
    Args:
        dir_path: Path to directory
        description: Description for error message
    
    Raises:
        NotADirectoryError: If directory doesn't exist
    """
    path = Path(dir_path)
    if not path.exists():
        raise NotADirectoryError(f"{description} not found: {dir_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {dir_path}")
