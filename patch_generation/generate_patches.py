"""
Flaky Test Patch Generator
Generates fixes for flaky test methods using GPT-OSS:20B via Ollama
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


def load_prompt_template(prompt_file: str = "patch_generation/prompt.txt") -> str:
    """Load the prompt template from file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_test_analysis(json_file: str) -> Dict[str, Any]:
    """Load the test method analysis JSON."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_patch_with_gpt_oss(prompt: str, base_url: str = None) -> str:
    """
    Generate patch using GPT-OSS:20B model via Ollama API.
    
    Args:
        prompt: The complete prompt with test analysis
        base_url: Base URL for Ollama API endpoint
    
    Returns:
        Generated patch response
    """
    import requests
    
    if base_url is None:
        base_url = os.getenv('GPT_OSS_BASE_URL', 'http://localhost:11434')
    
    # Try chat endpoint first, then generate endpoint
    endpoints = [
        (f"{base_url}/api/chat", "chat"),
        (f"{base_url}/api/generate", "generate")
    ]
    
    for endpoint_url, endpoint_type in endpoints:
        print(f"Trying {endpoint_type} endpoint: {endpoint_url}")
        
        try:
            if endpoint_type == "chat":
                # Ollama chat API format
                payload = {
                    "model": "gpt-oss:20b-cloud",
                    "messages": [
                        {"role": "system", "content": "You are an expert at fixing flaky tests. Provide the patch directly without any thinking or reasoning process."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 8192
                    }
                }
            else:
                # Ollama generate API format
                full_prompt = "You are an expert at fixing flaky tests.\n\n" + prompt
                payload = {
                    "model": "gpt-oss:20b-cloud",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 4096
                    }
                }
            
            response = requests.post(
                endpoint_url,
                json=payload,
                timeout=300  # 5 minute timeout for large responses
            )
            
            if response.status_code == 404:
                continue  # Try next endpoint
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content based on endpoint type
            if endpoint_type == "chat" and 'message' in result:
                message = result['message']
                content = message.get('content', '')
                
                # If content is empty, check for thinking field (some models use this)
                if not content or len(content.strip()) == 0:
                    if 'thinking' in message:
                        content = message['thinking']
                        print(f"[OK] Successfully connected using {endpoint_type} endpoint (using thinking field)")
                    else:
                        print(f"[WARNING] Received empty response from model")
                        print(f"Full response: {result}")
                        return content
                else:
                    print(f"[OK] Successfully connected using {endpoint_type} endpoint")
                
                return content
            elif endpoint_type == "generate" and 'response' in result:
                content = result['response']
                print(f"[OK] Successfully connected using {endpoint_type} endpoint")
                if not content or len(content.strip()) == 0:
                    print(f"[WARNING] Received empty response from model")
                    print(f"Full response: {result}")
                return content
            else:
                raise ValueError(f"Unexpected response format: {result}")
        
        except requests.exceptions.ConnectionError as e:
            if endpoint_type == endpoints[-1][1]:  # Last endpoint
                raise ConnectionError(
                    f"Cannot connect to GPT-OSS at {base_url}. "
                    f"Please ensure:\n"
                    f"1. Ollama is running (run 'ollama serve')\n"
                    f"2. The model 'gpt-oss:20b-cloud' is available (run 'ollama list')\n"
                    f"3. The endpoint is accessible\n"
                    f"Original error: {e}"
                )
            continue
        
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Request to {endpoint_url} timed out. The model might be taking too long to respond."
            )
        except Exception as e:
            if response.status_code != 404:
                raise Exception(f"Error calling GPT-OSS API at {endpoint_url}: {e}")
            continue
    
    raise ValueError(f"Could not find a working Ollama endpoint at {base_url}")


def save_patch_response(response: str, output_file: str):
    """Save the generated patch to a file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not response or len(response.strip()) == 0:
        raise ValueError("Response is empty, cannot save patch")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response)
    
    # Verify file was written
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Saved patch to: {output_file} ({output_path.stat().st_size} bytes)")
    else:
        raise IOError(f"Failed to write patch to {output_file}")


def main(json_file: str, output_dir: str = "patch_generation/output"):
    """
    Main function to generate patches using GPT-OSS.
    
    Args:
        json_file: Path to JSON file containing test method analysis
        output_dir: Directory to save generated patches
    """
    # Load prompt template and test analysis
    print(f"Loading prompt template...")
    prompt_template = load_prompt_template()
    
    print(f"Loading test analysis from: {json_file}")
    test_analysis = load_test_analysis(json_file)
    
    # Create complete prompt by replacing placeholder with JSON
    test_json_str = json.dumps(test_analysis, indent=2)
    complete_prompt = prompt_template.replace("<PASTE JSON HERE>", test_json_str)
    
    # Extract test method name for output file naming
    test_name = test_analysis.get('test_method', {}).get('name', 'unknown_test')
    
    # Generate patch with GPT-OSS
    print("\n" + "="*60)
    print("Generating patch with GPT-OSS:20B...")
    print("="*60)
    try:
        gpt_oss_response = generate_patch_with_gpt_oss(complete_prompt)
        output_file = f"{output_dir}/{test_name}_patch.txt"
        save_patch_response(gpt_oss_response, output_file)
        print("[OK] Patch generated successfully")
    except ConnectionError as e:
        print(f"[ERROR] Connection error: {e}")
        return False
    except TimeoutError as e:
        print(f"[ERROR] Timeout error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error generating patch: {e}")
        return False
    
    print("\n" + "="*60)
    print("Patch generation complete!")
    print("="*60)
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate flaky test patches using GPT-OSS:20B via Ollama"
    )
    parser.add_argument(
        "json_file",
        help="Path to JSON file containing test method analysis"
    )
    parser.add_argument(
        "--output-dir",
        default="patch_generation/output",
        help="Directory to save generated patches (default: patch_generation/output)"
    )
    
    args = parser.parse_args()
    
    success = main(json_file=args.json_file, output_dir=args.output_dir)
    exit(0 if success else 1)
