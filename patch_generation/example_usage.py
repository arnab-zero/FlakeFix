"""
Example usage of the patch generator
"""

from generate_patches import (
    load_prompt_template,
    load_test_analysis,
    generate_patch_with_gpt_oss,
    save_patch_response
)
import json


def example_basic_usage():
    """Example: Generate patch using GPT-OSS"""
    json_file = "test_analysis/routingTest.json"
    
    # Load data
    prompt_template = load_prompt_template()
    test_analysis = load_test_analysis(json_file)
    
    # Create complete prompt
    test_json_str = json.dumps(test_analysis, indent=2)
    complete_prompt = prompt_template.replace("<PASTE JSON HERE>", test_json_str)
    
    # Generate with GPT-OSS
    print("Generating with GPT-OSS...")
    gpt_oss_patch = generate_patch_with_gpt_oss(complete_prompt)
    save_patch_response(gpt_oss_patch, "output/patch.txt")


def example_custom_endpoint():
    """Example: Custom Ollama endpoint"""
    json_file = "test_analysis/routingTest.json"
    
    prompt_template = load_prompt_template()
    test_analysis = load_test_analysis(json_file)
    
    test_json_str = json.dumps(test_analysis, indent=2)
    complete_prompt = prompt_template.replace("<PASTE JSON HERE>", test_json_str)
    
    # GPT-OSS with custom endpoint
    gpt_oss_patch = generate_patch_with_gpt_oss(
        complete_prompt,
        base_url="http://custom-host:11434"
    )
    save_patch_response(gpt_oss_patch, "output/custom_patch.txt")


if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Basic usage")
    print("2. Custom endpoint")
    
    choice = input("Enter choice (1-2): ")
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_custom_endpoint()
    else:
        print("Invalid choice")
