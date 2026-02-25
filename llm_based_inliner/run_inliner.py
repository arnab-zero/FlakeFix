#!/usr/bin/env python3
"""
Enhanced runner for LLM-driven method inlining
Uses Ollama with qwen3-coder:480b-cloud model
"""

import json
import sys
import os
from pathlib import Path

print("=" * 100)
print("Enhanced Method Inliner with Consistency Checking")
print("Using Ollama with qwen3-coder:480b-cloud model")
print("=" * 100)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from method_inliner import MethodInliner


def prepare_input(analysis_path: str) -> dict:
    """Extract and prepare input from analysis JSON"""
    print("\n[1/3] Preparing input data...")
    
    with open(analysis_path, "r") as f:
        analysis_json = json.load(f)
    
    test_data = analysis_json["test_method"]
    
    def extract_method_calls(method_data):
        calls = []
        for call in method_data.get("calls", []):
            call_entry = {
                "full_qualified_name": call["full_qualified_name"],
                "name": call["name"],
                "class_name": call["class_name"],
                "package_name": call["package_name"],
                "body": {
                    "full_text": call["body"]["full_text"],
                    "lines": call["body"]["lines"]
                },
                "calls": extract_method_calls(call)
            }
            calls.append(call_entry)
        return calls
    
    prepared = {
        "test_method": {
            "full_qualified_name": test_data["full_qualified_name"],
            "name": test_data["name"],
            "class_name": test_data["class_name"],
            "package_name": test_data["package_name"],
            "file_path": test_data.get("file_path", ""),
            "body": {
                "full_text": test_data["body"]["full_text"],
                "lines": test_data["body"]["lines"]
            },
            "calls": extract_method_calls(test_data)
        }
    }
    
    # Log statistics
    hop1_count = len(prepared["test_method"]["calls"])
    hop2_count = sum(
        len(call["calls"]) for call in prepared["test_method"]["calls"]
    )
    print(f"  [OK] Test method: {prepared['test_method']['full_qualified_name']}")
    print(f"  [OK] Hop-1 methods: {hop1_count}")
    print(f"  [OK] Hop-2 methods: {hop2_count}")
    
    return prepared


def run_inlining(
    prepared_data: dict,
    model_name: str = None,
    log_file: str = "./inliner_log.txt"
) -> dict:
    """Run the LLM-based inlining with consistency checking"""
    print("\n[2/3] Running LLM-based inlining with consistency checks...")
    
    if model_name is None:
        model_name = os.getenv("INLINER_MODEL", "qwen3-coder:480b-cloud")
    
    print(f"  Model: {model_name}")
    print(f"  Log file: {log_file}")
    print()
    
    inliner = MethodInliner(
        model_name=model_name,
        log_file=log_file
    )
    result = inliner.process(prepared_data)
    
    if result["success"]:
        print(f"  [OK] Inlining completed successfully")
    else:
        print(f"  [ERROR] Inlining failed: {result.get('error', 'Unknown error')}")
    
    return result


def save_output(
    result: dict,
    output_path: str = None,
    log_file: str = "./inliner_log.txt"
) -> str:
    """Save results to output file"""
    print("\n[3/3] Saving results...")
    
    if output_path is None:
        output_path = "./output/inlining_result.json"
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "status": "success" if result["success"] else "failed",
        "original_test_method_body": result.get("original_test_body"),
        "updated_test_method_body": result.get("updated_test_body"),
        "error": result.get("error"),
        "log_file": log_file
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"  [OK] Results saved to {output_path}")
    
    return output_path


def print_summary(result: dict, output_path: str, log_file: str):
    """Print final summary"""
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if result["success"]:
        print("[SUCCESS] Method inlining completed with consistency checks")
    else:
        print(f"[FAILED] {result.get('error', 'Unknown error')}")
    
    print(f"\nOutput file: {output_path}")
    print(f"Detailed log: {log_file}")
    
    if result.get("original_test_body") and result.get("updated_test_body"):
        orig_size = len(result["original_test_body"])
        new_size = len(result["updated_test_body"])
        print(f"\nOriginal test method: {orig_size} chars")
        print(f"Inlined test method: {new_size} chars")
        print(f"Size change: {new_size - orig_size:+d} chars")
    
    print("\nFor detailed logs with all prompts and responses, see: " + log_file)
    print("=" * 100)


def main():
    """Main orchestration"""
    import sys
    
    # Determine input path from command line argument or hardcoded paths
    input_path = None
    
    # Check if file path provided as command line argument
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if Path(candidate).exists():
            input_path = candidate
        else:
            print(f"[ERROR] Error: Provided file not found: {candidate}")
            return 1
    else:
        # Fall back to hardcoded paths for backward compatibility
        for candidate in [
            "../routingTest_call_analysis.json",
            "./routingTest_call_analysis.json",
            "./analysis_metadata.json"
        ]:
            if Path(candidate).exists():
                input_path = candidate
                break
    
    if input_path is None:
        print("[ERROR] Error: analysis JSON not found")
        print("Usage: python run_inliner.py <path_to_call_graph.json>")
        print("\nOr place one of these files in the expected location:")
        print("  - ../routingTest_call_analysis.json")
        print("  - ./routingTest_call_analysis.json")
        print("  - ./analysis_metadata.json")
        return 1
    
    print(f"\nInput: {input_path}")
    print("=" * 100)
    
    try:
        # Determine log file path
        log_file = os.getenv("LOG_FILE", "./inliner_log.txt")
        
        # Step 1: Prepare input
        prepared = prepare_input(input_path)
        
        # Step 2: Run inlining
        result = run_inlining(prepared, log_file=log_file)
        
        # Step 3: Save output
        output_path = os.getenv("OUTPUT_PATH", "./output/inlining_result.json")
        output_file = save_output(result, output_path=output_path, log_file=log_file)
        
        # Step 4: Print summary
        print_summary(result, output_file, log_file)
        
        # Return status
        return 0 if result["success"] else 1
            
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
