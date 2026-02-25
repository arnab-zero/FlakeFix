import os
import json
import csv
import requests
from pathlib import Path
from prompt import build_flaky_prompt
import re


def run_inference(java_code, model="gpt-oss:20b-cloud", timeout=120):
    """Run inference on a Java test method."""
    prompt = build_flaky_prompt(java_code)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }
    
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception as e:
        return f"ERROR: {str(e)}"


def parse_response(response):
    """Parse the model response to extract label, evidence, and confidence."""
    label = "UNKNOWN"
    evidence = []
    confidence = "UNKNOWN"
    
    # Extract label
    label_match = re.search(r'Label:\s*([A-Z_]+)', response, re.IGNORECASE)
    if label_match:
        label = label_match.group(1).strip()
    
    # Extract evidence (bullet points)
    evidence_section = re.search(r'Evidence:(.*?)(?:Confidence:|$)', response, re.DOTALL | re.IGNORECASE)
    if evidence_section:
        evidence_text = evidence_section.group(1)
        evidence = [line.strip().lstrip('-').strip() 
                   for line in evidence_text.split('\n') 
                   if line.strip() and line.strip().startswith('-')]
    
    # Extract confidence
    conf_match = re.search(r'Confidence:\s*([A-Z]+)', response, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).strip()
    
    return label, evidence, confidence


def process_directory(input_dir, output_dir, csv_output_path):
    """Process all files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    Path(csv_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get all .txt files
    files = sorted(list(input_path.glob("*.txt")))
    
    if not files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(files)} files in {input_dir}")
    
    # Prepare CSV data
    csv_data = []
    
    for idx, file_path in enumerate(files, 1):
        filename = file_path.name
        print(f"[{idx}/{len(files)}] Processing: {filename}")
        
        try:
            # Read Java code
            with open(file_path, 'r', encoding='utf-8') as f:
                java_code = f.read()
            
            # Run inference
            response = run_inference(java_code)
            
            # Parse response
            label, evidence, confidence = parse_response(response)
            
            # Save full response to output file
            output_file = output_path / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            
            # Add to CSV data
            csv_data.append({
                'filename': filename,
                'label': label,
                'evidence': ' | '.join(evidence),
                'confidence': confidence,
                'full_response': response.replace('\n', ' ')
            })
            
            print(f"  Label: {label}, Confidence: {confidence}")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            csv_data.append({
                'filename': filename,
                'label': 'ERROR',
                'evidence': str(e),
                'confidence': 'N/A',
                'full_response': str(e)
            })
    
    # Write CSV file
    print(f"\nWriting CSV to {csv_output_path}")
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'label', 'evidence', 'confidence', 'full_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Completed processing {len(files)} files")
    print(f"Outputs saved to: {output_dir}")
    print(f"CSV saved to: {csv_output_path}")


def main():
    # Process test_files_v0
    print("=" * 60)
    print("Processing FlakyCat_data/test_files_v0")
    print("=" * 60)
    process_directory(
        "FlakyCat_data/test_files_v0",
        "outputs/test_files_v0",
        "csv_outputs/test_files_v0.csv"
    )
    
    print("\n" + "=" * 60)
    print("Processing FlakyCat_data/test_files_v12")
    print("=" * 60)
    process_directory(
        "FlakyCat_data/test_files_v12",
        "outputs/test_files_v12",
        "csv_outputs/test_files_v12.csv"
    )
    
    print("\n" + "=" * 60)
    print("All processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
