#!/usr/bin/env python3
"""
Flaky Test Identification Orchestrator

Orchestrates the analysis of Java test methods:
1. Extract test method + 2-hop call graph using java_method_analyzer

Usage:
    python flaky_test_identification_orchestrator.py <project_root> <test_file_path> <test_method_name>

Example:
    python flaky_test_identification_orchestrator.py \
        "C:/Downloads/mercury-main" \
        "C:\Downloads\mercury-main\system\platform-core\src\test\java\org\platformlambda\core\MulticastTest.java" \
        "routingTest"
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Tuple

# Import components
from java_method_analyzer import analyze_test_method


class FlakyTestOrchestrator:
    """
    Orchestrates the test method analysis.
    """
    
    def __init__(self, project_root: str):
        """
        Initialize the orchestrator.
        
        Args:
            project_root: Path to Java project root directory
        """
        self.project_root = Path(project_root).resolve()
    
    def analyze_test(self, test_file_path: str, test_method_name: str) -> Dict:
        """
        Run analysis on a test method.
        
        Args:
            test_file_path: Path to test file (relative or absolute)
            test_method_name: Name of the test method
            
        Returns:
            Dictionary with analysis results:
            {
                'test_info': {...},
                'call_graph_json': {...},
                'call_graph_file': 'path/to/saved.json'
            }
        """
        print("=" * 80)
        print("FLAKY TEST IDENTIFICATION PIPELINE")
        print("=" * 80)
        print(f"\nProject: {self.project_root}")
        print(f"Test File: {test_file_path}")
        print(f"Test Method: {test_method_name}")
        print("\n" + "=" * 80)
        
        # Extract test method + call graph
        print("\n[Stage 1] Extracting test method and call graph (2-hop)...")
        print("-" * 80)
        
        call_graph_json, call_graph_file = analyze_test_method(
            str(self.project_root),
            test_file_path,
            test_method_name,
            max_hops=2
        )
        
        if "error" in call_graph_json:
            print(f"\n❌ Error: {call_graph_json['error']}\n")
            return {
                'error': call_graph_json['error'],
                'stage': 'call_graph_extraction'
            }
        
        print(f"✓ Call graph extracted successfully")
        print(f"✓ Saved to: {call_graph_file}")
        print(f"✓ Methods traced: {call_graph_json['statistics']['total_methods_traced']}")
        print(f"✓ Library methods skipped: {call_graph_json['statistics']['library_methods_skipped']}")
        
        # Prepare result
        result = {
            'test_info': {
                'class_name': call_graph_json['test_method']['class_name'],
                'method_name': test_method_name,
                'package': call_graph_json['test_method']['package_name'],
                'file_path': test_file_path
            },
            'call_graph_json': call_graph_json,
            'call_graph_file': call_graph_file
        }
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nOutput:")
        print(f"  • Call graph JSON: {call_graph_file}")
        print("=" * 80 + "\n")
        
        return result


def main():
    """Main entry point for command-line usage"""
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nError: Missing required arguments")
        print("\nRequired arguments:")
        print("  1. project_root     - Path to Java project root directory")
        print("  2. test_file_path   - Path to test file (relative or absolute)")
        print("  3. test_method_name - Name of the test method to analyze")
        sys.exit(1)
    
    project_root = sys.argv[1]
    test_file_path = sys.argv[2]
    test_method_name = sys.argv[3]
    
    # Validate project root
    if not os.path.isdir(project_root):
        print(f"\n❌ Error: Project root directory not found: {project_root}\n")
        sys.exit(1)
    
    # Create orchestrator
    orchestrator = FlakyTestOrchestrator(project_root)
    
    # Run analysis
    result = orchestrator.analyze_test(test_file_path, test_method_name)
    
    # Check for errors
    if 'error' in result:
        print(f"\n❌ Analysis failed at stage: {result.get('stage', 'unknown')}")
        print(f"Error: {result['error']}\n")
        sys.exit(1)
    
    print("\n✅ Analysis complete!")
    print(f"\nTo view results:")
    print(f"  • Call graph: {result['call_graph_file']}")
    print()


if __name__ == "__main__":
    main()
