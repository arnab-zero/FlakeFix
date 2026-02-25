#!/usr/bin/env python3
"""
Flaky Test Detection and Patch Generation Pipeline
CLI Entry Point
"""

"""
python run_pipeline.py \
  --project-name MyProject \
  --project-root /path/to/project \
  --test-file src/test/java/MyTest.java \
  --test-method testMethod \
  --verbose
"""

import argparse
import sys
from pathlib import Path

from pipeline.main_pipeline import run_pipeline
from pipeline.results_generator import generate_results
from pipeline.summary_generator import generate_summary


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Flaky Test Detection and Patch Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_pipeline.py --project-name MyProject --project-root /path/to/project \\
      --test-file src/test/MyTest.java --test-method testMethod

  # With custom output directory
  python run_pipeline.py --project-name MyProject --project-root /path/to/project \\
      --test-file src/test/MyTest.java --test-method testMethod \\
      --output-dir ./results

  # With verbose logging
  python run_pipeline.py --project-name MyProject --project-root /path/to/project \\
      --test-file src/test/MyTest.java --test-method testMethod --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--project-name",
        required=True,
        help="Name of the project (used for output directory naming)"
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Path to test file (relative to project-root)"
    )
    parser.add_argument(
        "--test-method",
        required=True,
        help="Name of the test method to analyze"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Base output directory (default: output)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        results = run_pipeline(
            project_name=args.project_name,
            project_root=args.project_root,
            test_file=args.test_file,
            test_method=args.test_method,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # Check if pipeline succeeded
        if results.get("status") != "success":
            print(f"\n✗ Pipeline failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # Generate results.json
        results_file = generate_results(
            pipeline_results=results,
            project_name=args.project_name,
            test_method=args.test_method,
            output_dir=args.output_dir
        )
        
        # Generate summary.md
        summary_file = generate_summary(
            pipeline_results=results,
            project_name=args.project_name,
            test_method=args.test_method,
            output_dir=args.output_dir
        )
        
        # Print summary to console
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        with open(summary_file, 'r', encoding='utf-8') as f:
            print(f.read())
        
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        print("="*70)
        
        sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
