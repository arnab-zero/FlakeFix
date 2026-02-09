#!/usr/bin/env python3
"""
Example: Using JSON output from Java Method Analyzer
"""

"""
python java_method_analyzer.py "C:\Downloads\mercury-main" "C:\Downloads\mercury-main\system\platform-core\src\test\java\org\pla
tformlambda\core\MulticastTest.java" routingTest --json-output analysis.json
"""


import json
import sys

# Example 1: Load and explore JSON output
def example_load_json():
    print("="*80)
    print("Example 1: Loading and Exploring JSON Output")
    print("="*80)
    
    # Load the JSON file
    with open('analysis.json', 'r') as f:
        data = json.load(f)
    
    # Access metadata
    metadata = data['analysis_metadata']
    print(f"\nAnalysis Date: {metadata['analysis_timestamp']}")
    print(f"Project: {metadata['project_root']}")
    print(f"Max Hops: {metadata['max_hops']}")
    
    # Access test method
    test_method = data['test_method']
    print(f"\nTest Method: {test_method['full_qualified_name']}")
    print(f"Location: {test_method['file_path']}")
    print(f"Lines: {test_method['location']['start_line']}-{test_method['location']['end_line']}")
    
    # Access statistics
    stats = data['statistics']
    print(f"\nStatistics:")
    print(f"  Total methods traced: {stats['total_methods_traced']}")
    print(f"  Total calls: {stats['total_method_calls_found']}")
    print(f"  Library methods skipped: {stats['library_methods_skipped']}")
    print(f"  Files involved: {stats['unique_files_involved']}")
    print(f"  Packages involved: {stats['unique_packages_involved']}")
    
    print(f"\nCalls by hop level:")
    for hop, count in stats['calls_by_hop_level'].items():
        print(f"  Hop {hop}: {count} calls")


# Example 2: Traverse the call hierarchy
def example_traverse_hierarchy():
    print("\n" + "="*80)
    print("Example 2: Traversing Call Hierarchy")
    print("="*80 + "\n")
    
    with open('analysis.json', 'r') as f:
        data = json.load(f)
    
    def print_calls(method, indent=0):
        """Recursively print method call tree"""
        prefix = "  " * indent
        symbol = "└─" if indent > 0 else "📍"
        
        print(f"{prefix}{symbol} {method['name']} ({method['class_name']})")
        print(f"{prefix}   {method['file_path']}:{method['location']['start_line']}")
        
        for call in method.get('calls', []):
            print_calls(call, indent + 1)
    
    print_calls(data['test_method'])


# Example 3: Find methods by package
def example_find_by_package():
    print("\n" + "="*80)
    print("Example 3: Finding Methods by Package")
    print("="*80 + "\n")
    
    with open('analysis.json', 'r') as f:
        data = json.load(f)
    
    target_package = "com.example.security"
    
    def find_in_package(method, target, results):
        """Find all methods in a specific package"""
        if method['package_name'] == target:
            results.append({
                'name': method['name'],
                'class': method['class_name'],
                'file': method['file_path'],
                'lines': f"{method['location']['start_line']}-{method['location']['end_line']}"
            })
        
        for call in method.get('calls', []):
            find_in_package(call, target, results)
        
        return results
    
    methods = find_in_package(data['test_method'], target_package, [])
    
    print(f"Methods in package '{target_package}':")
    for method in methods:
        print(f"  • {method['class']}.{method['name']}")
        print(f"    {method['file']} (lines {method['lines']})")


# Example 4: Extract method bodies
def example_extract_bodies():
    print("\n" + "="*80)
    print("Example 4: Extracting Method Bodies")
    print("="*80 + "\n")
    
    with open('analysis.json', 'r') as f:
        data = json.load(f)
    
    def extract_bodies(method, class_filter=None):
        """Extract method bodies, optionally filtered by class"""
        bodies = []
        
        if class_filter is None or method['class_name'] == class_filter:
            bodies.append({
                'method': method['full_qualified_name'],
                'body': method['body']['full_text']
            })
        
        for call in method.get('calls', []):
            bodies.extend(extract_bodies(call, class_filter))
        
        return bodies
    
    # Get all UserService methods
    service_methods = extract_bodies(data['test_method'], 'UserService')
    
    print("UserService methods:")
    for method in service_methods:
        print(f"\n{method['method']}:")
        print("-" * 60)
        print(method['body'][:200] + "..." if len(method['body']) > 200 else method['body'])


# Example 5: Build dependency graph
def example_dependency_graph():
    print("\n" + "="*80)
    print("Example 5: Building Dependency Graph")
    print("="*80 + "\n")
    
    with open('analysis.json', 'r') as f:
        data = json.load(f)
    
    def build_edges(method, parent_fqn=None):
        """Build list of edges for graph"""
        edges = []
        method_fqn = method['full_qualified_name']
        
        if parent_fqn:
            edges.append((parent_fqn, method_fqn))
        
        for call in method.get('calls', []):
            edges.extend(build_edges(call, method_fqn))
        
        return edges
    
    edges = build_edges(data['test_method'])
    
    print("Dependency edges:")
    for parent, child in edges[:10]:  # Show first 10
        parent_name = parent.split('.')[-1]
        child_name = child.split('.')[-1]
        print(f"  {parent_name} → {child_name}")
    
    if len(edges) > 10:
        print(f"  ... and {len(edges) - 10} more edges")
    
    print(f"\nTotal edges: {len(edges)}")


# Example 6: Find all paths to a method
def example_find_paths():
    print("\n" + "="*80)
    print("Example 6: Finding All Paths to a Method")
    print("="*80 + "\n")
    
    with open('analysis.json', 'r') as f:
        data = json.load(f)
    
    target_method = "encode"
    
    def find_paths(method, target, current_path=[]):
        """Find all call paths to a target method"""
        paths = []
        new_path = current_path + [method['name']]
        
        if method['name'] == target:
            paths.append(new_path)
        
        for call in method.get('calls', []):
            paths.extend(find_paths(call, target, new_path))
        
        return paths
    
    paths = find_paths(data['test_method'], target_method)
    
    print(f"All paths to method '{target_method}':")
    for i, path in enumerate(paths, 1):
        print(f"\n  Path {i}: {' → '.join(path)}")
    
    if not paths:
        print(f"  No paths found to '{target_method}'")


# Example 7: Generate statistics report
def example_statistics_report():
    print("\n" + "="*80)
    print("Example 7: Generating Statistics Report")
    print("="*80 + "\n")
    
    with open('analysis.json', 'r') as f:
        data = json.load(f)
    
    # Count methods by package
    package_counts = {}
    
    def count_by_package(method):
        pkg = method['package_name']
        package_counts[pkg] = package_counts.get(pkg, 0) + 1
        
        for call in method.get('calls', []):
            count_by_package(call)
    
    count_by_package(data['test_method'])
    
    print("Methods by package:")
    for pkg, count in sorted(package_counts.items(), key=lambda x: -x[1]):
        print(f"  {pkg}: {count} methods")
    
    # Library methods report
    lib_methods = data['library_methods_skipped']
    print(f"\nLibrary methods skipped: {len(lib_methods)}")
    
    by_reason = {}
    for lib in lib_methods:
        reason = lib['reason']
        by_reason[reason] = by_reason.get(reason, 0) + 1
    
    print("\nBy reason:")
    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


# Example 8: Export to different formats
def example_export_formats():
    print("\n" + "="*80)
    print("Example 8: Exporting to Different Formats")
    print("="*80 + "\n")
    
    with open('analysis.json', 'r') as f:
        data = json.load(f)
    
    # Export to CSV (method list)
    print("Generating CSV export...")
    with open('methods.csv', 'w') as f:
        f.write("Method,Class,Package,File,StartLine,EndLine\n")
        
        def write_methods(method):
            f.write(f'"{method["name"]}","{method["class_name"]}","{method["package_name"]}",'
                   f'"{method["file_path"]}",{method["location"]["start_line"]},'
                   f'{method["location"]["end_line"]}\n')
            
            for call in method.get('calls', []):
                write_methods(call)
        
        write_methods(data['test_method'])
    
    print("✓ CSV saved to methods.csv")
    
    # Export to Markdown
    print("Generating Markdown report...")
    with open('report.md', 'w') as f:
        f.write(f"# Test Method Analysis\n\n")
        f.write(f"**Test Method:** {data['test_method']['full_qualified_name']}\n\n")
        f.write(f"**Date:** {data['analysis_metadata']['analysis_timestamp']}\n\n")
        
        f.write("## Statistics\n\n")
        stats = data['statistics']
        f.write(f"- Total methods: {stats['total_methods_traced']}\n")
        f.write(f"- Total calls: {stats['total_method_calls_found']}\n")
        f.write(f"- Library methods skipped: {stats['library_methods_skipped']}\n\n")
        
        f.write("## Call Hierarchy\n\n")
        
        def write_hierarchy(method, indent=0):
            prefix = "  " * indent
            f.write(f"{prefix}- **{method['name']}** ({method['class_name']})\n")
            f.write(f"{prefix}  - File: `{method['file_path']}`\n")
            f.write(f"{prefix}  - Lines: {method['location']['start_line']}-{method['location']['end_line']}\n")
            
            if method.get('calls'):
                for call in method['calls']:
                    write_hierarchy(call, indent + 1)
        
        write_hierarchy(data['test_method'])
    
    print("✓ Markdown report saved to report.md")


def main():
    """Run all examples"""
    
    # Check if analysis.json exists
    try:
        with open('analysis.json', 'r') as f:
            pass
    except FileNotFoundError:
        print("Error: analysis.json not found!")
        print("\nPlease run the analyzer first:")
        print("  python java_method_analyzer.py example_project \\")
        print("      src/test/java/com/example/service/UserServiceTest.java \\")
        print("      testAuthenticateUser --json-output analysis.json")
        sys.exit(1)
    
    # Run examples
    try:
        example_load_json()
        example_traverse_hierarchy()
        example_find_by_package()
        example_extract_bodies()
        example_dependency_graph()
        example_find_paths()
        example_statistics_report()
        example_export_formats()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()