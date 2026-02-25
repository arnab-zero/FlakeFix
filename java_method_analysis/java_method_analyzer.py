#!/usr/bin/env python3
"""
Java Test Method Call Analyzer using JavaLang Parser
Finds all methods called by a test method up to 2 hops deep,
tracking their locations across different files and ignoring library methods.

Requirements:
    pip install javalang
"""

"""
python java_method_extractor.py "C:\\Downloads\\mercury-main" "C:\\Downloads\\mercury-main\\system\\platform-core\\src\\test\\java\\org\\platformlambda\\core\\MulticastTest.java" routingTest
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from datetime import datetime, timezone

try:
    import javalang
    from javalang.tree import MethodDeclaration, MethodInvocation, ClassDeclaration
except ImportError:
    print("Error: javalang is not installed.")
    print("Please install it using: pip install javalang")
    sys.exit(1)


@dataclass
class MethodInfo:
    """Information about a method"""
    name: str
    file_path: str
    start_line: int
    end_line: int
    class_name: str
    package_name: str
    parameters: List[str] = field(default_factory=list)
    body_lines: List[str] = field(default_factory=list)
    
    @property
    def body(self) -> str:
        """Get the method body as a string"""
        return '\n'.join(self.body_lines)
    
    @property
    def full_name(self) -> str:
        """Get the full qualified name"""
        if self.package_name:
            return f"{self.package_name}.{self.class_name}.{self.name}"
        return f"{self.class_name}.{self.name}"


@dataclass
class MethodCall:
    """Information about a method call"""
    caller_method: str
    called_method: str
    called_method_info: Optional[MethodInfo]
    call_location: str
    hop_level: int
    is_library_method: bool = False


class JavaMethodAnalyzer:
    def __init__(self, project_root: str):
        """Initialize the analyzer with project root directory"""
        self.project_root = Path(project_root).resolve()
        
        # Cache for parsed files and method definitions
        self.file_cache: Dict[str, str] = {}
        self.parsed_trees: Dict[str, javalang.tree.CompilationUnit] = {}
        self.method_index: Dict[str, List[MethodInfo]] = defaultdict(list)
        self.class_to_file: Dict[str, str] = {}
        self.package_to_files: Dict[str, List[str]] = defaultdict(list)
        self.imports_by_file: Dict[str, Set[str]] = {}
        
        # Build index of all Java files
        self._index_project()
    
    def _index_project(self):
        """Index all Java files in the project"""
        print(f"Indexing project: {self.project_root}")
        print("-" * 80)
        
        indexed_count = 0
        failed_count = 0
        
        for java_file in self.project_root.rglob("*.java"):
            try:
                with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    self.file_cache[str(java_file)] = content
                    
                    # Parse the file
                    try:
                        tree = javalang.parse.parse(content)
                        self.parsed_trees[str(java_file)] = tree
                        
                        # Extract package and class information
                        self._extract_file_info(tree, str(java_file), content)
                        indexed_count += 1
                    except javalang.parser.JavaSyntaxError:
                        failed_count += 1
                        
            except Exception as e:
                print(f"Warning: Could not read {java_file}: {e}")
        
        print(f"[OK] Successfully indexed {indexed_count} Java files")
        if failed_count > 0:
            print(f"[WARN] Failed to parse {failed_count} files (syntax errors)")
        print(f"[OK] Found {len(self.class_to_file)} classes")
        print(f"[OK] Indexed {sum(len(methods) for methods in self.method_index.values())} methods")
        print("-" * 80)
    
    def _extract_file_info(self, tree: javalang.tree.CompilationUnit, file_path: str, content: str):
        """Extract package, class, and method information from parsed tree"""
        package_name = tree.package.name if tree.package else ""
        
        # Store imports
        imports = set()
        if tree.imports:
            for imp in tree.imports:
                imports.add(imp.path)
        self.imports_by_file[file_path] = imports
        
        # Add to package mapping
        if package_name:
            self.package_to_files[package_name].append(file_path)
        
        # Extract classes and methods
        for path, node in tree.filter(ClassDeclaration):
            class_name = node.name
            full_class_name = f"{package_name}.{class_name}" if package_name else class_name
            self.class_to_file[full_class_name] = file_path
            self.class_to_file[class_name] = file_path
            
            # Extract methods from this class
            for method_path, method_node in node.filter(MethodDeclaration):
                method_info = self._extract_method_info(
                    method_node, class_name, package_name, file_path, content
                )
                if method_info:
                    self.method_index[method_info.name].append(method_info)
                    full_key = f"{class_name}.{method_info.name}"
                    self.method_index[full_key].append(method_info)
    
    def _get_line_number(self, content: str, position: int) -> int:
        """Get line number from character position in content"""
        if position is None:
            return 0
        return content[:position].count('\n') + 1
    
    def _extract_method_info(self, method_node: MethodDeclaration, class_name: str, 
                            package_name: str, file_path: str, content: str) -> Optional[MethodInfo]:
        """Extract detailed information about a method"""
        try:
            method_name = method_node.name
            
            # Get line numbers
            start_line = method_node.position.line if method_node.position else 0
            
            # Extract parameters
            parameters = []
            if method_node.parameters:
                for param in method_node.parameters:
                    param_type = param.type.name if hasattr(param.type, 'name') else str(param.type)
                    parameters.append(f"{param_type} {param.name}")
            
            # Get method body by extracting from source
            body_lines = self._extract_method_body(content, start_line, method_name)
            end_line = start_line + len(body_lines) - 1 if body_lines else start_line
            
            return MethodInfo(
                name=method_name,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                class_name=class_name,
                package_name=package_name,
                parameters=parameters,
                body_lines=body_lines
            )
        except Exception as e:
            return None
    
    def _extract_method_body(self, content: str, start_line: int, method_name: str) -> List[str]:
        """Extract method body from source code"""
        lines = content.split('\n')
        
        # Find the method declaration line
        method_start = None
        for i in range(max(0, start_line - 1), min(len(lines), start_line + 10)):
            if method_name in lines[i] and '(' in lines[i]:
                method_start = i
                break
        
        if method_start is None:
            return []
        
        # Find opening brace
        brace_line = method_start
        while brace_line < len(lines) and '{' not in lines[brace_line]:
            brace_line += 1
        
        if brace_line >= len(lines):
            return []
        
        # Count braces to find method end
        brace_count = 0
        body_lines = []
        
        for i in range(brace_line, len(lines)):
            line = lines[i]
            body_lines.append(line)
            
            for char in line:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return body_lines
        
        return body_lines
    
    def _extract_method_calls(self, method_node: MethodDeclaration) -> List[Tuple[str, Optional[str]]]:
        """Extract all method calls from a method
        Returns list of (method_name, qualifier) tuples where qualifier is the object/class name
        """
        calls = []
        
        for path, node in method_node.filter(MethodInvocation):
            method_name = node.member
            qualifier = node.qualifier if hasattr(node, 'qualifier') else None
            calls.append((method_name, qualifier))
        
        return calls
    
    def _is_library_method(self, method_name: str, class_hint: Optional[str], 
                          current_file: str) -> bool:
        """Check if a method is from a library (not in the project)"""
        # If we can't find the method in our index, it's likely a library method
        if class_hint:
            full_name = f"{class_hint}.{method_name}"
            if full_name not in self.method_index and method_name not in self.method_index:
                # Check if class is in project
                if class_hint not in self.class_to_file:
                    return True
        
        # Check if method exists in our index
        if method_name in self.method_index:
            return False
        
        # Check common Java library patterns
        library_packages = ['java.', 'javax.', 'org.junit.', 'org.mockito.', 
                          'org.springframework.', 'com.google.', 'org.apache.']
        
        if class_hint:
            for lib_package in library_packages:
                if class_hint.startswith(lib_package):
                    return True
        
        # If not found in project, assume it's a library method
        return method_name not in self.method_index
    
    def _resolve_method_location(self, method_name: str, class_hint: Optional[str], 
                                  current_file: str) -> Optional[MethodInfo]:
        """Try to resolve where a method is defined in the project"""
        
        # Try with full qualified name first
        if class_hint:
            full_key = f"{class_hint}.{method_name}"
            if full_key in self.method_index:
                candidates = self.method_index[full_key]
                if candidates:
                    return candidates[0]
            
            # Try to find the class file
            if class_hint in self.class_to_file:
                target_file = self.class_to_file[class_hint]
                tree = self.parsed_trees.get(target_file)
                if tree:
                    for path, node in tree.filter(MethodDeclaration):
                        if node.name == method_name:
                            content = self.file_cache[target_file]
                            package = tree.package.name if tree.package else ""
                            # Find the class containing this method
                            for cls_path, cls_node in tree.filter(ClassDeclaration):
                                for m_path, m_node in cls_node.filter(MethodDeclaration):
                                    if m_node.name == method_name:
                                        return self._extract_method_info(
                                            m_node, cls_node.name, package, target_file, content
                                        )
        
        # Try in the same file
        current_tree = self.parsed_trees.get(current_file)
        if current_tree:
            for path, node in current_tree.filter(MethodDeclaration):
                if node.name == method_name:
                    content = self.file_cache[current_file]
                    package = current_tree.package.name if current_tree.package else ""
                    # Find the class containing this method
                    for cls_path, cls_node in current_tree.filter(ClassDeclaration):
                        for m_path, m_node in cls_node.filter(MethodDeclaration):
                            if m_node.name == method_name:
                                return self._extract_method_info(
                                    m_node, cls_node.name, package, current_file, content
                                )
        
        # Search in method index
        if method_name in self.method_index:
            candidates = self.method_index[method_name]
            if candidates:
                # Prefer methods in the same package
                current_tree = self.parsed_trees.get(current_file)
                if current_tree and current_tree.package:
                    current_package = current_tree.package.name
                    for candidate in candidates:
                        if candidate.package_name == current_package:
                            return candidate
                
                # Return first candidate
                return candidates[0]
        
        return None
    
    def analyze_test_method_internal(self, test_file_path: str, test_method_name: str, 
                           max_hops: int = 2) -> Dict:
        """Analyze a test method and trace calls up to max_hops deep, ignoring library methods"""
        
        test_file_path = str(Path(test_file_path).resolve())
        
        if test_file_path not in self.file_cache:
            return {"error": f"File not found: {test_file_path}"}
        
        if test_file_path not in self.parsed_trees:
            return {"error": f"File could not be parsed: {test_file_path}"}
        
        # Find the test method
        test_method = None
        tree = self.parsed_trees[test_file_path]
        content = self.file_cache[test_file_path]
        package = tree.package.name if tree.package else ""
        
        for cls_path, cls_node in tree.filter(ClassDeclaration):
            for method_path, method_node in cls_node.filter(MethodDeclaration):
                if method_node.name == test_method_name:
                    test_method = self._extract_method_info(
                        method_node, cls_node.name, package, test_file_path, content
                    )
                    test_method_node = method_node
                    break
            if test_method:
                break
        
        if not test_method:
            return {"error": f"Test method '{test_method_name}' not found in {test_file_path}"}
        
        print(f"\n{'='*80}")
        print(f"Analyzing Test Method: {test_method.class_name}.{test_method_name}")
        print(f"File: {test_file_path}")
        print(f"Location: Lines {test_method.start_line}-{test_method.end_line}")
        print(f"{'='*80}\n")
        
        # Track all calls and visited methods to avoid infinite loops
        all_calls: List[MethodCall] = []
        visited: Set[str] = set()
        library_methods_skipped = set()
        
        # Queue: (method_info, method_node, current_hop_level)
        queue: List[Tuple[MethodInfo, MethodDeclaration, int]] = [(test_method, test_method_node, 0)]
        
        while queue:
            current_method, current_node, hop_level = queue.pop(0)
            
            # Skip if we've already processed this method
            method_key = current_method.full_name
            if method_key in visited:
                continue
            visited.add(method_key)
            
            # Don't process beyond max hops
            if hop_level >= max_hops:
                continue
            
            # Extract method calls
            raw_calls = self._extract_method_calls(current_node)
            
            for called_method_name, class_hint in raw_calls:
                # Check if it's a library method
                if self._is_library_method(called_method_name, class_hint, current_method.file_path):
                    library_methods_skipped.add(f"{class_hint}.{called_method_name}" if class_hint else called_method_name)
                    continue
                
                # Try to resolve the method location
                called_method_info = self._resolve_method_location(
                    called_method_name, class_hint, current_method.file_path
                )
                
                if called_method_info:
                    call_info = MethodCall(
                        caller_method=f"{current_method.class_name}.{current_method.name}",
                        called_method=f"{called_method_info.class_name}.{called_method_info.name}",
                        called_method_info=called_method_info,
                        call_location=f"{called_method_info.file_path} (lines {called_method_info.start_line}-{called_method_info.end_line})",
                        hop_level=hop_level + 1,
                        is_library_method=False
                    )
                    all_calls.append(call_info)
                    
                    # Add to queue for further processing if within hop limit
                    if hop_level + 1 < max_hops:
                        # Find the method node for next iteration
                        target_tree = self.parsed_trees.get(called_method_info.file_path)
                        if target_tree:
                            for cls_path, cls_node in target_tree.filter(ClassDeclaration):
                                if cls_node.name == called_method_info.class_name:
                                    for m_path, m_node in cls_node.filter(MethodDeclaration):
                                        if m_node.name == called_method_info.name:
                                            queue.append((called_method_info, m_node, hop_level + 1))
                                            break
        
        return {
            "test_method": test_method,
            "calls": all_calls,
            "total_methods_found": len(visited),
            "library_methods_skipped": library_methods_skipped
        }
    
    def print_results(self, results: Dict):
        """Pretty print the analysis results"""
        if "error" in results:
            print(f"\n[ERROR] Error: {results['error']}\n")
            return
        
        test_method = results["test_method"]
        calls = results["calls"]
        library_methods = results["library_methods_skipped"]
        
        print(f"\n[ANALYSIS RESULTS]")
        print(f"{'='*80}")
        print(f"Test Method: {test_method.class_name}.{test_method.name}")
        print(f"File: {test_method.file_path}")
        print(f"Lines: {test_method.start_line}-{test_method.end_line}")
        print(f"Package: {test_method.package_name or '(default)'}")
        print(f"\nTotal project methods traced: {results['total_methods_found']}")
        print(f"Library methods skipped: {len(library_methods)}")
        print(f"{'='*80}\n")
        
        # Group by hop level
        by_hop = defaultdict(list)
        for call in calls:
            by_hop[call.hop_level].append(call)
        
        for hop_level in sorted(by_hop.keys()):
            print(f"\n{'-'*80}")
            print(f"[HOP {hop_level}] Methods called {'directly by test method' if hop_level == 1 else f'at depth {hop_level}'}")
            print(f"{'-'*80}\n")
            
            for idx, call in enumerate(by_hop[hop_level], 1):
                print(f"  [{idx}] {call.caller_method} -> {call.called_method}")
                print(f"      Location: {call.call_location}")
                
                if call.called_method_info:
                    print(f"      Package: {call.called_method_info.package_name or '(default)'}")
                    if call.called_method_info.parameters:
                        print(f"      Parameters: {', '.join(call.called_method_info.parameters)}")
                    
                    # Show method body
                    if call.called_method_info.body_lines:
                        print(f"      Method Body:")
                        for line in call.called_method_info.body_lines[:15]:  # Show first 15 lines
                            print(f"         {line}")
                        if len(call.called_method_info.body_lines) > 15:
                            print(f"         ... ({len(call.called_method_info.body_lines) - 15} more lines)")
                
                print()
        
        # Show skipped library methods
        if library_methods:
            print(f"\n{'-'*80}")
            print(f"[SKIPPED] LIBRARY METHODS SKIPPED ({len(library_methods)} total)")
            print(f"{'-'*80}\n")
            for lib_method in sorted(library_methods)[:20]:  # Show first 20
                print(f"  - {lib_method}")
            if len(library_methods) > 20:
                print(f"  ... and {len(library_methods) - 20} more")

    def _methodinfo_to_json(self, mi: MethodInfo) -> Dict:
        """Convert MethodInfo to JSON-serializable dict (keeps your current attributes)."""
        return {
            "name": mi.name,
            "class_name": mi.class_name,
            "package_name": mi.package_name,
            "full_qualified_name": mi.full_name,
            "file_path": mi.file_path,
            "location": {"start_line": mi.start_line, "end_line": mi.end_line},
            "parameters": mi.parameters or [],
            "body": {
                "lines": mi.body_lines or [],
                "full_text": mi.body or ""
            }
        }

    def _build_call_tree(self, results: Dict) -> List[Dict]:
        """
        Turn flat MethodCall list into a nested call tree under test_method.calls.
        Uses MethodInfo.full_name as stable keys.
        """
        calls: List[MethodCall] = results.get("calls", [])
        test_method: MethodInfo = results["test_method"]

        # caller_full_name -> list of MethodCall
        outgoing = defaultdict(list)
        for c in calls:
            if c.called_method_info is None:
                continue
            # We stored caller_method as "Class.method" (not fully qualified),
            # so we reconstruct a stable-ish key using available info:
            # Prefer to map by actual resolved MethodInfo.full_name when possible.
            # We'll use the caller string as grouping key first, then reconcile below.
            outgoing[c.caller_method].append(c)

        # Build a lookup for resolved MethodInfo by "Class.method" too
        resolved_by_simple = {}
        # include test method
        resolved_by_simple[f"{test_method.class_name}.{test_method.name}"] = test_method
        for c in calls:
            if c.called_method_info:
                resolved_by_simple[f"{c.called_method_info.class_name}.{c.called_method_info.name}"] = c.called_method_info

        def node_for_method(mi: MethodInfo) -> Dict:
            node = self._methodinfo_to_json(mi)
            node["calls"] = []  # fill recursively
            return node

        def expand(caller_simple: str, caller_node: Dict, depth: int):
            # depth here corresponds to hop expansion; stop at max_hops-1 expansions already handled,
            # but safe-guard recursion to avoid cycles.
            if caller_simple in seen_in_path:
                return
            seen_in_path.add(caller_simple)

            for call in outgoing.get(caller_simple, []):
                callee_mi = call.called_method_info
                if not callee_mi:
                    continue
                callee_node = node_for_method(callee_mi)
                caller_node["calls"].append(callee_node)

                callee_simple = f"{callee_mi.class_name}.{callee_mi.name}"
                expand(callee_simple, callee_node, depth + 1)

            seen_in_path.remove(caller_simple)

        # Top-level calls are those directly from the test method (hop_level == 1)
        root_calls = [c for c in calls if c.hop_level == 1 and c.called_method_info is not None]
        test_simple = f"{test_method.class_name}.{test_method.name}"

        top_nodes: List[Dict] = []
        for c in root_calls:
            callee_mi = c.called_method_info
            callee_node = node_for_method(callee_mi)
            top_nodes.append(callee_node)

            seen_in_path = set()
            expand(f"{callee_mi.class_name}.{callee_mi.name}", callee_node, 1)

        return top_nodes

    def _compute_stats(self, results: Dict, max_hops: int) -> Dict:
        """Compute statistics section from existing outputs."""
        calls: List[MethodCall] = results.get("calls", [])
        library_methods = results.get("library_methods_skipped", set())

        files = set()
        packages = set()
        for c in calls:
            if c.called_method_info:
                files.add(c.called_method_info.file_path)
                if c.called_method_info.package_name:
                    packages.add(c.called_method_info.package_name)

        calls_by_hop = defaultdict(int)
        for c in calls:
            calls_by_hop[str(c.hop_level)] += 1

        return {
            "total_methods_traced": int(results.get("total_methods_found", 0)),
            "total_method_calls_found": int(len(calls)),
            "library_methods_skipped": int(len(library_methods)),
            "unique_files_involved": int(len(files)),
            "unique_packages_involved": int(len(packages)),
            "calls_by_hop_level": dict(calls_by_hop)
        }

    def _library_methods_to_json(self, library_methods: Set[str]) -> List[Dict]:
        """
        Your current code stores these as strings like:
          - "System.out.println" or "println" or "SomeClass.someMethod"
        We keep them, and fill what we can without guessing too much.
        """
        out = []
        for s in sorted(library_methods):
            qualifier = None
            method_name = s
            full_reference = s

            if "." in s:
                parts = s.split(".")
                method_name = parts[-1]
                qualifier = ".".join(parts[:-1])

            out.append({
                "method_name": method_name,
                "qualifier": qualifier,
                "full_reference": full_reference,
                "reason": "Not found in project index / treated as library method"
            })
        return out

    def _files_analyzed_to_json(self, test_method: MethodInfo, calls: List[MethodCall]) -> List[Dict]:
        """
        Build a lightweight view of involved files.
        'methods_found' = count of resolved methods whose file is that file.
        'role' = 'test' if it's the test file else 'dependency'
        """
        file_method_counts = defaultdict(int)
        file_package = {}

        # test method
        file_method_counts[test_method.file_path] += 1
        file_package[test_method.file_path] = test_method.package_name

        for c in calls:
            if c.called_method_info:
                fp = c.called_method_info.file_path
                file_method_counts[fp] += 1
                file_package[fp] = c.called_method_info.package_name

        out = []
        for fp, cnt in sorted(file_method_counts.items()):
            out.append({
                "file_path": fp,
                "package_name": file_package.get(fp, ""),
                "methods_found": int(cnt),
                "role": "test" if fp == test_method.file_path else "dependency"
            })
        return out

    def _packages_involved_to_json(self, test_method: MethodInfo, calls: List[MethodCall]) -> List[Dict]:
        """
        packages_involved: package -> {files_count, methods_count}
        """
        pkg_files = defaultdict(set)
        pkg_methods = defaultdict(int)

        # test
        pkg = test_method.package_name or ""
        pkg_files[pkg].add(test_method.file_path)
        pkg_methods[pkg] += 1

        for c in calls:
            if c.called_method_info:
                pkg2 = c.called_method_info.package_name or ""
                pkg_files[pkg2].add(c.called_method_info.file_path)
                pkg_methods[pkg2] += 1

        out = []
        for pkg_name in sorted(pkg_files.keys()):
            out.append({
                "package_name": pkg_name,
                "files_count": int(len(pkg_files[pkg_name])),
                "methods_count": int(pkg_methods[pkg_name])
            })
        return out

    def save_results_json(
        self,
        results: Dict,
        output_path: str,
        test_file_path: str,
        test_method_name: str,
        max_hops: int = 2,
        analyzer_version: str = "1.0.0"
    ):
        """Save analysis results to a JSON file in the requested schema."""
        if "error" in results:
            payload = {"error": results["error"]}
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\n[ERROR] Saved error JSON to: {output_path}\n")
            return

        test_method: MethodInfo = results["test_method"]
        calls: List[MethodCall] = results.get("calls", [])
        library_methods = results.get("library_methods_skipped", set())

        analysis_payload = {
            "analysis_metadata": {
                "analyzer_version": analyzer_version,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "project_root": str(self.project_root),
                "max_hops": max_hops
            },
            "test_method": {
                **self._methodinfo_to_json(test_method),
                # ensure these match your requested layout naming
                "full_qualified_name": test_method.full_name,  # alias (same meaning)
                "calls": self._build_call_tree(results)
            },
            "statistics": self._compute_stats(results, max_hops=max_hops),
            "library_methods_skipped": self._library_methods_to_json(library_methods),
            "files_analyzed": self._files_analyzed_to_json(test_method, calls),
            "packages_involved": self._packages_involved_to_json(test_method, calls)
        }

        # Small fix: you used "full_qualified_name" in your example (typo-like),
        # but many people prefer "fully_qualified_name". Keeping your example key.
        # Also ensure test_method has those keys even if empty.
        analysis_payload["test_method"]["package_name"] = analysis_payload["test_method"].get("package_name", "")
        analysis_payload["test_method"]["parameters"] = analysis_payload["test_method"].get("parameters", [])

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_payload, f, indent=2)

        print(f"\n[SUCCESS] Saved analysis JSON to: {output_path}\n")


def analyze_test_method(project_root: str, test_file_path: str, test_method_name: str, 
                        max_hops: int = 2) -> Tuple[Dict, str]:
    
    """
    Entrypoint function: Analyze a test method and generate JSON with 2-hop call graph.
    
    Args:
        project_root: Path to Java project root directory
        test_file_path: Path to test file (absolute or relative to project_root)
        test_method_name: Name of the test method to analyze
        max_hops: Maximum call depth to trace (default: 2)
        
    Returns:
        Tuple of (json_dict, saved_file_path)
        - json_dict: The analysis results as a dictionary
        - saved_file_path: Path where the JSON file was saved
        
    Example:
        json_data, file_path = analyze_test_method(
            "/path/to/project",
            "src/test/java/MyTest.java",
            "testMyFeature"
        )
    """

    # Make test_file_path absolute if it's relative to project root
    if not os.path.isabs(test_file_path):
        test_file_path = os.path.join(project_root, test_file_path)
    
    # Initialize analyzer
    analyzer = JavaMethodAnalyzer(project_root)
    
    # Analyze the test method
    results = analyzer.analyze_test_method_internal(test_file_path, test_method_name, max_hops=max_hops)
    
    # Handle errors
    if "error" in results:
        error_json = {"error": results["error"]}
        return error_json, ""
    
    # Extract test class name for filename
    test_method_info: MethodInfo = results["test_method"]
    test_class_name = test_method_info.class_name
    
    # Create output directory
    output_dir = Path("./method-analysis-output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename: {testClassName}_{testMethodName}_invoked_methods_analysis.json
    output_filename = f"{test_class_name}_{test_method_name}_invoked_methods_analysis.json"
    output_path = output_dir / output_filename
    
    # Build JSON payload
    calls: List[MethodCall] = results.get("calls", [])
    library_methods = results.get("library_methods_skipped", set())
    
    analysis_payload = {
        "analysis_metadata": {
            "analyzer_version": "1.0.0",
            "analysis_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "project_root": str(Path(project_root).resolve()),
            "max_hops": max_hops
        },
        "test_method": {
            **analyzer._methodinfo_to_json(test_method_info),
            "full_qualified_name": test_method_info.full_name,
            "calls": analyzer._build_call_tree(results)
        },
        "statistics": analyzer._compute_stats(results, max_hops=max_hops),
        "library_methods_skipped": analyzer._library_methods_to_json(library_methods),
        "files_analyzed": analyzer._files_analyzed_to_json(test_method_info, calls),
        "packages_involved": analyzer._packages_involved_to_json(test_method_info, calls)
    }
    
    # Ensure required fields
    analysis_payload["test_method"]["package_name"] = analysis_payload["test_method"].get("package_name", "")
    analysis_payload["test_method"]["parameters"] = analysis_payload["test_method"].get("parameters", [])
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)
    
    print(f"\n[SUCCESS] Saved analysis JSON to: {output_path}\n")
    
    return analysis_payload, str(output_path)


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 4:
        print("Usage: python java_method_analyzer.py <project_root> <test_file_path> <test_method_name>")
        print("\nExample:")
        print("  python java_method_analyzer.py /path/to/project src/test/MyTest.java testMyFeature")
        print("\nDescription:")
        print("  Analyzes a Java test method and traces all method calls up to 2 hops deep.")
        print("  Ignores library methods and shows only project methods with their bodies.")
        sys.exit(1)
    
    project_root = sys.argv[1]
    test_file_path = sys.argv[2]
    test_method_name = sys.argv[3]
    
    # Use the entrypoint function
    json_data, saved_path = analyze_test_method(project_root, test_file_path, test_method_name)
    
    if "error" in json_data:
        print(f"\n[ERROR] Error: {json_data['error']}\n")
        sys.exit(1)



if __name__ == "__main__":
    main()