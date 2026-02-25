"""
Enhanced LLM-driven Method Inlining System using Ollama
Includes consistency checking and detailed logging
"""

import json
import os
import requests
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from prompts import (
    get_inline_prompt,
    get_consistency_check_prompt,
    get_rewrite_prompt,
    get_comment_removal_prompt
)


@dataclass
class MethodInfo:
    """Represents a Java method"""
    fqn: str
    name: str
    class_name: str
    package_name: str
    body_lines: List[str]
    body_text: str
    calls: List[str]


@dataclass
class InlineResult:
    """Result of an inline operation"""
    success: bool
    inlined_body: Optional[str] = None
    error: Optional[str] = None
    kept_original: bool = False


class MethodInliner:
    """Enhanced LLM-based method inliner with consistency checking"""

    def __init__(
        self,
        model_name: str = "qwen3-coder:480b-cloud",
        log_file: str = "./inliner_log.txt",
        max_retries: int = 2
    ):
        """Initialize with Ollama model and logging"""
        self.model_name = model_name
        self.log_file = log_file
        self.max_retries = max_retries
        
        # Initialize log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"Enhanced Method Inliner Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Max Retries: {max_retries}\n")
            f.write("=" * 100 + "\n\n")
        
        print("=" * 100)
        print("Enhanced Method Inliner Started")
        print("=" * 100)
        print(f"Logs: {log_file}")
        print(f"Model: {model_name}")
        print("=" * 100)
        print()
        
        self._log("Inliner initialized successfully")
        self._log("")
    
    def _log(self, message: str, indent: int = 0):
        """Write message to log file with optional indentation"""
        indent_str = "  " * indent
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(indent_str + message + "\n")
    
    def _log_section(self, title: str, level: int = 1):
        """Log a section header"""
        if level == 1:
            self._log("")
            self._log("=" * 100)
            self._log(title)
            self._log("=" * 100)
        elif level == 2:
            self._log("")
            self._log("-" * 80)
            self._log(title)
            self._log("-" * 80)
        else:
            self._log("")
            self._log(f">>> {title}")
    
    def _call_ollama(self, prompt: str, purpose: str) -> Optional[str]:
        """Call Ollama API with detailed logging"""
        self._log(f"Calling Ollama for: {purpose}", indent=1)
        self._log(f"Prompt length: {len(prompt)} chars", indent=2)
        
        max_attempts = 3
        timeout = 120
        
        for attempt in range(max_attempts):
            try:
                self._log(f"Attempt {attempt + 1}/{max_attempts}", indent=2)
                
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': self.model_name,
                        'prompt': prompt,
                        'stream': False,
                        'temperature': 0.2,
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('response', '').strip()
                    if content:
                        self._log(f"✓ Response received: {len(content)} chars", indent=2)
                        return content
                    else:
                        self._log(f"✗ Empty response", indent=2)
                        if attempt < max_attempts - 1:
                            time.sleep(2)
                            continue
                        return None
                else:
                    self._log(f"✗ HTTP {response.status_code}", indent=2)
                    if attempt < max_attempts - 1:
                        time.sleep(2)
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                self._log(f"✗ Timeout after {timeout}s", indent=2)
                if attempt < max_attempts - 1:
                    time.sleep(5)
                    continue
                return None
                
            except requests.exceptions.ConnectionError:
                self._log(f"✗ Connection error - Ollama not running?", indent=2)
                if attempt < max_attempts - 1:
                    time.sleep(5)
                    continue
                return None
                
            except Exception as e:
                self._log(f"✗ Error: {str(e)}", indent=2)
                if attempt < max_attempts - 1:
                    time.sleep(2)
                    continue
                return None
        
        return None
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Java code from LLM response"""
        self._log("Extracting code from response", indent=2)
        response = response.strip()
        
        # Remove markdown code blocks
        if "```java" in response:
            response = response.split("```java")[1].split("```")[0].strip()
            self._log("Removed ```java``` markers", indent=3)
        elif "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                response = parts[1].strip()
                self._log("Removed ``` markers", indent=3)
            elif response.endswith("```"):
                response = response[:-3].strip()
                self._log("Removed trailing ```", indent=3)
        
        # Remove any remaining markdown
        if response.startswith("```"):
            response = response[3:].strip()
        if response.endswith("```"):
            response = response[:-3].strip()
        
        self._log(f"Extracted code: {len(response)} chars", indent=3)
        return response
    
    def _check_consistency(
        self,
        original_body: str,
        invoked_methods: List[Dict[str, str]],
        inlined_body: str
    ) -> tuple[bool, bool, List[str]]:
        """Check if inlined code is consistent with original"""
        self._log_section("Consistency Check", level=3)
        
        prompt = get_consistency_check_prompt(
            original_body,
            invoked_methods,
            inlined_body
        )
        
        response = self._call_ollama(prompt, "consistency check")
        if not response:
            self._log("✗ No response from model", indent=2)
            return False, False, ["No response from consistency check"]
        
        try:
            # Extract JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                parts = response.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") and part.endswith("}"):
                        json_str = part
                        break
                else:
                    json_str = response
            else:
                json_str = response
            
            result = json.loads(json_str)
            is_consistent = result.get("is_consistent", False)
            signature_preserved = result.get("signature_preserved", False)
            problems = result.get("problems", [])
            
            self._log(f"Consistency: {is_consistent}", indent=2)
            self._log(f"Signature preserved: {signature_preserved}", indent=2)
            if problems:
                self._log(f"Problems found: {len(problems)}", indent=2)
                for i, problem in enumerate(problems, 1):
                    self._log(f"{i}. {problem}", indent=3)
            else:
                self._log("No problems found", indent=2)
            
            return is_consistent, signature_preserved, problems
            
        except json.JSONDecodeError as e:
            self._log(f"✗ JSON parse error: {str(e)}", indent=2)
            return False, False, ["Could not parse consistency check response"]
        except Exception as e:
            self._log(f"✗ Error: {str(e)}", indent=2)
            return False, False, [f"Consistency check error: {str(e)}"]
    
    def _inline_single_method(
        self,
        caller_body: str,
        callee_fqn: str,
        callee_body: str,
        all_invoked_methods: List[Dict[str, str]]
    ) -> Optional[str]:
        """Inline a single method with consistency checking and retries"""
        self._log_section(f"Inlining: {callee_fqn}", level=2)
        
        self._log("Original caller body:", indent=1)
        self._log(caller_body, indent=2)
        self._log("", indent=0)
        
        self._log("Callee body to inline:", indent=1)
        self._log(callee_body, indent=2)
        self._log("", indent=0)
        
        for attempt in range(self.max_retries + 1):
            self._log(f"Attempt {attempt + 1}/{self.max_retries + 1}", indent=1)
            
            # Step 1: Get inlined code
            if attempt == 0:
                prompt = get_inline_prompt(caller_body, callee_fqn, callee_body)
                inlined = self._call_ollama(prompt, "initial inlining")
            else:
                # Use rewrite prompt with problems from previous attempt
                prompt = get_rewrite_prompt(
                    caller_body,
                    all_invoked_methods,
                    previous_attempt,
                    problems
                )
                inlined = self._call_ollama(prompt, f"rewrite attempt {attempt}")
            
            if not inlined:
                self._log("✗ No response from model", indent=2)
                if attempt < self.max_retries:
                    continue
                return None
            
            # Step 2: Extract code
            inlined = self._extract_code_from_response(inlined)
            
            if not inlined or len(inlined.strip()) == 0:
                self._log("✗ Empty code after extraction", indent=2)
                if attempt < self.max_retries:
                    previous_attempt = inlined if inlined else ""
                    problems = ["Empty or null code returned"]
                    continue
                return None
            
            self._log("Inlined code:", indent=2)
            self._log(inlined, indent=3)
            self._log("", indent=0)
            
            # Step 3: Consistency check
            is_consistent, signature_preserved, problems = self._check_consistency(
                caller_body,
                all_invoked_methods,
                inlined
            )
            
            if is_consistent and signature_preserved:
                self._log("✓ Consistency check passed", indent=2)
                return inlined
            else:
                self._log("✗ Consistency check failed", indent=2)
                if not signature_preserved:
                    self._log("  - Method signature was changed!", indent=2)
                if attempt < self.max_retries:
                    self._log(f"  - Will retry with rewrite prompt", indent=2)
                    previous_attempt = inlined
                    continue
                else:
                    self._log(f"  - Max retries reached, keeping original", indent=2)
                    return None
        
        return None
    
    def inline_hop2_into_hop1(
        self,
        hop1_method: MethodInfo,
        hop2_methods: Dict[str, MethodInfo]
    ) -> InlineResult:
        """Inline all hop-2 methods into a hop-1 method"""
        self._log_section(f"PHASE 1: Inlining Hop-2 into Hop-1 Method: {hop1_method.fqn}", level=1)
        
        current_body = hop1_method.body_text
        current_calls = hop1_method.calls
        
        # Filter to only hop2 methods that are called (and not recursive)
        relevant_hop2 = {
            fqn: method for fqn, method in hop2_methods.items()
            if fqn in current_calls and fqn != hop1_method.fqn  # Ignore recursive calls
        }
        
        if not relevant_hop2:
            self._log("No hop-2 methods to inline (or all are recursive)", indent=1)
            return InlineResult(success=True, inlined_body=current_body)
        
        self._log(f"Found {len(relevant_hop2)} hop-2 methods to inline:", indent=1)
        for fqn in relevant_hop2.keys():
            self._log(f"- {fqn}", indent=2)
        self._log("", indent=0)
        
        # Prepare invoked methods list for consistency checking
        all_invoked_methods = [
            {"fqn": fqn, "body": method.body_text}
            for fqn, method in relevant_hop2.items()
        ]
        
        # Inline each hop-2 method one by one
        for fqn, hop2_method in relevant_hop2.items():
            result = self._inline_single_method(
                current_body,
                fqn,
                hop2_method.body_text,
                all_invoked_methods
            )
            
            if result:
                self._log(f"✓ Successfully inlined {fqn}", indent=1)
                current_body = result
            else:
                self._log(f"✗ Failed to inline {fqn}, keeping original body", indent=1)
        
        self._log("", indent=0)
        self._log("Final hop-1 method body after all inlining:", indent=1)
        self._log(current_body, indent=2)
        
        return InlineResult(success=True, inlined_body=current_body)
    
    def inline_hop1_into_test(
        self,
        test_method: MethodInfo,
        hop1_methods: Dict[str, MethodInfo]
    ) -> InlineResult:
        """Inline all hop-1 methods into test method"""
        self._log_section(f"PHASE 2: Inlining Hop-1 into Test Method: {test_method.fqn}", level=1)
        
        current_body = test_method.body_text
        current_calls = test_method.calls
        
        # Filter to only hop1 methods that are called (and not recursive)
        relevant_hop1 = {
            fqn: method for fqn, method in hop1_methods.items()
            if fqn in current_calls and fqn != test_method.fqn  # Ignore recursive calls
        }
        
        if not relevant_hop1:
            self._log("No hop-1 methods to inline (or all are recursive)", indent=1)
            return InlineResult(success=True, inlined_body=current_body)
        
        self._log(f"Found {len(relevant_hop1)} hop-1 methods to inline:", indent=1)
        for fqn in relevant_hop1.keys():
            self._log(f"- {fqn}", indent=2)
        self._log("", indent=0)
        
        # Prepare invoked methods list for consistency checking
        all_invoked_methods = [
            {"fqn": fqn, "body": method.body_text}
            for fqn, method in relevant_hop1.items()
        ]
        
        # Inline each hop-1 method one by one
        for fqn, hop1_method in relevant_hop1.items():
            result = self._inline_single_method(
                current_body,
                fqn,
                hop1_method.body_text,
                all_invoked_methods
            )
            
            if result:
                self._log(f"✓ Successfully inlined {fqn}", indent=1)
                current_body = result
            else:
                self._log(f"✗ Failed to inline {fqn}, keeping original body", indent=1)
        
        self._log("", indent=0)
        self._log("Final test method body after all inlining:", indent=1)
        self._log(current_body, indent=2)
        
        return InlineResult(success=True, inlined_body=current_body)
    
    def remove_comments(self, method_body: str) -> str:
        """Remove comments from method"""
        self._log_section("PHASE 3: Removing Comments", level=1)
        
        prompt = get_comment_removal_prompt(method_body)
        cleaned = self._call_ollama(prompt, "comment removal")
        
        if cleaned:
            cleaned = self._extract_code_from_response(cleaned)
            self._log(f"✓ Comments removed", indent=1)
            self._log(f"Original size: {len(method_body)} chars", indent=2)
            self._log(f"Cleaned size: {len(cleaned)} chars", indent=2)
            return cleaned
        else:
            self._log(f"✗ Could not remove comments, keeping original", indent=1)
            return method_body
    
    def process(self, analysis_json: Dict[str, Any]) -> Dict[str, Any]:
        """Main orchestration function"""
        self._log_section("STARTING INLINING PROCESS", level=1)
        
        # Extract test method
        test_data = analysis_json["test_method"]
        test_method = MethodInfo(
            fqn=test_data["full_qualified_name"],
            name=test_data["name"],
            class_name=test_data["class_name"],
            package_name=test_data["package_name"],
            body_lines=test_data["body"]["lines"],
            body_text=test_data["body"]["full_text"],
            calls=[call["full_qualified_name"] for call in test_data["calls"]]
        )
        
        self._log(f"Test method: {test_method.fqn}", indent=1)
        self._log(f"Test method calls {len(test_method.calls)} hop-1 methods", indent=1)
        
        # Extract hop-1 methods
        hop1_methods: Dict[str, MethodInfo] = {}
        for call in test_data["calls"]:
            hop1_methods[call["full_qualified_name"]] = MethodInfo(
                fqn=call["full_qualified_name"],
                name=call["name"],
                class_name=call["class_name"],
                package_name=call["package_name"],
                body_lines=call["body"]["lines"],
                body_text=call["body"]["full_text"],
                calls=[c["full_qualified_name"] for c in call.get("calls", [])]
            )
        
        # Extract hop-2 methods
        hop2_methods: Dict[str, MethodInfo] = {}
        for hop1_fqn, hop1_method in hop1_methods.items():
            hop1_call_data = next(
                (call for call in test_data["calls"]
                 if call["full_qualified_name"] == hop1_fqn),
                None
            )
            if hop1_call_data:
                for hop2_call in hop1_call_data.get("calls", []):
                    hop2_methods[hop2_call["full_qualified_name"]] = MethodInfo(
                        fqn=hop2_call["full_qualified_name"],
                        name=hop2_call["name"],
                        class_name=hop2_call["class_name"],
                        package_name=hop2_call["package_name"],
                        body_lines=hop2_call["body"]["lines"],
                        body_text=hop2_call["body"]["full_text"],
                        calls=[c["full_qualified_name"] for c in hop2_call.get("calls", [])]
                    )
        
        self._log(f"Total hop-1 methods: {len(hop1_methods)}", indent=1)
        self._log(f"Total hop-2 methods: {len(hop2_methods)}", indent=1)
        self._log("", indent=0)
        
        # Phase 1: Inline hop-2 into hop-1
        updated_hop1_methods: Dict[str, MethodInfo] = {}
        
        for hop1_fqn, hop1_method in hop1_methods.items():
            result = self.inline_hop2_into_hop1(hop1_method, hop2_methods)
            
            if result.success:
                updated_hop1 = MethodInfo(
                    fqn=hop1_method.fqn,
                    name=hop1_method.name,
                    class_name=hop1_method.class_name,
                    package_name=hop1_method.package_name,
                    body_lines=result.inlined_body.split("\n"),
                    body_text=result.inlined_body,
                    calls=hop1_method.calls
                )
                updated_hop1_methods[hop1_fqn] = updated_hop1
        
        # Phase 2: Inline hop-1 into test
        result = self.inline_hop1_into_test(test_method, updated_hop1_methods)
        
        if not result.inlined_body:
            self._log_section("INLINING FAILED", level=1)
            return {
                "success": False,
                "error": "Failed to inline hop-1 into test",
                "original_test_body": test_method.body_text,
                "updated_test_body": None
            }
        
        # Phase 3: Remove comments
        cleaned_body = self.remove_comments(result.inlined_body)
        
        self._log_section("INLINING COMPLETE", level=1)
        self._log(f"✓ Success", indent=1)
        self._log(f"Original size: {len(test_method.body_text)} chars", indent=1)
        self._log(f"Final size: {len(cleaned_body)} chars", indent=1)
        
        return {
            "success": True,
            "original_test_body": test_method.body_text,
            "updated_test_body": cleaned_body,
            "error": None
        }
