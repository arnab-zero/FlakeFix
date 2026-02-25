"""
Enhanced Prompt Templates for LLM-based Method Inlining
Includes consistency checking and rewrite prompts
"""

INLINE_METHOD_PROMPT = """You are a Java method inlining expert. Your task is to inline a called method into the caller method.

CRITICAL RULES:
1. DO NOT MODIFY THE METHOD SIGNATURE - Keep it exactly as it is
2. Replace ONLY the method call with the method body
3. DO NOT add or remove any other code
4. Handle variable name collisions by prefixing callee variables with "__inlined_<method_name>_"
5. Replace 'return' statements appropriately:
   - If the call is used as an expression (e.g., `x = foo()`), replace return with the value
   - If the call is a statement (e.g., `foo();`), just inline the body
6. Keep 'this' references from the callee as-is (they refer to the same object)
7. DO NOT try to inline library calls (java.*, javax.*, org.junit.*)
8. If the method is a simple getter/setter, just replace the call with the field access
9. Return ONLY the complete method body with the SAME signature

CALLEE METHOD (to be inlined):
Full Qualified Name: {callee_fqn}
```java
{callee_body}
```

CALLER METHOD (where to inline):
```java
{caller_body}
```

TASK:
Find the call to `{callee_method_name}` in the caller and replace it with the callee's method body.
PRESERVE THE METHOD SIGNATURE EXACTLY AS IT IS.

EXAMPLE:
If caller has: 
```java
public static EventEmitter getInstance() {{
    EventEmitter po = EventEmitter.getInstance();
    return po;
}}
```
And callee is: 
```java
public static EventEmitter getInstance() {{ 
    return INSTANCE; 
}}
```
Then replace with: 
```java
public static EventEmitter getInstance() {{
    EventEmitter po = INSTANCE;
    return po;
}}
```

OUTPUT:
Return ONLY the complete inlined caller method body as valid Java code. 
DO NOT include markdown formatting, explanations, or anything else.
DO NOT change the method signature.
"""

CONSISTENCY_CHECK_PROMPT = """You are a Java code consistency checker. Your task is to verify if an inlined method is semantically correct.

ORIGINAL METHOD BODY (before inlining):
```java
{original_body}
```

INVOKED METHOD BODIES (that were inlined):
{invoked_methods_info}

GENERATED INLINED METHOD BODY (after inlining):
```java
{inlined_body}
```

TASK:
Check if the inlined method body is semantically correct by verifying:
1. Method signature is EXACTLY the same as original (not changed)
2. All method calls to the invoked methods are properly replaced with their bodies
3. Variable names don't conflict
4. Return statements are handled correctly
5. The logic flow is preserved
6. No syntax errors introduced
7. Semantic meaning is preserved

Return JSON with format:
{{
    "is_consistent": true/false,
    "signature_preserved": true/false,
    "problems": ["problem1", "problem2", ...] or []
}}

If everything looks correct, return:
{{"is_consistent": true, "signature_preserved": true, "problems": []}}

If there are issues, list them in "problems" array.

OUTPUT (JSON only, no markdown):
"""

REWRITE_PROMPT = """You are a Java method inlining expert. The previous inlining attempt had problems. Please rewrite it correctly.

ORIGINAL METHOD BODY (before inlining):
```java
{original_body}
```

INVOKED METHOD BODIES (to be inlined):
{invoked_methods_info}

PREVIOUS ATTEMPT (had problems):
```java
{previous_attempt}
```

PROBLEMS FOUND:
{problems}

TASK:
Rewrite the inlined method body correctly, fixing all the problems listed above.

CRITICAL RULES:
1. DO NOT MODIFY THE METHOD SIGNATURE - Keep it exactly as in the original
2. Replace method calls with their bodies correctly
3. Handle variable name collisions properly
4. Preserve semantic correctness
5. Fix all the problems listed above

OUTPUT:
Return ONLY the complete corrected method body as valid Java code.
DO NOT include markdown formatting, explanations, or anything else.
DO NOT change the method signature.
"""

REMOVE_COMMENTS_PROMPT = """Remove all comments (// and /* */) from this Java method while preserving functionality.

METHOD:
```java
{method_body}
```

OUTPUT:
Return ONLY the method body without any comments. Do not include markdown formatting.
"""


def get_inline_prompt(caller_body: str, callee_fqn: str, callee_body: str) -> str:
    """Get the inline method prompt"""
    callee_method_name = callee_fqn.split('.')[-1]
    return INLINE_METHOD_PROMPT.format(
        callee_fqn=callee_fqn,
        callee_body=callee_body,
        caller_body=caller_body,
        callee_method_name=callee_method_name
    )


def get_consistency_check_prompt(
    original_body: str,
    invoked_methods: list,
    inlined_body: str
) -> str:
    """Get the consistency check prompt"""
    # Format invoked methods info
    invoked_info_parts = []
    for i, method in enumerate(invoked_methods, 1):
        invoked_info_parts.append(f"{i}. {method['fqn']}:\n```java\n{method['body']}\n```")
    
    invoked_methods_info = "\n\n".join(invoked_info_parts)
    
    return CONSISTENCY_CHECK_PROMPT.format(
        original_body=original_body,
        invoked_methods_info=invoked_methods_info,
        inlined_body=inlined_body
    )


def get_rewrite_prompt(
    original_body: str,
    invoked_methods: list,
    previous_attempt: str,
    problems: list
) -> str:
    """Get the rewrite prompt"""
    # Format invoked methods info
    invoked_info_parts = []
    for i, method in enumerate(invoked_methods, 1):
        invoked_info_parts.append(f"{i}. {method['fqn']}:\n```java\n{method['body']}\n```")
    
    invoked_methods_info = "\n\n".join(invoked_info_parts)
    
    # Format problems
    problems_text = "\n".join(f"- {p}" for p in problems)
    
    return REWRITE_PROMPT.format(
        original_body=original_body,
        invoked_methods_info=invoked_methods_info,
        previous_attempt=previous_attempt,
        problems=problems_text
    )


def get_comment_removal_prompt(method_body: str) -> str:
    """Get the comment removal prompt"""
    return REMOVE_COMMENTS_PROMPT.format(method_body=method_body)
