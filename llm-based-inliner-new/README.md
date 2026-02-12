# Enhanced LLM-Based Method Inliner

Enhanced version with consistency checking and detailed logging.

## Features

- **Consistency Checking**: After each inlining, verifies semantic correctness
- **Automatic Rewrite**: If inconsistencies found, automatically rewrites with context
- **Method Signature Preservation**: Strictly enforces that method signatures don't change
- **Recursive Call Handling**: Automatically ignores recursive calls
- **Detailed Logging**: Logs every step, prompt, response, and decision
- **Null/Empty Handling**: Retries on null or empty responses, keeps original if max retries exceeded

## Workflow

### Phase 1: Inline Hop-2 into Hop-1

For each Hop-1 method that has Hop-2 calls:

1. Send inline prompt to model
2. Extract code from response
3. Check for null/empty (retry if needed)
4. Run consistency check
5. If inconsistent, send rewrite prompt with problems
6. Repeat up to max_retries
7. If still fails, keep original method body

### Phase 2: Inline Hop-1 into Test

Same process as Phase 1, but inlining updated Hop-1 methods into test method.

### Phase 3: Remove Comments

Clean up the final inlined test method.

## Usage

```bash
cd llm-based-inliner-new
python run_inliner.py
```

## Input

Expects `../routingTest_call_analysis.json` with structure:

```json
{
  "test_method": {
    "full_qualified_name": "...",
    "body": {"full_text": "..."},
    "calls": [
      {
        "full_qualified_name": "...",
        "body": {"full_text": "..."},
        "calls": [...]
      }
    ]
  }
}
```

## Output

- `./output/inlining_result.json` - Final result with original and updated test method
- `./inliner_log.txt` - Detailed log with all prompts, responses, and decisions

## Log Format

The log includes:

- Section headers with clear visual separation
- Indentation for nested operations
- All prompts sent to the model
- All responses received
- Consistency check results
- Problems found and retry attempts
- Final results

## Key Improvements Over Original

1. **Consistency Checking**: Validates semantic correctness after each inlining
2. **Rewrite on Failure**: Provides context and asks model to fix problems
3. **Signature Preservation**: Explicitly checks that method signatures don't change
4. **Better Null Handling**: Retries on null/empty, keeps original if max retries exceeded
5. **Recursive Call Filtering**: Automatically ignores recursive calls
6. **Detailed Logging**: Every step is logged with clear formatting

## Configuration

Environment variables:

- `INLINER_MODEL`: Model name (default: qwen3-coder:480b-cloud)
- `LOG_FILE`: Log file path (default: ./inliner_log.txt)
- `OUTPUT_PATH`: Output file path (default: ./output/inlining_result.json)

## Requirements

- Python 3.8+
- Ollama running on localhost:11434
- qwen3-coder:480b-cloud model installed in Ollama
