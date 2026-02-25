FLAKY_TEST_CLASSIFICATION_PROMPT = """
You are a software testing analyst. Your task is to classify a Java test method body into the most likely flaky-test type based ONLY on the definitions below.

DEFINITIONS (use these as the only decision criteria):

1) Async-await flaky tests (ASYNC_AWAIT):
Tests become non-deterministic because they invoke asynchronous operations (e.g., network calls, background tasks, promises, futures, callbacks) but fail to properly wait for completion or verify resource readiness before continuing, so results vary due to timing instead of correctness.

2) Test Order Dependent flaky tests (TEST_ORDER_DEPENDENCY):
Tests become non-deterministic because their outcome depends on the execution order of other tests due to shared state, common resources, missing cleanup, or missing preconditions; results vary because of leftover effects from earlier tests.

3) Time Dependent flaky tests (TIME_DEPENDENT):
Tests become non-deterministic because they rely on system local time/date/timezone/timestamp precision, so they pass/fail due to temporal changes rather than correctness.

4) Concurrency flaky tests (CONCURRENCY):
Tests become non-deterministic due to unsafe interactions between multiple threads (race conditions, deadlocks, improper synchronization), where behavior depends on implicit thread ordering/timing, causing unpredictable pass/fail due to synchronization issues in test or production code.

5) Unordered Collection flaky tests (UNORDERED_COLLECTION):
Tests become non-deterministic because they assume a fixed iteration/return order from unordered data structures (e.g., sets, maps), causing intermittent failures when traversal order changes.

INSTRUCTIONS:
- Decide the single BEST label among:
  {{ASYNC_AWAIT, TEST_ORDER_DEPENDENCY, TIME_DEPENDENT, CONCURRENCY, UNORDERED_COLLECTION, NOT_FLAKY_OR_UNKNOWN}}
- Base your decision on concrete evidence from the code.
- If multiple types appear, pick the dominant root cause.
- If insufficient evidence, return NOT_FLAKY_OR_UNKNOWN.

JAVA TEST METHOD BODY:
<<<
{java_code}
>>>

CRITICAL: You MUST respond with ONLY valid JSON in this exact format. Do not include any text before or after the JSON object.

{{
  "label": "<MUST be exactly one of: ASYNC_AWAIT, CONCURRENCY, UNORDERED_COLLECTION, TIME_DEPENDENT, TEST_ORDER_DEPENDENCY, NOT_FLAKY_OR_UNKNOWN>",
  "confidence": "<MUST be exactly one of: HIGH, MEDIUM, LOW>",
  "justification": "<detailed explanation of why you chose this label, 2-3 sentences>",
  "code_patterns": ["<list of 2-5 specific code patterns that indicate this flakiness type>"],
  "recommendation": "<brief recommendation for fixing this issue, 1-2 sentences>"
}}

Example valid response:
{{
  "label": "ASYNC_AWAIT",
  "confidence": "HIGH",
  "justification": "The test uses waitForProvider which returns a Future and relies on asynchronous callbacks. The test polls a BlockingQueue with a timeout, which is a classic async-await pattern that can fail if the async operation takes longer than expected.",
  "code_patterns": ["Future.onSuccess callback", "BlockingQueue.poll with timeout", "Asynchronous event handling"],
  "recommendation": "Add explicit synchronization barriers after async calls to ensure completion before assertions."
}}
"""


def build_flaky_prompt(java_code: str) -> str:
    return FLAKY_TEST_CLASSIFICATION_PROMPT.format(java_code=java_code)
