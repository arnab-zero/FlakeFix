# Workflow Diagram

## Overall Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: routingTest_call_analysis.json         │
│  {                                                               │
│    test_method: { body, calls: [hop1_methods] }                 │
│    hop1_methods: [{ body, calls: [hop2_methods] }]              │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Inline Hop-2 → Hop-1                │
│                                                                  │
│  For each hop1_method:                                          │
│    ├─ Skip if no calls                                          │
│    └─ For each hop2_method in calls:                            │
│        ├─ Skip if recursive (hop2 == hop1)                      │
│        └─ Inline hop2 into hop1:                                │
│            ├─ Send inline prompt                                │
│            ├─ Extract code                                      │
│            ├─ Check consistency                                 │
│            ├─ If problems: rewrite                              │
│            └─ If max retries: keep original                     │
│                                                                  │
│  Result: Updated hop1_methods with hop2 inlined                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: Inline Hop-1 → Test                 │
│                                                                  │
│  For test_method:                                               │
│    For each hop1_method in calls:                               │
│      ├─ Skip if recursive (hop1 == test)                        │
│      └─ Inline hop1 into test:                                  │
│          ├─ Use updated hop1 body from Phase 1                  │
│          ├─ Send inline prompt                                  │
│          ├─ Extract code                                        │
│          ├─ Check consistency                                   │
│          ├─ If problems: rewrite                                │
│          └─ If max retries: keep original                       │
│                                                                  │
│  Result: Updated test_method with hop1 inlined                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: Remove Comments                      │
│                                                                  │
│  ├─ Send comment removal prompt                                 │
│  ├─ Extract cleaned code                                        │
│  └─ Return final result                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: inlining_result.json                  │
│  {                                                               │
│    status: "success",                                           │
│    original_test_method_body: "...",                            │
│    updated_test_method_body: "...",                             │
│    error: null                                                  │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Single Inlining Operation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    START: Inline Method A into Method B          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Attempt = 0    │
                    └─────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Is this a recursive call?              │
        │  (Method A == Method B)                 │
        └─────────────────────────────────────────┘
                │                    │
               YES                  NO
                │                    │
                ▼                    ▼
        ┌─────────────┐    ┌─────────────────────┐
        │  SKIP       │    │  Attempt <= Max?    │
        │  (Log it)   │    └─────────────────────┘
        └─────────────┘            │          │
                                  YES        NO
                                   │          │
                                   ▼          ▼
                        ┌──────────────┐  ┌──────────────┐
                        │ First time?  │  │ FAIL: Keep   │
                        └──────────────┘  │ original     │
                            │      │      └──────────────┘
                           YES    NO
                            │      │
                            ▼      ▼
                    ┌─────────┐  ┌─────────┐
                    │ Inline  │  │ Rewrite │
                    │ Prompt  │  │ Prompt  │
                    └─────────┘  └─────────┘
                            │      │
                            └──┬───┘
                               ▼
                    ┌──────────────────┐
                    │ Call Ollama API  │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ Extract Code     │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ Null or Empty?   │
                    └──────────────────┘
                        │          │
                       YES        NO
                        │          │
                        │          ▼
                        │   ┌──────────────────┐
                        │   │ Consistency      │
                        │   │ Check            │
                        │   └──────────────────┘
                        │          │
                        │          ▼
                        │   ┌──────────────────┐
                        │   │ Consistent?      │
                        │   │ Signature OK?    │
                        │   └──────────────────┘
                        │      │          │
                        │     YES        NO
                        │      │          │
                        │      ▼          │
                        │   ┌──────┐     │
                        │   │SUCCESS│    │
                        │   └──────┘     │
                        │                │
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Attempt++       │
                        │ Loop back       │
                        └─────────────────┘
```

## Consistency Check Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                    Consistency Check Prompt                      │
│                                                                  │
│  Input:                                                          │
│    - Original method body (before inlining)                     │
│    - All invoked method bodies (that were inlined)              │
│    - Generated inlined body (after inlining)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Send to Ollama Model                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Returns JSON                            │
│  {                                                               │
│    "is_consistent": true/false,                                 │
│    "signature_preserved": true/false,                           │
│    "problems": [                                                │
│      "Method signature changed",                                │
│      "Variable conflict: x",                                    │
│      "Return statement not handled correctly"                   │
│    ]                                                            │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Parse JSON      │
                    └─────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  is_consistent AND signature_preserved? │
        └─────────────────────────────────────────┘
                │                    │
               YES                  NO
                │                    │
                ▼                    ▼
        ┌─────────────┐    ┌─────────────────────┐
        │  SUCCESS    │    │  Log problems       │
        │  Use code   │    │  Prepare rewrite    │
        └─────────────┘    └─────────────────────┘
```

## Rewrite Prompt Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Consistency Check Failed                      │
│  Problems:                                                       │
│    - Method signature changed                                   │
│    - Variable conflict                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Build Rewrite Prompt                          │
│                                                                  │
│  Include:                                                        │
│    1. Original method body (before inlining)                    │
│    2. All invoked method bodies (to be inlined)                 │
│    3. Previous attempt (that failed)                            │
│    4. List of problems found                                    │
│    5. Instructions to fix the problems                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Send to Ollama Model                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Returns Corrected Code                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Extract Code                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Run Consistency Check Again                   │
└─────────────────────────────────────────────────────────────────┘
```

## Log Structure

```
====================================================================================================
MAJOR SECTION (Level 1)
====================================================================================================
  Content at level 1
    Content at level 2
      Content at level 3

--------------------------------------------------------------------------------
Subsection (Level 2)
--------------------------------------------------------------------------------
  Content at level 1
    Content at level 2

>>> Sub-subsection (Level 3)
  Content at level 1
    Content at level 2
```

## Example Log Flow

````
====================================================================================================
Enhanced Method Inliner Log - 2026-02-12 15:30:00
====================================================================================================
Model: qwen3-coder:480b-cloud
Max Retries: 2
====================================================================================================

Inliner initialized successfully

====================================================================================================
STARTING INLINING PROCESS
====================================================================================================
  Test method: org.platformlambda.core.MulticastTest.routingTest
  Test method calls 9 hop-1 methods
  Total hop-1 methods: 9
  Total hop-2 methods: 15

====================================================================================================
PHASE 1: Inlining Hop-2 into Hop-1 Method: org.platformlambda.core.system.AsyncObjectStreamReader.get
====================================================================================================
  Found 10 hop-2 methods to inline:
    - org.platformlambda.core.system.EventEmitter.getInstance
    - org.platformlambda.core.system.Platform.getInstance
    ...

--------------------------------------------------------------------------------
Inlining: org.platformlambda.core.system.EventEmitter.getInstance
--------------------------------------------------------------------------------
  Original caller body:
    public Future<Object> get() { ... }

  Callee body to inline:
    public static EventEmitter getInstance() { return INSTANCE; }

  Attempt 1/3
    Calling Ollama for: initial inlining
      Prompt length: 1234 chars
      Attempt 1/3
      ✓ Response received: 1282 chars
    Extracting code from response
      Removed ```java``` markers
      Extracted code: 1266 chars
    Inlined code:
      public Future<Object> get() { final EventEmitter po = INSTANCE; ... }

    >>> Consistency Check
      Calling Ollama for: consistency check
        Prompt length: 2345 chars
        Attempt 1/3
        ✓ Response received: 156 chars
      Consistency: true
      Signature preserved: true
      No problems found
    ✓ Consistency check passed
  ✓ Successfully inlined org.platformlambda.core.system.EventEmitter.getInstance

...
````

## Decision Tree

```
                    ┌─────────────────┐
                    │ Start Inlining  │
                    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │ Recursive?      │
                    └─────────────────┘
                        │       │
                       YES     NO
                        │       │
                        ▼       ▼
                    ┌────┐  ┌────────┐
                    │Skip│  │Continue│
                    └────┘  └────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │ Send Prompt  │
                        └──────────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │ Get Response │
                        └──────────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │ Null/Empty?  │
                        └──────────────┘
                            │       │
                           YES     NO
                            │       │
                            │       ▼
                            │   ┌──────────────┐
                            │   │ Consistency  │
                            │   │ Check        │
                            │   └──────────────┘
                            │       │
                            │       ▼
                            │   ┌──────────────┐
                            │   │ Pass?        │
                            │   └──────────────┘
                            │   │       │
                            │  YES     NO
                            │   │       │
                            │   ▼       │
                            │ ┌────┐   │
                            │ │Done│   │
                            │ └────┘   │
                            │           │
                            └─────┬─────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │ Retries Left?│
                          └──────────────┘
                              │       │
                             YES     NO
                              │       │
                              ▼       ▼
                          ┌────────┐ ┌────────┐
                          │Rewrite │ │Keep    │
                          │& Retry │ │Original│
                          └────────┘ └────────┘
```
