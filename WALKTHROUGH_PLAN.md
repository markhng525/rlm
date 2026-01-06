# RLM Paper-to-Code Walkthrough Plan

## Target Audience
Aspiring frontier labs research engineer familiar with self-supervised learning and SFT. Learns best through:
1. Understanding underlying motivation
2. Canonical examples in math
3. Then code implementation

## Learning Approach
- Socratic method: Ask guiding questions, let learner think, provide feedback
- Map paper concepts directly to implementation files
- Progress from problem → theory → code

---

## Phase 1: The Problem & Core Insight

### Key Concepts
- **Context Rot**: Performance degrades as context length increases, even when information "fits"
- **Context Length Limits**: Fixed windows can't handle truly massive inputs (10M+ tokens)
- **Why existing solutions fail**:
  - RAG: Requires upfront indexing, retrieval quality degrades with corpus size
  - Hierarchical summarization: Imposes fixed structure, loses information
  - Long-context models: Still suffer from context rot, expensive per-query

### Core Insight
> "Treat context as an external environment the LM can programmatically interact with, rather than requiring the entire input to fit within the context window at once."

### Guiding Questions
1. Why can't you stuff 10M tokens into a 10M context window? (Context rot, O(n²) attention)
2. Why doesn't RAG solve this? (Retrieval is the hard part, multi-hop reasoning fails)
3. What if the LM could *decide* how to access context? (Key insight → RLM)

### Paper References
- Blogpost: "Context Rot" section
- Paper: Introduction, Related Work on long-context approaches

---

## Phase 2: The Mathematical Framework

### Formal Definition

**Standard LM Call**:
```
M(q, C) → str
```
Where q = query, C = context (concatenated into prompt)

**RLM Call**:
```
RLM_M(q, C) → str
```
Where:
- Context C is stored externally (not in prompt)
- LM receives only query q
- LM can call `RLM_M(ĉ, Ĉ)` recursively on transformed context
- Environment ℰ mediates interaction

### Code Mapping

| Math | Code | File |
|------|------|------|
| `RLM_M(q, C)` | `rlm.completion(prompt)` | `rlm/core/rlm.py:153` |
| `RLM_M(ĉ, Ĉ)` sub-call | `llm_query(prompt)` | `rlm/environments/local_repl.py` |
| Context C | `context` variable in REPL | `rlm/environments/local_repl.py:88` |
| Environment ℰ | `BaseEnv` implementations | `rlm/environments/base_env.py` |

### Guiding Questions
1. In the math, what does it mean that context is "external"? How is this different from RAG?
2. What enables recursion? Why is `llm_query` the key primitive?
3. What's the relationship between depth and the recursive call stack?

### Paper References
- Blogpost: "Formal Definition" section
- Paper: Section 3 (Methodology)

---

## Phase 3: The Execution Loop

### The Iterative Process

```
for i in range(max_iterations):
    1. Build prompt = message_history + user_prompt
    2. Call LM → get response
    3. Extract code blocks (```repl ... ```)
    4. Execute code in environment
    5. Check for FINAL() or FINAL_VAR()
    6. If final → return answer
    7. Else → format results, append to message_history
```

### Code Walkthrough

**Main loop**: `rlm/core/rlm.py:178-218`
```python
for i in range(self.max_iterations):
    current_prompt = message_history + [build_user_prompt(root_prompt, i)]
    iteration = self._completion_turn(prompt, lm_handler, environment)
    final_answer = find_final_answer(iteration.response, environment=environment)
    if final_answer is not None:
        return RLMChatCompletion(...)
    message_history.extend(format_iteration(iteration))
```

**Single turn**: `rlm/core/rlm.py:236-261`
```python
def _completion_turn(self, prompt, lm_handler, environment):
    response = lm_handler.completion(prompt)
    code_block_strs = find_code_blocks(response)
    code_blocks = []
    for code_block_str in code_block_strs:
        code_result = environment.execute_code(code_block_str)
        code_blocks.append(CodeBlock(code=code_block_str, result=code_result))
    return RLMIteration(prompt, response, code_blocks, ...)
```

### Key Files
- `rlm/core/rlm.py` - Main orchestration
- `rlm/utils/parsing.py` - Code block extraction, final answer detection
- `rlm/utils/prompts.py` - System prompt, user prompt construction

### Guiding Questions
1. Why is this an iterative loop rather than a single call?
2. What information persists across iterations? What doesn't?
3. Why does the LM need to see its previous outputs (message_history)?

---

## Phase 4: The Environment Abstraction

### Design Principle
> The root LM's context window never sees the full context directly.

The environment provides:
1. `context` variable - the massive input stored in memory
2. `llm_query(prompt)` - recursive LM calls
3. `llm_query_batched(prompts)` - concurrent recursive calls
4. `print()` - feedback to the LM (truncated)
5. `FINAL_VAR(var)` - return a computed variable

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    RLM (rlm.py)                     │
│  - Orchestrates iteration loop                       │
│  - Manages message history                           │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│   LMHandler      │   │   Environment    │
│ (lm_handler.py)  │   │ (local_repl.py)  │
│                  │   │                  │
│ - TCP server     │◄──│ - context var    │
│ - Routes llm_    │   │ - exec() code    │
│   query calls    │   │ - llm_query()    │
│ - Usage tracking │   │ - FINAL_VAR()    │
└──────────────────┘   └──────────────────┘
```

### Socket Communication

When code calls `llm_query()` in the REPL:
1. Environment sends request via socket to LMHandler
2. LMHandler calls the LM client
3. Response sent back via socket
4. Result returned to REPL namespace

**Protocol**: `rlm/core/comms_utils.py`
- 4-byte big-endian length prefix + JSON payload
- `LMRequest` / `LMResponse` dataclasses

### Code Walkthrough

**LocalREPL initialization**: `rlm/environments/local_repl.py:60-95`
- Creates sandboxed namespace with safe builtins
- Injects `llm_query`, `llm_query_batched`, `FINAL_VAR` functions
- Loads context from payload

**Code execution**: `rlm/environments/local_repl.py:120-160`
- Captures stdout/stderr
- Tracks LLM calls made during execution
- Returns `REPLResult` with outputs and timing

### Guiding Questions
1. Why use sockets instead of direct function calls?
2. What security considerations arise from `exec()`? How are they mitigated?
3. Why is `llm_query_batched` important for efficiency?

---

## Phase 5: Prompt Engineering as Algorithm Design

### The System Prompt as a "Compiler"

The system prompt (`rlm/utils/prompts.py`) teaches the LM:
1. What variables are available (`context`, `llm_query`, etc.)
2. Decomposition strategies (peeking, chunking, mapping)
3. Output format (`FINAL()`, `FINAL_VAR()`)

### Decomposition Strategies Encoded in Prompt

**Peeking**: Examine context structure before committing
```python
chunk = context[:10000]
answer = llm_query(f"What is the structure? {chunk}")
```

**Chunking + Mapping**: Divide and conquer
```python
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
answers = llm_query_batched([f"Extract info: {c}" for c in chunks])
final = llm_query(f"Aggregate: {answers}")
```

**Iterative Refinement**: Maintain state across chunks
```python
buffer = ""
for section in context:
    buffer = llm_query(f"Update buffer {buffer} with {section}")
```

### Key Insight
> The RLM doesn't execute a fixed algorithm—it *synthesizes* one at inference time based on the query and context structure.

### Code Walkthrough

**System prompt**: `rlm/utils/prompts.py:6-50`
- Describes REPL environment capabilities
- Provides example decomposition strategies
- Specifies final answer format

**User prompt construction**: `rlm/utils/prompts.py:88-99`
- Iteration 0: "You have not interacted with the REPL yet"
- Iteration N: "The history before is your previous interactions"

### Guiding Questions
1. How is the system prompt like a "compiler" for natural language algorithms?
2. What would happen if you removed the example strategies from the prompt?
3. How does the prompt prevent the LM from answering without exploring context?

---

## Phase 6: Advanced Topics

### Depth and Recursion

Currently `max_depth=1` is supported:
- Depth 0: Root RLM with full REPL capabilities
- Depth 1: Sub-calls via `llm_query` → treated as regular LM (no further recursion)

**Code**: `rlm/core/rlm.py:172-173`
```python
if self.depth >= self.max_depth:
    return self._fallback_answer(prompt)
```

### Isolated Environments

For production safety:
- `DockerREPL`: Runs in container
- `ModalREPL`: Runs on Modal.com sandboxes
- `PrimeREPL`: Prime Intellect sandboxes (not yet implemented)

### Usage Tracking

Aggregated across all recursive calls:
- `lm_handler.get_usage_summary()` returns total tokens
- Enables cost comparison with baselines

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `rlm/core/rlm.py` | Main RLM class, completion loop |
| `rlm/core/lm_handler.py` | TCP server for routing llm_query calls |
| `rlm/core/types.py` | Data structures (RLMIteration, REPLResult, etc.) |
| `rlm/environments/local_repl.py` | Default sandboxed Python REPL |
| `rlm/environments/base_env.py` | Abstract environment interface |
| `rlm/clients/base_lm.py` | Abstract LM client interface |
| `rlm/utils/prompts.py` | System prompt, prompt construction |
| `rlm/utils/parsing.py` | Code block extraction, FINAL detection |
| `examples/quickstart.py` | Example usage |

---

## Progress Tracker

- [ ] Phase 1: The Problem & Core Insight
- [ ] Phase 2: The Mathematical Framework
- [ ] Phase 3: The Execution Loop
- [ ] Phase 4: The Environment Abstraction
- [ ] Phase 5: Prompt Engineering as Algorithm Design
- [ ] Phase 6: Advanced Topics

---

## Current Session State

**Current Phase**: Phase 1
**Current Question**: Why can't you (1) stuff 10M tokens in a 10M window, or (2) use RAG?
**Awaiting**: Learner's response to guiding question
