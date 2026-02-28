# Task 3: LangGraph — Manual ToolNode vs. ReAct Agent

Comparing two approaches to building a persistent multi-turn conversational agent in LangGraph:
a **manual ToolNode pipeline** (explicit graph with `call_model → tools` loop) and a
**ReAct agent** (`create_react_agent` with built-in reasoning loop).

Both systems support the same three tools, maintain conversation history, and handle
special commands (`verbose`, `quiet`, `exit`). The graphs reveal how differently they
are structured underneath.

---

## System Overview

| Feature | Manual ToolNode | ReAct Agent |
|---|---|---|
| **Tool dispatch** | Explicit `ToolNode` with conditional routing | Built into `create_react_agent` |
| **Graph structure** | `input → call_model ⇄ tools → output → trim_history` | `input → call_react_agent → output → trim_history` |
| **Looping** | `call_model ↔ tools` bidirectional edge | Internal to agent node |
| **History trimming** | Explicit `trim_history` node | Explicit `trim_history` node |
| **Parallel tool dispatch** | ✅ Yes — ToolNode dispatches multiple tools simultaneously | ❌ No — sequential within agent |

### Available Tools
- `get_weather(location)` — returns weather data
- `get_population(city)` — returns population data
- `calculate(expression)` — evaluates math expressions

---

## Graph Diagrams

### Manual ToolNode Graph
![Manual Tool Graph](langchain_manual_tool_graph.png)

> The `call_model` and `tools` nodes form a bidirectional loop (solid arrows). After
> the model decides no more tools are needed, it routes to `output`, then `trim_history`,
> then back to `input` to await the next turn.

### ReAct Agent Graph
![ReAct Agent Graph](langchain_react_agent.png)

> The internal `agent → tools` loop is hidden inside `call_react_agent`. From the outer
> graph's perspective, it is a single node. The `__end__` node is the agent's internal
> termination state, not the conversation's end.

### Conversation Graph (ReAct)
![Conversation Graph](langchain_conversation_graph.png)

> The outer conversation wrapper: `input → call_react_agent → output → trim_history → input`.
> The complexity of reasoning and tool use is invisible at this level.

---

## Session Transcript Highlights

### Manual ToolNode — key trace
```
[DEBUG] Model requested 1 tool call(s): get_weather({'location': 'New York'})
[DEBUG] Routing to tools              ← ToolNode executes
[DEBUG] Calling model with 7 messages ← model re-invoked with tool result
[DEBUG] Model response (no tools): "The weather in New York is currently sunny..."
```

### ReAct Agent — key trace
```
[DEBUG] Invoking ReAct agent with 3 messages in history
[DEBUG] Agent generated 3 new messages   ← think + tool call + tool result, all internal
[DEBUG] Tool calls: ['get_weather']
[DEBUG] Response preview: "The current weather in New York is sunny..."
```

---

## Portfolio Questions

### 1. What Python features does ToolNode use for parallel dispatch? What tools benefit most?

`ToolNode` uses **`asyncio`** and **`concurrent.futures`** (specifically `ThreadPoolExecutor`
for synchronous tools and `asyncio.gather` for async ones) to dispatch multiple tool calls
simultaneously. When the model returns a message containing multiple `tool_call` objects,
ToolNode iterates over them and fans them out — each tool runs in its own thread or
coroutine rather than waiting for the previous one to complete.

The key Python mechanisms involved are:
- **`asyncio.gather()`** — awaits multiple coroutines concurrently in the async path
- **`ThreadPoolExecutor`** — runs blocking (synchronous) tool functions in parallel threads
- **`functools.reduce` / list comprehension** — assembles the list of `ToolMessage` results
  back into the conversation state after all tools finish

**Tools that benefit most from parallel dispatch:**

| Tool type | Why parallel helps | Example |
|---|---|---|
| **External API calls** | Network latency is the bottleneck, not CPU | `get_weather` + `get_population` simultaneously |
| **Database queries** | I/O bound — can run while other queries wait | Two separate SQL lookups |
| **File reads** | Disk I/O is independent per file | Reading multiple documents |
| **Web scraping** | Each HTTP request is independent | Fetching 5 URLs at once |

Tools that do **not** benefit:
- CPU-bound calculations — Python's GIL limits true thread parallelism for pure Python math
- Tools with shared mutable state — race conditions become a risk
- Sequential-dependent tools — if tool B needs tool A's output, they cannot run in parallel

In this exercise, if a user asked *"What is the weather in New York AND the population of Chicago?"* simultaneously, ToolNode would fire both `get_weather` and `get_population` in parallel. The ReAct agent would call them one at a time.

---

### 2. How do the two programs handle special inputs like "verbose" and "exit"?

Both programs intercept special commands **at the `input_node`** before any routing
decision is made. The `input_node` reads the raw user string and checks it before
appending it to conversation history or sending it to the model.

**`exit` / `quit`:**
```
[DEBUG] Exit command received
[DEBUG] Routing to END (exit requested)
[SYSTEM] Conversation ended. Goodbye!
```
When the input is `"exit"` or `"quit"`, `input_node` sets a flag in the graph state and
routes directly to `__end__`, bypassing `call_model` entirely. The LLM never sees the
word "exit" — it is consumed by the graph control flow.

**`verbose` / `quiet`:**
These toggle a `verbose` boolean in the graph state. The `[DEBUG]` lines seen throughout
the transcript are gated on this flag. When `verbose=False` (quiet mode), the `call_model`
and `tools` nodes still execute — but suppress their debug `print` statements. The graph
structure is identical either way; only the logging output changes.

**Key design insight:** Treating these as **graph-level routing decisions** rather than
prompt injections means the LLM is never confused by them. If you sent "exit" to the model
it might try to help you exit something. The `input_node` intercepts it first, making the
control flow clean and deterministic regardless of what the model might do with that word.

The two implementations handle this identically because both wrap the agent in the same
outer conversation graph (`input → [agent logic] → output → trim_history`). The special
command handling lives in the shared `input_node`, not in either agent implementation.

---

### 3. Compare the graph diagrams — how do they differ?

The most important difference is **where the tool loop lives**.

**Manual ToolNode graph (Image 2):**
```
__start__ → input ──→ call_model ⇄ tools
                              ↓
                           output → trim_history → input
```
- The `call_model ↔ tools` loop is **explicit and visible** as two distinct nodes
  connected by bidirectional conditional edges
- You can see exactly when the model is deciding (call_model) vs. when tools are
  executing (tools)
- The graph has **5 functional nodes**: input, call_model, tools, output, trim_history
- Adding a new routing rule (e.g., "if tool fails, go to error_handler") means adding
  a new node and edge to this visible graph

**ReAct Agent graph (Image 1 — outer wrapper):**
```
__start__ → input → call_react_agent → output → trim_history → input
```
- The `call_react_agent` node is a **black box** — the agent's internal `agent ↔ tools`
  loop (Image 3) is hidden inside it
- The outer conversation graph has only **4 functional nodes**; the tool loop is abstracted away
- The `__end__` visible in Image 1 is the *agent's* internal end state, not the
  conversation's end — the conversation continues looping at the outer level

**ReAct internal graph (Image 3):**
```
__start__ → agent ──→ __end__
                 ↘ tools ↗
```
- This is what `create_react_agent` builds internally — a standard Reason + Act loop
- It is structurally identical to the manual graph's `call_model ↔ tools` portion,
  but encapsulated as a sub-graph you do not directly control

**Summary of differences:**

| Aspect | Manual ToolNode | ReAct Agent |
|---|---|---|
| Tool loop visibility | Explicit in main graph | Hidden inside sub-graph |
| Node count (outer graph) | 5 | 4 |
| Routing customisability | Every edge is editable | Internal edges are fixed |
| Graph readability | More complex but fully transparent | Simpler outer view, opaque internals |
| `trim_history` position | Same — after output | Same — after output |

Both graphs share the identical outer conversation loop structure — the difference is
purely in how much of the agent's internal machinery is exposed to the developer.

---

### 4. When would the manual ToolNode approach be preferable to the ReAct agent?

The ReAct agent imposes a fixed **Reason → Act → Observe → Reason** cycle. Every tool
call goes through the model first (reason), executes (act), feeds back to the model
(observe), and repeats. This structure is excellent for open-ended reasoning but becomes
a liability in several real scenarios:

**Scenario: Multi-step pipeline with guaranteed tool order**

Imagine a medical triage system: always run `check_vitals()` first, then
`lookup_patient_history()`, then `calculate_risk_score()`. The ReAct agent's reasoning
step might decide to skip `lookup_patient_history()` if the vitals look fine — it reasons
its way to an early exit. The manual graph can enforce the exact sequence with hard edges,
making it impossible to skip a step regardless of what the model thinks.

**Scenario: Tool results that must be validated before the model sees them**

If `get_financial_data()` returns sensitive figures, you may want a
`validate_and_redact()` node to sanitise the output before it enters the conversation
history. In the manual graph, you insert a node between `tools` and `call_model`. In
ReAct, tool results flow directly back into the agent with no interception point.

**Scenario: Parallel tool calls that must all complete before the model responds**

As discussed in Q1 — if a user asks for weather in 5 cities simultaneously, ToolNode
dispatches all 5 at once and waits for all 5 results before re-invoking the model. The
ReAct agent calls `get_weather(city1)`, gets the result, reasons, calls `get_weather(city2)`,
and so on — 5 sequential round-trips to the model instead of 1.

**Scenario: Hard fallback logic**

If a tool fails (raises an exception), the manual graph can route to a dedicated
`error_handler` node that either retries, alerts a human, or returns a safe default
without asking the model to handle the failure in natural language. The ReAct agent
passes the exception back to the model and asks it to reason about what to do — which
works but is slower, less deterministic, and uses more tokens.

**General rule:** Use ReAct when the *reasoning path* between tools is unknown in advance
and needs to be discovered dynamically. Use manual ToolNode when the control flow between
tools is known, must be enforced reliably, or requires nodes that operate on tool results
before the model sees them.

---

## Conclusion

Both implementations produce correct, identical user-facing behaviour for the three tools
tested. The session transcript confirms that weather, math, and open-ended questions
(self-respect) all work in both systems. The meaningful difference is not in what they
*do* but in who controls the *how*:

- **ReAct** delegates control to the model's reasoning — flexible, fewer lines of code,
  but the tool loop is opaque and fixed
- **Manual ToolNode** delegates control to the graph — explicit, fully customisable,
  but requires you to define every routing decision yourself

For a production RAG system where tool calls have business logic constraints, security
requirements, or performance SLAs, the manual ToolNode approach is the safer foundation.
For a general-purpose assistant where the model should decide its own reasoning path,
ReAct is the right choice.
