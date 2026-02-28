# Topic 2: LangGraph Agent Frameworks

A series of progressively enhanced LangGraph agents built on top of Hugging Face LLMs (Llama-3.2-1B-Instruct and Qwen2.5-1.5B-Instruct), exploring graph-based agent design, parallel inference, chat history, model routing, and crash recovery.

---

## Table of Contents

1. [File Structure](#file-structure)
2. [Setup & Requirements](#setup--requirements)
3. [Task Descriptions & Results](#task-descriptions--results)
   - [Task 1 – Verbose/Quiet Tracing](#task-1--verbosequiet-tracing)
   - [Task 2 – Empty Input Handling & 3-Way Branch](#task-2--empty-input-handling--3-way-branch)
   - [Task 3 – Parallel Llama + Qwen](#task-3--parallel-llama--qwen)
   - [Task 4 – Model Routing by Prefix](#task-4--model-routing-by-prefix)
   - [Task 5 – Chat History with Message API](#task-5--chat-history-with-message-api)
   - [Task 6 – Chat History + Model Switching](#task-6--chat-history--model-switching)
   - [Task 7 – Crash Recovery & Checkpointing](#task-7--crash-recovery--checkpointing)
4. [Key Observations](#key-observations)

---

## File Structure

```
Topic2Frameworks/
│
├── README.md                     This file
├── requirements.txt              Python dependencies
│
├── langgraph_simple_agent.py     Original base agent (provided starter code)
│
├── task3.py                      Parallel Llama + Qwen agent
├── task4.py                      Model routing by "Hey Qwen" prefix
├── task5.py                      Chat history with Message API (Llama only)
├── task6.py                      Chat history + Llama/Qwen model switching
├── task7.py                      Crash recovery with LangGraph checkpointing
│
├── lg_graph_task1.png            Graph visualization: 3-way branch (Tasks 1 & 2)
├── lg_graph_task2_2.png          Graph visualization: empty-input loop
├── lg_graph_task5.png            Graph visualization: chat history (Llama only)
├── lg_graph_task6.png            Graph visualization: dual-model + history
│
├── task1_output.txt              Terminal output: verbose/quiet tracing
├── task2_1_output.txt            Terminal output: empty input hallucination demo
├── task2_2_output.txt            Terminal output: empty input blocked by 3-way branch
├── task3_output.txt              Terminal output: parallel model responses
├── task4_output.txt              Terminal output: model routing by prefix
├── task5_output.txt              Terminal output: chat history memory test
└── task7_output.txt              Terminal output: crash simulation + recovery
```

---

## Setup & Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt** includes:
- `langgraph`
- `langchain`
- `transformers`
- `torch`
- `accelerate`

All models are loaded from Hugging Face Hub and run on CPU (or GPU if available).

---

## Task Descriptions & Results

### Task 1 – Verbose/Quiet Tracing

**File:** `langgraph_simple_agent.py` (modified)  
**Graph:** `lg_graph_task1.png`  
**Output:** `task1_output.txt`
<img width="364" height="348" alt="image" src="https://github.com/user-attachments/assets/323bb691-349d-40c6-9904-8be7cee39c56" />
**What was done:**  
Modified the base LangGraph agent so that typing `verbose` at the prompt enables per-node tracing output to stdout, and typing `quiet` disables it. The graph has a 3-way conditional edge from `get_user_input`: to `call_llm`, to `__end__` (on `quit`), or back to itself.

**Result:**  
The agent correctly loads Llama-3.2-1B-Instruct, accepts user input, and routes through `call_llm` then `print_response` → back to `get_user_input`. The verbose/quiet toggle is acknowledged in the prompt. The output also confirms that the model has **no memory between turns** — when asked "Did you remember what I asked about prioritizing tasks?", it replies that it has no recollection of prior conversation. This motivates Task 5.

---

### Task 2 – Empty Input Handling & 3-Way Branch

**File:** `langgraph_simple_agent.py` (further modified)  
**Graph:** `lg_graph_task2_2.png`  
**Output:** `task2_1_output.txt`, `task2_2_output.txt`
<img width="364" height="348" alt="image" src="https://github.com/user-attachments/assets/d5fbd92f-70f7-401d-ba90-6b88ff89373e" />

**What was done:**  
First, empty input was deliberately passed to the LLM to observe its behavior. Then the `get_user_input` router was modified to implement a **3-way conditional branch**: routing to `call_llm` (valid input), `__end__` (on `quit`), or **back to `get_user_input` itself** (on empty input), preventing the LLM from ever receiving a blank prompt.

**Observed behavior before fix (`task2_1_output.txt`):**  
When given an empty string, the model hallucinated completely unrelated and incoherent content — first a math puzzle about a 5×5 grid of squares, then on the second empty input a discussion about 1960s American literature. Crucially, both hallucinations were entirely different, demonstrating the model's sensitivity to random sampling with no grounding signal.

**What this reveals:**  
Small LLMs like Llama-3.2-1B-Instruct lack the robustness to recognize and gracefully handle degenerate inputs. Without a meaningful prompt, they latch onto noise in the sampling distribution and produce unpredictable, confident-sounding nonsense. Larger models (e.g., GPT-4, Claude) would typically respond with something like "It seems you didn't enter anything — how can I help?" Small models cannot make this meta-level judgment.

---

### Task 3 – Parallel Llama + Qwen

**File:** `task3.py`  
**Graph:** `lg_graph_task6.png` (dual-branch graph)  
**Output:** `task3_output.txt`
<img width="509" height="348" alt="image" src="https://github.com/user-attachments/assets/f86c7bca-213b-4b09-a079-9ed2cc08f76f" />

**What was done:**  
The `call_llm` node was replaced by two parallel nodes — `call_llama` and `call_qwen` — both receiving the same input simultaneously. Their outputs flow into a shared `print_last` node that displays both responses side by side.

**Result:**  
Both models responded to the same prompts. On the multiplication question `17 * 23`:
- **Llama** produced an incorrect answer (221 instead of 391), making an arithmetic error in its step-by-step breakdown.
- **Qwen** began a correct approach using the distributive property but was cut off mid-calculation due to the token limit.

On the productivity question, both models gave sensible but differently structured advice — Llama used numbered lists with bold headers, while Qwen gave a more concise, plain response.

---

### Task 4 – Model Routing by Prefix

**File:** `task4.py`  
**Output:** `task4_output.txt`

**What was done:**  
A routing function was added so that if the user's input begins with `"Hey Qwen"` (case-insensitive), the message is sent to Qwen; otherwise it goes to Llama. 


---

### Task 5 – Chat History with Message API

**File:** `task5.py`  
**Graph:** `lg_graph_task5.png`  
**Output:** `task5_output.txt`
<img width="372" height="348" alt="image" src="https://github.com/user-attachments/assets/80c1d360-69a1-4b9f-bfce-b9b927994785" />

**What was done:**  
The LangGraph state was extended to include a `messages` list using LangChain's Message API (`HumanMessage`, `AIMessage`, `SystemMessage`). Each turn appends to this list, which is passed as the full conversation context to the LLM on every call. Qwen was disabled for this task. A `new` command was added to reset the conversation history.

**Result:**  
Chat memory works correctly across turns:
- The user introduces themselves as Nadia.
- The model acknowledges this and asks a follow-up question about her PhD experience.
- Later, when asked "What is my name?", the model correctly recalls **Nadia** — demonstrating that the full message history is being forwarded on each call.

This is a significant improvement over Task 1, where the same question yielded "I have no recollection of our previous conversation."

---

### Task 6 – Chat History + Model Switching

**File:** `task6.py`  
**Graph:** `lg_graph_task6.png`  
**Output:** *(see task7_output.txt for a combined session)*

**What was done:**  
Combined the chat history system from Task 5 with the model-routing logic from Task 4. Since a chat history only has `user`/`assistant` roles, a naming convention was adopted to distinguish participants:

- Human turns are stored as `{"role": "user", "content": "Human: <message>"}`
- Llama turns are stored with `"assistant"` role when Llama was the last to speak, and as `"user"` role (prefixed `"Llama: ..."`) when the current model is Qwen (and vice versa)
- Each model receives a system prompt naming all participants and clarifying its own role

This allows each model to "see" prior turns from the other model as part of the shared conversation context.

**Example history passed to Qwen after a Llama exchange:**
```
[
  {"role": "user",      "content": "Human: What is the best ice cream flavor?"},
  {"role": "user",      "content": "Llama: There is no one best flavor, but the most popular is vanilla."}
]
```

---

### Task 7 – Crash Recovery & Checkpointing

**File:** `task7.py`  
**Output:** `task7_output.txt`

**What was done:**  
LangGraph's built-in checkpointing was integrated using `MemorySaver` (or `SqliteSaver` for persistence across process restarts). The graph state — including full message history, active model, and current node — is saved after every node execution. A simulated `crash` command triggers recovery to demonstrate the feature. On restart, the program detects an existing checkpoint and resumes from the last saved state.

**Result:**  
The output demonstrates a complete crash-recovery cycle:
1. Nadia introduces herself - Llama responds
2. She asks Qwen if it remembers her name - Qwen correctly recalls "Nadia"
3. She asks about sunburn protection - Qwen responds with sun safety tips
4. She types `crash` - the session simulates a crash
5. On resume, the **full prior conversation is recovered** and displayed
6. The conversation continues seamlessly: Qwen still remembers her name 9 messages later

This demonstrates that LangGraph checkpointing preserves not just the message list but also the active model and graph position, enabling true fault-tolerant long-running agents.

---

## Key Observations

| # | Observation |
|---|-------------|
| 1 | Small LLMs (Llama-3.2-1B) hallucinate confidently on empty input — they cannot self-diagnose degenerate prompts |
| 2 | Graph-level empty-input guards (routing back to `get_user_input`) are more robust than prompt-level validation |
| 3 | Llama-3.2-1B made an arithmetic error on a simple multiplication; Qwen2.5-1.5B used a correct method but hit the token limit |
| 4 | LangChain's Message API enables stateful multi-turn conversations that small LLMs can leverage effectively within context length |
| 5 | Multi-model chat histories can be managed with a single `user`/`assistant` role scheme by prefixing speaker names |
| 6 | LangGraph's checkpointing allows crash recovery with zero conversation loss, making it suitable for long-running production agents |
