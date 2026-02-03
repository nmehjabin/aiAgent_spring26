# langgraph_dual_llm_agent.py
#
# LangGraph agent with a UNIFIED chat history, dynamic Llama ↔ Qwen routing,
# and SQLite-backed checkpointing for crash recovery.
#
# This version is modified to follow the SAME checkpoint/resume method
# as the class reference:
#   - checkpointer = SqliteSaver.from_conn_string("checkpoint.db")
#   - saved = graph.get_state(config)
#   - if saved and saved.next: graph.invoke(None, config=config)  (resume)
#   - else: graph.invoke(initial_state, config=config)            (fresh)
#   - try/except SystemExit and KeyboardInterrupt

import re, json, time, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List
from typing_extensions import Annotated
import operator


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    print("Using CPU for inference")
    return "cpu"


# ---------------------------------------------------------------------------
# Unified message (stored in state — no LangChain types needed)
# ---------------------------------------------------------------------------
class ChatMessage(TypedDict):
    speaker: str   # "Human" | "Llama" | "Qwen"
    content: str


# ---------------------------------------------------------------------------
# AgentState  (LangGraph state schema)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    history:       Annotated[List[ChatMessage], operator.add]
    active_model:  str                 # "Llama" | "Qwen"
    route:         str                 # "end" | "retry_input" | "to_llm"


# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = (
    "You are {self_name}, a helpful AI assistant. "
    "You are in a conversation that includes a Human and {other_llm}. "
    "Messages from {other_llm} will be prefixed with '{other_llm}:' so you can identify them. "
    "Messages from the Human have no prefix. "
    "Reply directly to whoever spoke last. Do NOT prefix your reply with your own name. "
    "Keep responses concise (1-3 sentences) and conversational. "
    "You may remember and use facts stated earlier in the conversation."
)

def system_prompt_for(model_name: str) -> str:
    other_llm = "Qwen" if model_name == "Llama" else "Llama"
    return SYSTEM_PROMPT_TEMPLATE.format(self_name=model_name, other_llm=other_llm)


# ---------------------------------------------------------------------------
# Post-processor: stop the model from role-playing other speakers
# ---------------------------------------------------------------------------
def _strip_other_speakers(text: str, active_model: str) -> str:
    lines = text.split('\n')
    kept = []
    for line in lines:
        stripped = line.strip()
        m = re.match(r'^(Human|Llama|Qwen)\s*:', stripped, re.IGNORECASE)
        if m:
            if m.group(1).lower() != active_model.lower():
                break
            kept.append(stripped[m.end():].strip())
        else:
            kept.append(line)
    return '\n'.join(kept).strip()


# ---------------------------------------------------------------------------
# Project unified history → chat-template-ready dicts for model X
# ---------------------------------------------------------------------------
def project_history(history: List[ChatMessage], model_name: str) -> List[dict]:
    projected = [{"role": "system", "content": system_prompt_for(model_name)}]

    for msg in history:
        if msg["speaker"] == model_name:
            projected.append({"role": "assistant", "content": msg["content"]})
        elif msg["speaker"] == "Human":
            projected.append({"role": "user", "content": msg["content"]})
        else:
            projected.append({"role": "user", "content": f"{msg['speaker']}: {msg['content']}"})

    return projected


# ---------------------------------------------------------------------------
# Load a single model → (HuggingFacePipeline, tokenizer)
# ---------------------------------------------------------------------------
def load_model(model_id: str, label: str):
    device = get_device()
    print(f"[{label}] Loading {model_id} …")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        max_length=None,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"[{label}] Loaded successfully!")
    return llm, tokenizer


# ---------------------------------------------------------------------------
# Convert projected chat dicts → single prompt string via chat template
# ---------------------------------------------------------------------------
def chat_to_prompt(chat: List[dict], tokenizer) -> str:
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Routing: detect which model the user wants
# ---------------------------------------------------------------------------
_LLAMA_RE = re.compile(r'\bllama\b', re.IGNORECASE)
_QWEN_RE  = re.compile(r'\bqwen\b',  re.IGNORECASE)

def detect_target_model(text: str, current: str) -> str:
    has_llama = bool(_LLAMA_RE.search(text))
    has_qwen  = bool(_QWEN_RE.search(text))

    if has_llama and not has_qwen:
        return "Llama"
    if has_qwen and not has_llama:
        return "Qwen"
    return current


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE      = "conversations.log"
CHECKPOINT_DB = "checkpoint.db"
THREAD_ID     = "main_session"

def log_message(speaker: str, content: str):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "speaker": speaker,
        "content": content,
    }
    with open(LOG_FILE, "a") as f:
        json.dump(entry, f)
        f.write("\n")

def log_separator():
    with open(LOG_FILE, "a") as f:
        f.write("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Save Mermaid diagram
# ---------------------------------------------------------------------------
def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need: pip install grandalf")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def create_graph(llama_llm, llama_tok, qwen_llm, qwen_tok, checkpointer):

    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("Enter your text  (or 'quit' / 'new' to exit / reset):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            log_separator()
            return {"route": "end"}

        if user_input.lower() == "crash":
            print("💥 Simulating crash now...")
            os._exit(1)

        if user_input.lower() == "new":
            print("\n🗑️  Clearing conversation and checkpoint …")
            log_separator()
            if os.path.exists(CHECKPOINT_DB):
                os.remove(CHECKPOINT_DB)
            return {"route": "end"}

        if user_input.strip() == "":
            print("Empty input — please type something.")
            return {"route": "retry_input"}

        new_active = detect_target_model(user_input, state.get("active_model", "Llama"))
        log_message("Human", user_input)

        return {
            "route": "to_llm",
            "history": [ChatMessage(speaker="Human", content=user_input)],
            "active_model": new_active,
        }

    def route_after_input(state: AgentState) -> str:
        r = state.get("route", "retry_input")
        if r == "end":
            return END
        if r == "retry_input":
            return "get_user_input"
        return "call_llama" if state.get("active_model") == "Llama" else "call_qwen"

    def _call_model(state: AgentState, model_name: str, llm, tok) -> dict:
        history = state["history"]
        chat = project_history(history, model_name)
        prompt = chat_to_prompt(chat, tok)

        print(f"\n[Calling {model_name} with {len(history)} history messages …]")
        raw = llm.invoke(prompt)

        if isinstance(raw, list) and raw:
            raw = raw[0]
        response = str(raw).strip()

        cleaned = _strip_other_speakers(response, model_name)
        if cleaned == "" and response != "":
            print(f"\n  [{model_name}] output mis-attributed; using raw text")
            cleaned = response

        response = cleaned
        log_message(model_name, response)

        return {"history": [ChatMessage(speaker=model_name, content=response)]}

    def call_llama(state: AgentState) -> dict:
        return _call_model(state, "Llama", llama_llm, llama_tok)

    def call_qwen(state: AgentState) -> dict:
        return _call_model(state, "Qwen", qwen_llm, qwen_tok)

    def print_last(state: AgentState) -> dict:
        last = state["history"][-1]
        print("\n" + "-" * 50)
        print(f"{last['speaker']}:")
        print("-" * 50)
        print(last["content"])
        return {}

    g = StateGraph(AgentState)
    g.add_node("get_user_input", get_user_input)
    g.add_node("call_llama", call_llama)
    g.add_node("call_qwen", call_qwen)
    g.add_node("print_last", print_last)

    g.add_edge(START, "get_user_input")

    g.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    g.add_edge("call_llama", "print_last")
    g.add_edge("call_qwen", "print_last")
    g.add_edge("print_last", "get_user_input")

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Pretty-print recovered history
# ---------------------------------------------------------------------------
def print_recovered_history(history: list):
    print("\n " + "=" * 54)
    print("   RECOVERED SESSION — history before crash:")
    print(" " + "=" * 54)
    for msg in history:
        print(f"\n  {msg['speaker']}:")
        print(f"  {msg['content']}")
    print("\n" + "─" * 57)
    print("   Resuming conversation …")
    print("─" * 57)


# ---------------------------------------------------------------------------
# Class-style runner (resume if saved.next exists)
# ---------------------------------------------------------------------------
def run_chat(graph, thread_id=THREAD_ID):
    config = {"configurable": {"thread_id": thread_id}}

    try:
        saved = graph.get_state(config)

        if saved and saved.next:
            print("\n RESUMING from checkpoint...")
            hist = saved.values.get("history", [])
            if hist:
                print_recovered_history(hist)
            print(f"   Next nodes: {saved.next}")
            print(f"   Active model: {saved.values.get('active_model', 'Llama')}")
            print(f"   History length: {len(hist)}")

            return graph.invoke(None, config=config)

        print("\n STARTING new chat session...")
        initial_state: AgentState = {
            "history": [],
            "active_model": "Llama",
            "route": "retry_input",
        }
        return graph.invoke(initial_state, config=config)

    except SystemExit as e:
        print(f"\n Program crashed: {e}")
        print("State has been saved. Restart the program to resume.")
        return None

    except KeyboardInterrupt:
        print("\n Interrupted (Ctrl+C). State should be saved. Restart to resume.")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" LangGraph Dual-LLM Agent  (Llama + Qwen, unified history)")
    print(" with SQLite checkpointing for crash recovery")
    print("=" * 60)

    llama_llm, llama_tok = load_model("meta-llama/Llama-3.2-1B-Instruct", "Llama")
    qwen_llm,  qwen_tok  = load_model("Qwen/Qwen2.5-1.5B-Instruct", "Qwen")

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = create_graph(llama_llm, llama_tok, qwen_llm, qwen_tok, checkpointer)
        print("Graph built!")
        save_graph_image(graph, filename="lg_graph.png")

        print("\n💡 Tip: mention 'Llama' or 'Qwen' to switch models.")
        print("   'quit' to exit, 'new' to wipe history and start fresh.")
        print("   'crash' to simulate a crash.\n")

        run_chat(graph, thread_id=THREAD_ID)


if __name__ == "__main__":
    main()
