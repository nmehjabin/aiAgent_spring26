# langgraph_dual_llm_agent.py
#
# LangGraph agent with a UNIFIED chat history and dynamic Llama ↔ Qwen routing.
#
# Key design decisions
# --------------------
# 1. Unified message store
#        Each message is just { "speaker": "Human"|"Llama"|"Qwen", "content": str }
#        This is model-agnostic — no LangChain Message types in the store.
#
# 2. Per-model projection (project_history)
#        Before calling model X the unified list is reshaped:
#            X's own past messages  → role "assistant"   (so the model sees them as its own)
#            everyone else's msgs   → role "user"        (prefixed with speaker name)
#        A model-specific system prompt is prepended.
#
# 3. Routing
#        A simple regex scanner checks whether the user's latest message
#        mentions "llama" or "qwen" (case-insensitive).  If neither is
#        mentioned the router keeps the previously-active model.  Default
#        on the very first turn is Llama.
#
# 4. Graph topology  (identical to the original, just with two LLM nodes)
#
#        START → get_user_input ──(empty)──► get_user_input   (self-loop)
#                     │                      │
#                     │ (quit)               │ (valid input)
#                     ▼                      ▼
#                   END            route_to_model
#                                    │           │
#                               call_llama   call_qwen
#                                    │           │
#                                    ▼           ▼
#                                print_last ──► get_user_input
#
# 5. Conversation logging
#        Every exchange is appended to  conversations.log  so you can
#        review interesting multi-model dialogues after the fact.

import re, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional
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
    history:       Annotated[List[ChatMessage], operator.add]   # LangGraph appends via operator.add
    active_model:  str                 # "Llama" | "Qwen"
    route:         str                 # "end" | "retry_input" | "to_llm"


# ---------------------------------------------------------------------------
# System-prompt templates  (parameterised by the receiving model)
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
    return SYSTEM_PROMPT_TEMPLATE.format(
        self_name=model_name,
        other_llm=other_llm,
    )


# ---------------------------------------------------------------------------
# Post-processor: stop the model from role-playing other speakers
# ---------------------------------------------------------------------------
# Small models often continue generating in a multi-turn pattern when they
# see consecutive "user" messages prefixed with names.  This regex finds the
# first line that looks like another speaker taking over and chops there.
_SPEAKER_LINE_RE = re.compile(
    r'(?:^|\n)\s*(?:Human|Llama|Qwen)\s*:', re.IGNORECASE
)

def _strip_other_speakers(text: str, active_model: str) -> str:
    """
    Keep only the first speaker-turn in `text`.  If the model outputs:
        "Great question!\nQwen: blah\nLlama: blah"
    and active_model is "Llama", we return just "Great question!".
    """
    # Walk line-by-line; stop at the first line that is a *different* speaker header
    lines = text.split('\n')
    kept  = []
    for line in lines:
        stripped = line.strip()
        # Check if this line starts with "SomeName:"
        m = re.match(r'^(Human|Llama|Qwen)\s*:', stripped, re.IGNORECASE)
        if m:
            # It's a speaker header.  If it's NOT the active model, stop here.
            if m.group(1).lower() != active_model.lower():
                break
            # It IS the active model re-prefixing itself — strip the prefix and keep going
            kept.append(stripped[m.end():].strip())
        else:
            kept.append(line)

    return '\n'.join(kept).strip()


# ---------------------------------------------------------------------------
# Project unified history → chat-template-ready dicts for model X
# ---------------------------------------------------------------------------
def project_history(history: List[ChatMessage], model_name: str) -> List[dict]:
    """
    Reshapes the unified history into chat-template-ready dicts for model_name.

    Three cases per message:
        • speaker == model_name  →  role "assistant", plain content
                                    (the model sees these as its own prior turns)
        • speaker == "Human"     →  role "user", plain content
                                    (no prefix — the role label is sufficient,
                                     and adding "Human:" confuses small models
                                     into role-playing as the other LLM)
        • speaker == other LLM   →  role "user", prefixed "LLMname: ..."
                                    (prefix is the ONLY way to distinguish this
                                     from a Human message inside the same role)
    """
    projected = [{"role": "system", "content": system_prompt_for(model_name)}]

    for msg in history:
        if msg["speaker"] == model_name:
            # Own past turn → assistant
            projected.append({"role": "assistant", "content": msg["content"]})
        elif msg["speaker"] == "Human":
            # Human turn → user, NO prefix
            projected.append({"role": "user", "content": msg["content"]})
        else:
            # The other LLM's turn → user, WITH prefix
            projected.append({
                "role": "user",
                "content": f"{msg['speaker']}: {msg['content']}"
            })

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
        max_length=None,          # suppress the max_length/max_new_tokens warning
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,   # do NOT echo the prompt back in the output
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"[{label}] Loaded successfully!")
    return llm, tokenizer


# ---------------------------------------------------------------------------
# Convert projected chat dicts → single prompt string via chat template
# ---------------------------------------------------------------------------
def chat_to_prompt(chat: List[dict], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Route-detection: which model does the user want to talk to?
# ---------------------------------------------------------------------------
_LLAMA_RE = re.compile(r'\bllama\b', re.IGNORECASE)
_QWEN_RE  = re.compile(r'\bqwen\b',  re.IGNORECASE)

def detect_target_model(text: str, current: str) -> str:
    """Return 'Llama' or 'Qwen'.  Falls back to current if ambiguous."""
    has_llama = bool(_LLAMA_RE.search(text))
    has_qwen  = bool(_QWEN_RE.search(text))

    if has_llama and not has_qwen:
        return "Llama"
    if has_qwen and not has_llama:
        return "Qwen"
    # Both mentioned or neither → keep current
    return current


# ---------------------------------------------------------------------------
# Conversation logger
# ---------------------------------------------------------------------------
LOG_FILE = "conversations.log"

def log_message(speaker: str, content: str):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "speaker":   speaker,
        "content":   content,
    }
    with open(LOG_FILE, "a") as f:
        json.dump(entry, f)
        f.write("\n")

def log_separator():
    with open(LOG_FILE, "a") as f:
        f.write("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Save LangGraph Mermaid diagram as PNG
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
# Build the LangGraph
# ---------------------------------------------------------------------------
def create_graph(llama_llm, llama_tok, qwen_llm, qwen_tok):

    # ---- node: get_user_input ------------------------------------------------
    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            log_separator()
            return {"route": "end"}

        if user_input.strip() == "":
            print("Empty input — please type something.")
            return {"route": "retry_input"}

        # Detect which model the user wants
        new_active = detect_target_model(user_input, state.get("active_model", "Llama"))

        log_message("Human", user_input)

        # Return only the NEW message — operator.add appends it to history
        return {
            "route":        "to_llm",
            "history":      [ChatMessage(speaker="Human", content=user_input)],
            "active_model": new_active,
        }

    # ---- router: after get_user_input -----------------------------------------
    def route_after_input(state: AgentState) -> str:
        r = state.get("route", "retry_input")
        if r == "end":
            return END
        if r == "retry_input":
            return "get_user_input"
        # Route to the active model's node
        return "call_llama" if state.get("active_model") == "Llama" else "call_qwen"

    # ---- shared LLM-call helper -----------------------------------------------
    def _call_model(state: AgentState, model_name: str, llm, tok) -> dict:
        history = state["history"]

        # Project history for this model and convert to prompt
        chat      = project_history(history, model_name)
        prompt    = chat_to_prompt(chat, tok)

        print(f"\n[Calling {model_name} with {len(history)} history messages …]")
        raw = llm.invoke(prompt)

        # Normalise
        if isinstance(raw, list) and raw:
            raw = raw[0]
        response = str(raw).strip()

        # --- POST-PROCESS: chop off any lines where the model starts
        #     role-playing as another speaker
        cleaned = _strip_other_speakers(response, model_name)

        if cleaned == "" and response != "":
            # The model's entire output was mis-attributed (started with
            # another speaker's name).  Fall back to raw output and warn.
            print(f"\n  [{model_name}] entire output was mis-attributed; using raw text")
            cleaned = response

        response = cleaned

        log_message(model_name, response)

        # Return only the NEW message — operator.add appends it to history
        return {"history": [ChatMessage(speaker=model_name, content=response)]}

    # ---- node: call_llama -----------------------------------------------------
    def call_llama(state: AgentState) -> dict:
        return _call_model(state, "Llama", llama_llm, llama_tok)

    # ---- node: call_qwen ------------------------------------------------------
    def call_qwen(state: AgentState) -> dict:
        return _call_model(state, "Qwen", qwen_llm, qwen_tok)

    # ---- node: print_last -----------------------------------------------------
    def print_last(state: AgentState) -> dict:
        last = state["history"][-1]
        print("\n" + "-" * 50)
        print(f"{last['speaker']}:")
        print("-" * 50)
        print(last["content"])
        return {}

    # ---- wire the graph -------------------------------------------------------
    g = StateGraph(AgentState)
    g.add_node("get_user_input", get_user_input)
    g.add_node("call_llama",     call_llama)
    g.add_node("call_qwen",      call_qwen)
    g.add_node("print_last",     print_last)

    g.add_edge(START, "get_user_input")

    g.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama":     "call_llama",
            "call_qwen":      "call_qwen",
            "get_user_input": "get_user_input",   # self-loop (empty input)
            END:              END,
        },
    )

    # Both model nodes feed into print_last → back to input
    g.add_edge("call_llama", "print_last")
    g.add_edge("call_qwen",  "print_last")
    g.add_edge("print_last", "get_user_input")

    return g.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" LangGraph Dual-LLM Agent  (Llama + Qwen, unified history)")
    print("=" * 60)

    # --- load both models ---------------------------------------------------
    llama_llm, llama_tok = load_model(
        "meta-llama/Llama-3.2-1B-Instruct", "Llama"
    )
    qwen_llm, qwen_tok = load_model(
        "Qwen/Qwen2.5-1.5B-Instruct", "Qwen"
    )

    # --- build & visualise --------------------------------------------------
    print("\nBuilding LangGraph …")
    graph = create_graph(llama_llm, llama_tok, qwen_llm, qwen_tok)
    print("Graph built!")
    save_graph_image(graph, filename="lg_graph.png")

    # --- start a fresh log session ------------------------------------------
    log_separator()
    with open(LOG_FILE, "a") as f:
        f.write(f"SESSION START  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_separator()

    # --- initial state ------------------------------------------------------
    initial_state: AgentState = {
        "history":      [],           # empty; system prompt is injected per-model at call time
        "active_model": "Llama",      # default first responder
        "route":        "retry_input",
    }

    print("\n Tip: mention 'Llama' or 'Qwen' in your message to switch models.")
    print("   Type 'quit' to exit.\n")

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()