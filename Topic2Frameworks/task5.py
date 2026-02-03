# langgraph_stateful_llama_agent.py
# A simple LangGraph agent that:
# - Maintains chat history using the Message API (system/user/assistant roles)
# - Never sends empty input to the LLM (self-loop on empty)
# - Uses ONLY Llama (Qwen disabled)
# - Saves a Mermaid-rendered PNG of the graph to lg_graph.png (requires grandalf)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from typing_extensions import Annotated
import operator

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# Device selection: CUDA > MPS > CPU
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
# State: messages persists across turns (Message API)
# `messages` will be merged by LangGraph using operator.add
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    route: str  # "end" | "retry_input" | "to_llm"


# ---------------------------------------------------------------------------
# Build Llama model + tokenizer (Qwen disabled)
# ---------------------------------------------------------------------------
def create_llama():
    device = get_device()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model downloads...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    # IMPORTANT:
    # - return_full_text=False prevents the pipeline from echoing your entire prompt
    # - max_length=None avoids the max_length/max_new_tokens warning
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
    print("Model loaded successfully!")
    return llm, tokenizer


# ---------------------------------------------------------------------------
# Convert Message list -> model prompt using the tokenizer chat template
# This prevents the model from roleplaying multiple turns.
# ---------------------------------------------------------------------------
def messages_to_prompt(messages: List[AnyMessage], tokenizer) -> str:
    chat = []
    for m in messages:
        if isinstance(m, SystemMessage):
            chat.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            chat.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            chat.append({"role": "assistant", "content": m.content})
        else:
            # Tool/function messages if you add them later
            chat.append({"role": "tool", "content": getattr(m, "content", str(m))})

    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Save LangGraph Mermaid diagram as PNG
# Requires: pip install grandalf
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
# Build LangGraph
# ---------------------------------------------------------------------------
def create_graph(llama_llm, llama_tokenizer):

    # -------------------------
    # Node: get_user_input
    # -------------------------
    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        # Quit
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {"route": "end"}

        # Empty input => self-loop (no model call)
        if user_input.strip() == "":
            print("Empty input — please type something.")
            return {"route": "retry_input"}

        # Normal input => add HumanMessage to history
        return {
            "route": "to_llm",
            "messages": [HumanMessage(content=user_input)],
        }

    # -------------------------
    # Router: 3-way branching out of get_user_input
    # -------------------------
    def route_after_input(state: AgentState) -> str:
        r = state.get("route", "retry_input")
        if r == "end":
            return END
        if r == "retry_input":
            return "get_user_input"
        return "call_llama"

    # -------------------------
    # Node: call_llama
    # Uses full chat history
    # -------------------------
    def call_llama(state: AgentState) -> dict:
        prompt = messages_to_prompt(state["messages"], llama_tokenizer)
        print("\nRunning Llama with chat history...")

        response_text = llama_llm.invoke(prompt)

        # Normalize return type
        if isinstance(response_text, list) and len(response_text) > 0:
            response_text = response_text[0]

        return {"messages": [AIMessage(content=str(response_text).strip())]}

    # -------------------------
    # Node: print_last
    # Print only the last assistant message
    # -------------------------
    def print_last(state: AgentState) -> dict:
        print("\n" + "-" * 50)
        print("Assistant:")
        print("-" * 50)

        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            print(last.content)
        else:
            print(str(last))

        return {}

    # Graph
    g = StateGraph(AgentState)
    g.add_node("get_user_input", get_user_input)
    g.add_node("call_llama", call_llama)
    g.add_node("print_last", print_last)

    g.add_edge(START, "get_user_input")

    g.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "get_user_input": "get_user_input",  # self-loop for empty input
            END: END,
        },
    )

    g.add_edge("call_llama", "print_last")
    g.add_edge("print_last", "get_user_input")

    return g.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("LangGraph Stateful Chat Agent (Llama only, Message API history)")
    print("=" * 50)

    llama_llm, llama_tokenizer = create_llama()

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, llama_tokenizer)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph, filename="lg_graph.png")

    # Initial state: system message seeds the conversation
    initial_state: AgentState = {
        "route": "retry_input",
        "messages": [
            SystemMessage(content=
                          "You are a helpful assistant. Keep responses concise and clear. "
    "You may remember and use facts the user tells you earlier in this conversation (e.g., their name). "
    "If asked for the user's name, answer with the name the user provided in this chat.")
        ],
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
