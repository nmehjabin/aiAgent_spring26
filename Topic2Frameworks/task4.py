import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    route: str
    model_route: str      # "llama" | "qwen"
    llm_response: str

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_hf_llm(model_id: str, device: str) -> HuggingFacePipeline:
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=pipe)

def create_models():
    device = get_device()
    llama_id = "meta-llama/Llama-3.2-1B-Instruct"
    qwen_id  = "Qwen/Qwen2.5-1.5B-Instruct"

    print("This may take a moment on first run as models download...")
    llama_llm = build_hf_llm(llama_id, device)
    qwen_llm  = build_hf_llm(qwen_id, device)

    print("Both models loaded successfully!")
    return llama_llm, qwen_llm

# ---------------------------------------------------------------------------
# Graph image save (Mermaid -> PNG)
# ---------------------------------------------------------------------------
def save_graph_image(graph, filename="lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    Requires: pip install grandalf
    """
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need: pip install grandalf")

# ---------------------------------------------------------------------------
# Create LangGraph
# ---------------------------------------------------------------------------
def create_graph(llama_llm, qwen_llm):

    # -------------------------
    # Node: get_user_input
    # -------------------------
    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True, "route": "end"}

        if user_input.strip() == "":
            print("Empty input — please type something.")
            return {"user_input": "", "should_exit": False, "route": "retry_input"}

        return {"user_input": user_input, "should_exit": False, "route": "to_model_router"}

    # -------------------------
    # Router after input (3-way)
    # -------------------------
    def route_after_input(state: AgentState) -> str:
        if state.get("route") == "end":
            return END
        if state.get("route") == "retry_input":
            return "get_user_input"
        return "model_router"

    # -------------------------
    # Node: model_router
    # If input starts with "Hey Qwen", route to Qwen else Llama
    # -------------------------
    def model_router(state: AgentState) -> dict:
        text = state["user_input"]
        if text.lower().startswith("hey qwen"):
            return {"model_route": "qwen"}
        return {"model_route": "llama"}

    def route_to_model(state: AgentState) -> str:
        return "call_qwen" if state.get("model_route") == "qwen" else "call_llama"

    # -------------------------
    # Node: call_llama
    # -------------------------
    def call_llama(state: AgentState) -> dict:
        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"
        print("\nRunning Llama...")
        response = llama_llm.invoke(prompt)
        return {"llm_response": response}

    # -------------------------
    # Node: call_qwen
    # -------------------------
    def call_qwen(state: AgentState) -> dict:
        user_input = state["user_input"]

        # Strip "Hey Qwen" so Qwen sees just the request (optional, but nicer)
        cleaned = user_input
        if cleaned.lower().startswith("hey qwen"):
            cleaned = cleaned[len("hey qwen"):].lstrip(" ,:;-")

        prompt = f"User: {cleaned}\nAssistant:"
        print("\nRunning Qwen...")
        response = qwen_llm.invoke(prompt)
        return {"llm_response": response}

    # -------------------------
    # Node: print_response
    # -------------------------
    def print_response(state: AgentState) -> dict:
        print("\n" + "-" * 50)
        print(f"Model used: {state.get('model_route', '?')}")
        print("Response:")
        print("-" * 50)
        print(state["llm_response"])
        return {}

    # Build graph
    g = StateGraph(AgentState)

    g.add_node("get_user_input", get_user_input)
    g.add_node("model_router", model_router)
    g.add_node("call_llama", call_llama)
    g.add_node("call_qwen", call_qwen)
    g.add_node("print_response", print_response)

    g.add_edge(START, "get_user_input")

    # 3-way out of get_user_input (includes self-loop)
    g.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "model_router": "model_router",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    # Route to chosen model
    g.add_conditional_edges(
        "model_router",
        route_to_model,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
        },
    )

    # Both model nodes go to printer, then loop
    g.add_edge("call_llama", "print_response")
    g.add_edge("call_qwen", "print_response")
    g.add_edge("print_response", "get_user_input")

    return g.compile()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("LangGraph Agent: 'Hey Qwen' routes to Qwen, otherwise Llama")
    print("=" * 50)

    llama_llm, qwen_llm = create_models()

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    # Save graph visualization BEFORE execution
    print("\nSaving graph visualization...")
    save_graph_image(graph, filename="lg_graph.png")

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "route": "retry_input",
        "model_route": "llama",
        "llm_response": "",
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
