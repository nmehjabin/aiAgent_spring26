# langgraph_simple_agent.py
# Parallel Llama + Qwen inference with LangGraph routing.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor

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
    llama_response: str
    qwen_response: str

# ---------------------------------------------------------------------------
# LLM builders
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
# Graph
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
            return {
                "user_input": user_input,
                "should_exit": True,
                "route": "end",
            }

        if user_input.strip() == "":
            print("Empty input — please type something.")
            return {
                "user_input": "",
                "should_exit": False,
                "route": "retry_input",
            }

        return {
            "user_input": user_input,
            "should_exit": False,
            "route": "to_dispatch",
        }

    # -------------------------
    # Router after input
    # -------------------------
    def route_after_input(state: AgentState) -> str:
        if state.get("route") == "end":
            return END
        if state.get("route") == "retry_input":
            return "get_user_input"
        return "dispatch"

    # -------------------------
    # Node: dispatch (runs both models in parallel)
    # -------------------------
    def dispatch(state: AgentState) -> dict:
        user_input = state["user_input"]

        # You can keep your simple prompt format:
        prompt = f"User: {user_input}\nAssistant:"

        print("\nRunning Llama + Qwen in parallel...")

        def run_llama():
            return llama_llm.invoke(prompt)

        def run_qwen():
            return qwen_llm.invoke(prompt)

        # Run both at the same time
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(run_llama)
            f2 = ex.submit(run_qwen)
            llama_out = f1.result()
            qwen_out  = f2.result()

        return {
            "llama_response": llama_out,
            "qwen_response": qwen_out,
        }

    # -------------------------
    # Node: print both
    # -------------------------
    def print_both(state: AgentState) -> dict:
        print("\n" + "-" * 50)
        print("LLM Responses:")
        print("-" * 50)

        print("\n[Llama-3.2-1B-Instruct]")
        print(state["llama_response"])

        print("\n[Qwen2.5-1.5B-Instruct]")
        print(state["qwen_response"])

        return {}

    # Build graph
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("dispatch", dispatch)
    graph_builder.add_node("print_both", print_both)

    graph_builder.add_edge(START, "get_user_input")

    # 3-way branch out of get_user_input (including self-loop)
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "dispatch": "dispatch",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    # After dispatch, print both, then loop back for next input
    graph_builder.add_edge("dispatch", "print_both")
    graph_builder.add_edge("print_both", "get_user_input")

    return graph_builder.compile()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("LangGraph Parallel Agent: Llama-3.2-1B + Qwen2.5")
    print("=" * 50)
    print()

    llama_llm, qwen_llm = create_models()

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "route": "retry_input",
        "llama_response": "",
        "qwen_response": "",
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
