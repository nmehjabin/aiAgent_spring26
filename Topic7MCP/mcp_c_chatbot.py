"""
Exercise C: Asta-Powered Research Chatbot
==========================================
Dynamically fetches Asta tool schemas from MCP at startup, then uses
GPT-4o mini to decide which tools to call.

Runs a set of demo queries automatically (no interactive input needed).
To add your own queries, edit the DEMO_QUERIES list at the bottom.
"""

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

asta_headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a research assistant with access to the Semantic Scholar \
academic database (225+ million papers) via Asta tools. Help users discover papers, \
trace citation networks, explore authors, and understand the intellectual landscape \
of any research topic. Synthesize results clearly — include titles, authors, and years."""


# =============================================================================
# SSE parser
# =============================================================================

def parse_sse(text: str) -> dict:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload and payload != "[DONE]":
                return json.loads(payload)
    raise ValueError(f"No data in SSE response:\n{text[:300]}")


# =============================================================================
# MCP → OpenAI schema conversion
# =============================================================================

def mcp_to_openai_tool(mcp_tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": mcp_tool["name"],
            "description": mcp_tool["description"],
            "parameters": mcp_tool["inputSchema"],
        },
    }


def get_asta_tools() -> list:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    resp = requests.post(MCP_URL, headers=asta_headers, json=payload, timeout=15)
    resp.raise_for_status()
    mcp_tools = parse_sse(resp.text)["result"]["tools"]
    return [mcp_to_openai_tool(t) for t in mcp_tools]


# =============================================================================
# Tool execution
# =============================================================================

def call_asta_tool(name: str, arguments: dict) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    try:
        resp = requests.post(MCP_URL, headers=asta_headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = parse_sse(resp.text)
        content = data["result"]["content"][0]["text"]
        # Truncate large results to stay within context window
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "data" in parsed:
            parsed["data"] = parsed["data"][:8]
            content = json.dumps(parsed)
        return content
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Chat — one full turn with automatic tool call handling
# =============================================================================

def chat(user_message: str, messages: list, tools: list) -> str:
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            reply = msg.content or ""
            messages.append({"role": "assistant", "content": reply})
            return reply

        messages.append(msg)

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"    🔧 Tool: {name}  args: {json.dumps(args)}")
            result = call_asta_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


# =============================================================================
# Demo queries
# =============================================================================

DEMO_QUERIES = [
    "Find recent papers about large language model agents.",
    "What papers cite the original BERT paper?",
    "Summarize the references used in the ReAct paper (ARXIV:2210.03629).",
]


def main():
    print("=" * 60)
    print("Asta Research Chatbot — loading tools...")
    print("=" * 60)

    tools = get_asta_tools()
    print(f"Loaded {len(tools)} tools from MCP.\n")

    for query in DEMO_QUERIES:
        print(f"\n{'=' * 60}")
        print(f"You: {query}")
        print(f"{'=' * 60}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        answer = chat(query, messages, tools)

        print(f"\nAssistant:\n{answer}\n")


if __name__ == "__main__":
    main()