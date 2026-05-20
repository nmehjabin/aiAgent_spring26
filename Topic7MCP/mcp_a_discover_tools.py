import os, json, requests
from dotenv import load_dotenv
load_dotenv()

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

def parse_sse(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload and payload != "[DONE]":
                return json.loads(payload)
    raise ValueError(f"No data in SSE:\n{text[:200]}")

def print_tool(tool):
    name = tool.get("name", "unknown")
    desc = tool.get("description", "").strip().splitlines()[0]
    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    print(f"\nTool: {name}")
    print(f"  Description: {desc}")
    req, opt = [], []
    for p, info in props.items():
        t = info.get("type", "any")
        if "anyOf" in info:
            t = "/".join(x.get("type","any") for x in info["anyOf"] if "type" in x)
        (req if p in required else opt).append(f"{p} ({t})")
    if req: print(f"  Required: {', '.join(req)}")
    if opt: print(f"  Optional: {', '.join(opt)}")
    print("-" * 60)

payload = {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
print("=" * 60)
print("Fetching Asta MCP tool list...")
print("=" * 60)
resp = requests.post(MCP_URL, headers=headers, json=payload, timeout=15)
resp.raise_for_status()
tools = parse_sse(resp.text)["result"]["tools"]
print(f"\nFound {len(tools)} tools:\n")
for t in tools:
    print_tool(t)
