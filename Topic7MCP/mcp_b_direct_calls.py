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
            p = line[len("data:"):].strip()
            if p and p != "[DONE]":
                return json.loads(p)
    raise ValueError(f"No data in SSE:\n{text[:200]}")

def call_tool(name, args):
    payload = {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":name,"arguments":args}}
    resp = requests.post(MCP_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = parse_sse(resp.text)
    raw = data["result"]["content"][0]["text"]
    if not raw.strip():
        return {}
    return json.loads(raw)

# Drill 1
print("\n" + "="*60)
print("Drill 1: Search 'large language model agents'")
print("="*60)
data = call_tool("search_papers_by_relevance", {"query":"large language model agents","fields":"title,year,authors","limit":5})
for i,p in enumerate(data.get("data",[]),1):
    authors = ", ".join(a.get("name","") for a in p.get("authors",[])[:3])
    print(f"  {i}. [{p.get('year','N/A')}] {p.get('title','')}")
    print(f"     {authors}")

# Drill 2
print("\n" + "="*60)
print("Drill 2: Citations of BERT from 2023+")
print("="*60)
data = call_tool("get_citations", {"paper_id":"ARXIV:1810.04805","fields":"title,year","limit":10,"publication_date_range":"2023-01-01:"})
papers = data.get("data",[])
print(f"Found {len(papers)} citing papers. First 5:")
for i,item in enumerate(papers[:5],1):
    p = item.get("citingPaper", item)
    print(f"  {i}. [{p.get('year','N/A')}] {p.get('title','')}")

# Drill 3
print("\n" + "="*60)
print("Drill 3: References of ReAct, sorted by year")
print("="*60)
data = call_tool("get_references", {"paper_id":"ARXIV:2210.03629","fields":"title,year","limit":50})
refs = []
for item in data.get("data",[]):
    p = item.get("citedPaper", item)
    if p.get("title"):
        refs.append((p.get("year") or 0, p.get("title"), p.get("year")))
refs.sort()
for _, title, yr in refs:
    print(f"  [{yr or 'N/A'}] {title}")

print("\nDone.")
