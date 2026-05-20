import os, sys, json, requests
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta
load_dotenv()

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

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

paper_id = sys.argv[1] if len(sys.argv) > 1 else "ARXIV:2210.03629"
print(f"\nBuilding citation neighborhood for: {paper_id}\n")

print("  [1/5] Fetching seed paper...")
seed = call_tool("get_paper", {"paper_id": paper_id, "fields": "title,abstract,year,authors,fieldsOfStudy,citationCount"})
print(f"        → {seed.get('title','')}")

print("  [2/5] Fetching references...")
ref_data = call_tool("get_references", {"paper_id": paper_id, "fields": "title,year,authors,citationCount,abstract", "limit": 50})
refs = [item.get("citedPaper", item) for item in ref_data.get("data", []) if item.get("citedPaper", item).get("title")]
refs.sort(key=lambda x: x.get("citationCount") or 0, reverse=True)
refs = refs[:5]

print("  [3/5] Fetching recent citations...")
cutoff = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
cit_data = call_tool("get_citations", {"paper_id": paper_id, "fields": "title,year,authors,abstract", "limit": 5, "publication_date_range": f"{cutoff}:"})
cits = [item.get("citingPaper", item) for item in cit_data.get("data", []) if item.get("citingPaper", item).get("title")][:5]

print("  [4/5] Fetching author highlights...")
highlights = {}
for author in seed.get("authors", [])[:5]:
    aid = author.get("authorId")
    name = author.get("name", "Unknown")
    if not aid: continue
    try:
        adata = call_tool("get_author_papers", {"author_id": aid, "fields": "title,year,citationCount", "limit": 20})
        papers = sorted(adata.get("data", []), key=lambda x: x.get("citationCount") or 0, reverse=True)
        highlights[name] = papers[:1]
    except:
        highlights[name] = []

print("  [5/5] Generating report with GPT...")
context = {
    "seed": {"title": seed.get("title"), "year": seed.get("year"), "abstract": (seed.get("abstract") or "")[:1200], "authors": [a.get("name") for a in seed.get("authors", [])], "citations": seed.get("citationCount")},
    "references": [{"title": p.get("title"), "year": p.get("year"), "authors": [a.get("name") for a in p.get("authors",[])[:3]], "citations": p.get("citationCount"), "abstract": (p.get("abstract") or "")[:200]} for p in refs],
    "recent_citations": [{"title": p.get("title"), "year": p.get("year"), "authors": [a.get("name") for a in p.get("authors",[])[:3]]} for p in cits],
    "author_highlights": {n: [{"title": p.get("title"), "year": p.get("year"), "citations": p.get("citationCount")} for p in ps] for n, ps in highlights.items()},
}

prompt = f"""Write a structured markdown research report based on this Semantic Scholar data:

{json.dumps(context, indent=2)}

Sections:
1. Title header (paper name + year)
2. **Overview** — one paragraph on the paper's contribution
3. **Foundational Works** — each reference with title, year, authors, citation count, one sentence on why it matters
4. **Recent Developments** — each citing paper with title, year, one sentence on how it extends the work
5. **Author Profiles** — each author's most notable other work
6. **Research Gaps** — 3-5 open problems inferred from the literature

Use only real titles from the data. Be concise."""

resp = client.chat.completions.create(model=LLM_MODEL, messages=[{"role":"user","content":prompt}], max_tokens=2000)
report = resp.choices[0].message.content

print("\n" + "="*60)
print(report)
print("="*60)

out = f"report_{paper_id.replace(':','_')}.md"
with open(out, "w") as f:
    f.write(report)
print(f"\n✅ Saved to {out}")
