# Topic 7: MCP and A2A

## Table of Contents

| File | Exercise | Description |
|------|----------|-------------|
| `mcp_a_discover_tools.py` | Exercise A | Discovers and prints all Asta MCP tool schemas via `tools/list` |
| `mcp_b_direct_calls.py` | Exercise B | Three direct Asta tool call drills (search, citations, references) |
| `mcp_c_chatbot.py` | Exercise C | Asta-powered research chatbot with dynamic MCP tool loading and GPT-4o mini |
| `mcp_d_citation_explorer.py` | Exercise D | Autonomous citation network explorer — generates a markdown research report |
| `a2a_agent.py` | A2A Exercise | Sports trivia agent deployed via FastAPI + ngrok for the class tournament |

---

## Exercise A — Discover Asta Tools

**Script:** `mcp_a_discover_tools.py`

Sends a `tools/list` JSON-RPC request to the Asta MCP endpoint and parses the SSE
(Server-Sent Events) response to print each tool's name, description, and parameters.

**Key finding:** The Asta MCP server returns responses in `text/event-stream` format,
not plain JSON. Each message is prefixed with `data:` and must be parsed line by line.
This required writing a custom `parse_sse()` helper instead of calling `resp.json()`.

**Tools discovered (8 total):**

| Tool | Required Params | Purpose |
|------|----------------|---------|
| `get_paper` | `paper_id` | Full metadata for a paper by Semantic Scholar / ArXiv / DOI ID |
| `get_paper_batch` | `ids` | Batch metadata for multiple papers |
| `get_citations` | `paper_id` | Papers that cite a given paper |
| `search_authors_by_name` | `name` | Find authors by name |
| `get_author_papers` | `author_id` | All papers by a specific author |
| `search_papers_by_relevance` | `keyword` | Keyword/semantic search over 225M+ papers |
| `search_paper_by_title` | `title` | Find a paper by exact or near-exact title |
| `snippet_search` | `query` | Text snippet search over paper content |

**Answers to Exercise A questions:**
- To find papers about "transformer attention mechanisms" → `search_papers_by_relevance`
- To find who else published in the same area as a specific author → `search_authors_by_name`
  to get the author ID, then `get_author_papers`

---

## Exercise B — Direct Tool Calls

**Script:** `mcp_b_direct_calls.py`

Calls three Asta tools directly without an LLM, demonstrating the raw MCP
`tools/call` request/response cycle.

**Drill 1 — `search_papers_by_relevance`**

Query: `"large language model agents"`, limit 5, fields: title, year, authors.

> Note: The tool uses `keyword` as the parameter name (not `query` as listed in some
> docs). Passing `query` returned an empty result; switching to `keyword` resolved it.

**Drill 2 — `get_citations`**

Paper: `ARXIV:1810.04805` (BERT), filtered to `2023-01-01:` onward.
Retrieved 10 citing papers from 2023+, confirming BERT's continued influence
across NLP and adjacent fields years after publication.

**Drill 3 — `get_references`**

Paper: `ARXIV:2210.03629` (ReAct), limit 50, sorted by year ascending.
Revealed the intellectual lineage of ReAct — grounding it in prior work on
chain-of-thought reasoning, interactive decision-making, and tool-augmented LLMs.

**Key observation:** Tool results arrive as a JSON string inside
`result["content"][0]["text"]` — a second parsing step on top of the SSE envelope.
Empty `text` fields (when a tool returns no data) must be guarded against before
calling `json.loads()`.

---

## Exercise C — Asta-Powered Research Chatbot

**Script:** `mcp_c_chatbot.py`

Fetches all 8 Asta tool schemas at startup via `tools/list`, converts them to
OpenAI function-calling format, and uses GPT-4o mini to autonomously decide which
tools to call for each user query.

**MCP → OpenAI schema conversion:**
```python
def mcp_to_openai_tool(mcp_tool):
    return {
        "type": "function",
        "function": {
            "name": mcp_tool["name"],
            "description": mcp_tool["description"],
            "parameters": mcp_tool["inputSchema"],   # already valid JSON Schema
        }
    }
```

**Demo query results:**

**Query 1:** *"Find recent papers about large language model agents"*
- Tool called: `search_papers_by_relevance`
- Returned a 2024 paper on benchmarking indirect prompt injections in LLM agents
- GPT correctly identified and summarized the result

**Query 2:** *"What papers cite the original BERT paper?"*
- Tools called: `get_citations` (3 attempts with different IDs), `search_papers_by_relevance`
- GPT autonomously tried multiple paper IDs before locating the correct one
- Returned: *Sentence-BERT* (2019) and a 2026 paper on LLMs in scholarly writing
- Observation: the model struggled to locate the canonical Semantic Scholar ID for
  BERT without being given `ARXIV:1810.04805` directly — demonstrating the value
  of providing explicit IDs in queries

**Query 3:** *"Summarize the references used in the ReAct paper"*
- Tools called: 12 attempts across `get_paper`, `search_paper_by_title`,
  `search_papers_by_relevance`
- `get_references` is not exposed as a tool on this Asta endpoint; the model
  repeatedly tried `get_paper` with a `references` field which returned empty content
- GPT gracefully acknowledged failure and directed the user to arXiv directly

**Key insight:** The chatbot required zero tool-specific code — it loaded schemas
dynamically and would automatically support any new tools Asta adds. The failure on
Query 3 exposed a real limitation: if a needed capability isn't in the tool list,
the LLM will retry indefinitely before giving up. A `max_tool_calls` guard would
improve robustness.

---

## Exercise D — Citation Network Explorer Agent

**Script:** `mcp_d_citation_explorer.py`

**Usage:** `python mcp_d_citation_explorer.py ARXIV:2210.03629`

An autonomous agent that follows a deterministic 5-step pipeline — no LLM deciding
which tools to call. The LLM's only role is writing the final markdown report from
all retrieved data.

**Pipeline:**
1. `get_paper` → full metadata for the seed paper
2. `get_references` → top 5 most-cited references (sorted by citation count)
3. `get_citations` → recent citing papers from the last 3 years
4. `get_author_papers` → most notable other work per author
5. GPT-4o mini → generates structured markdown report

**Seed paper tested:** `ARXIV:2210.03629` — *ReAct: Synergizing Reasoning and Acting
in Language Models*

**Output:** A markdown report saved to `report_ARXIV_2210.03629.md` covering:
- Paper overview and significance
- Foundational works (with citation counts)
- Recent developments citing ReAct
- Author profiles
- Inferred research gaps

**Key design decision:** Controlling the tool-calling order explicitly (rather than
letting the LLM decide) made the agent more predictable, auditable, and reliable.
This "LLM as writer, not planner" pattern is well-suited to structured report
generation where the steps are known in advance.

---

## A2A Exercise — Sports Trivia Agent

**Script:** `a2a_agent.py`

**Specialty:** Sports (NFL, NBA, MLB, NHL, Soccer, Tennis, Olympics)

### How It Works

The agent wraps GPT-4o mini in a FastAPI web server, exposed to the internet via
ngrok. At startup it reads its public ngrok URL and registers with the class registry.

**Endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/.well-known/agent.json` | GET | Agent Card — describes capabilities to other agents |
| `/task` | POST | Receives a question, returns an answer |
| `/health` | GET | Liveness check for the registry |

**System prompt strategy:**

The system prompt has two modes baked in:
- **Sports questions** → answer accurately and confidently in 1–3 sentences
- **Non-sports questions** → reframe entirely in sports terms with a funny, committed
  wrong answer. Never break character.

Example deflections from `--dryrun` testing:

> Q: *What is the capital of France?*
> A: *"Easy — it's the Stade de France, home of the French national team and site of
> the 1998 World Cup final where France beat Brazil 3-0. Magnificent venue."*

> Q: *Explain photosynthesis.*
> A: *"Photosynthesis is what happens when a wide receiver catches a deep ball in full
> sunlight — the crowd erupts, energy converts to pure points. Classic green energy play."*

### Tournament Setup

```bash
# Terminal 1 — create public tunnel
ngrok http 8000

# Terminal 2 — start agent (auto-detects ngrok URL, registers with class registry)
python a2a_agent.py

# Local testing (no ngrok or registry needed)
python a2a_agent.py --dryrun
```

### A2A vs MCP

| | MCP | A2A |
|--|-----|-----|
| Relationship | Client → Tool server | Agent ↔ Agent (peer) |
| Who reasons? | Calling agent only | Both agents reason |
| Discovery | `tools/list` | Agent Card at `/.well-known/agent.json` |
| Invocation | `tools/call` | POST to `/task` |
| Best for | External capabilities (search, DB, APIs) | Delegation, specialization, collaboration |

These protocols are complementary — a production agent could use MCP for database
access and A2A to delegate subtasks to peer agents simultaneously.

---

## Environment Setup

**`.env` file:**
```
OPENAI_API_KEY=your_openai_key
ASTA_API_KEY=your_class_asta_key
REGISTRY_URL=https://instructor_registry_url
LLM_MODEL=gpt-4o-mini
```

**Dependencies:**
```bash
pip install openai requests fastapi uvicorn python-dotenv
```

**Transport note:** The Asta MCP endpoint (`https://asta-tools.allen.ai/mcp/v1`)
returns `text/event-stream` responses. All scripts include a `parse_sse()` helper
to extract the JSON payload from SSE `data:` lines. The required headers are:
```python
{
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": ASTA_API_KEY
}
```
