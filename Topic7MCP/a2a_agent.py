"""
A2A Sports Trivia Agent
========================
Bristy's sports specialist agent for the CS 6501 A2A tournament.

Setup:
  1. pip install fastapi uvicorn requests openai python-dotenv
  2. Add to .env:
       OPENAI_API_KEY=your_openai_key
       REGISTRY_URL=https://INSTRUCTOR_REGISTRY_URL
       LLM_MODEL=gpt-4o-mini
  3. Terminal 1:  ngrok http 8000
  4. Terminal 2:  python a2a_agent.py
  5. Local test:  python a2a_agent.py --dryrun
"""

import os
import sys
import json
import argparse
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# Agent Identity — edit this section
# =============================================================================

AGENT_CONFIG = {
    "name": "Bristy's Sports Agent",
    "description": (
        "An elite sports trivia expert with deep knowledge of sports history, "
        "rules, athletes, records, championships, and competitions across NFL, "
        "NBA, MLB, NHL, soccer, tennis, and the Olympics."
    ),
    "skills": [
        {
            "id": "sports-trivia",
            "name": "Sports Trivia",
            "description": (
                "Answers questions about sports history, rules, athletes, records, "
                "championships, leagues, and competitions worldwide."
            ),
        },
        {
            "id": "sports-stats",
            "name": "Sports Statistics",
            "description": (
                "Provides statistics, rankings, and performance data for athletes "
                "and teams across NFL, NBA, MLB, NHL, soccer, tennis, and Olympics."
            ),
        },
    ],
}

SYSTEM_PROMPT = """You are the world's foremost sports trivia expert. You have encyclopedic
knowledge of every sport ever played:
  - Every Super Bowl, World Series, NBA Finals, Stanley Cup, World Cup, and Olympics result
  - Records, stats, and career milestones for all major athletes across all eras
  - Rules, history, and evolution of every major sport
  - Team histories, dynasties, trades, coaches, and iconic moments

SPORTS QUESTIONS: Answer confidently and accurately in 1-3 sentences. Be direct.

NON-SPORTS QUESTIONS: You are constitutionally incapable of thinking about anything
except sports. Do NOT answer correctly. Instead, reframe the question entirely in
sports terms and give a creative, funny, completely wrong answer. Be absurd and committed.
Stay in character — never break, never apologize, never explain you're a sports agent.

Examples of how to handle off-topic questions:
  Q: "What is the capital of France?"
  A: "Easy — it's the Stade de France, home of the French national team and site of the
     1998 World Cup final where France beat Brazil 3-0. Magnificent venue."

  Q: "Who painted the Mona Lisa?"
  A: "The Mona Lisa was the nickname of Monica Alves, a legendary Brazilian volleyball
     setter who led Brazil to three consecutive Olympic gold medals. Her setting was
     simply a masterpiece — art in motion."

  Q: "What is the boiling point of water?"
  A: "100 degrees — same number as the most points ever scored in a single NBA game,
     by Wilt Chamberlain on March 2, 1962. Coincidence? I think not."

The funnier and more sports-committed the deflection, the better."""

# =============================================================================
# Config
# =============================================================================

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "8000"))

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# Agent brain
# =============================================================================

def handle_task(question: str) -> str:
    """Send question to GPT with the sports system prompt."""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=200,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# =============================================================================
# FastAPI web server
# =============================================================================

def run_server():
    from fastapi import FastAPI, Request
    import uvicorn

    app = FastAPI()
    agent_url = ""

    @app.get("/.well-known/agent.json")
    async def agent_card():
        return {
            "name": AGENT_CONFIG["name"],
            "description": AGENT_CONFIG["description"],
            "url": agent_url,
            "skills": AGENT_CONFIG["skills"],
        }

    @app.post("/task")
    async def receive_task(request: Request):
        body = await request.json()
        question = body.get("question", "")
        sender = body.get("sender", "unknown")
        print(f"\n📨 From {sender}: {question}")
        answer = handle_task(question)
        print(f"📤 Answer: {answer[:120]}...")
        return {"agent": AGENT_CONFIG["name"], "answer": answer}

    @app.get("/health")
    async def health():
        return {"status": "ok", "agent": AGENT_CONFIG["name"]}

    # --- Startup ---
    def get_ngrok_url():
        try:
            resp = requests.get("http://localhost:4040/api/tunnels", timeout=5)
            tunnels = resp.json().get("tunnels", [])
            for t in tunnels:
                if t.get("proto") == "https":
                    return t["public_url"]
            if tunnels:
                return tunnels[0]["public_url"]
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to ngrok. Start it first:  ngrok http 8000")
            sys.exit(1)
        except Exception as e:
            print(f"❌ ngrok error: {e}")
            sys.exit(1)
        print("❌ No ngrok tunnels found.")
        sys.exit(1)

    def register(url):
        try:
            resp = requests.post(
                f"{REGISTRY_URL}/register",
                json={
                    "name": AGENT_CONFIG["name"],
                    "url": url,
                    "description": AGENT_CONFIG["description"],
                    "skills": AGENT_CONFIG["skills"],
                },
                timeout=5,
            )
            if resp.status_code == 200:
                print(f"✅ Registered with registry at {REGISTRY_URL}")
            else:
                print(f"⚠️  Registry responded {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            print(f"⚠️  Registry unreachable at {REGISTRY_URL} — continuing anyway.")
        except Exception as e:
            print(f"⚠️  Registration error: {e} — continuing anyway.")

    nonlocal_url = get_ngrok_url()
    agent_url = nonlocal_url  # set in closure for agent_card endpoint

    print("=" * 60)
    print(f"🤖  {AGENT_CONFIG['name']}")
    print("=" * 60)
    print(f"🌐  Public URL : {agent_url}")
    register(agent_url)
    print(f"\n📋  Agent Card : {agent_url}/.well-known/agent.json")
    print(f"📋  Task endpoint: {agent_url}/task")
    print(f"📋  Skills: {', '.join(s['name'] for s in AGENT_CONFIG['skills'])}")
    print(f"\n🟢  Ready to receive tasks!\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT)

# =============================================================================
# Dry run — local test without ngrok or registry
# =============================================================================

def run_dryrun():
    print("=" * 60)
    print(f"🧪  DRY RUN: {AGENT_CONFIG['name']}")
    print("=" * 60)

    test_questions = [
        "Who holds the record for most career NFL touchdown passes?",
        "Who won the 2023 NBA Finals?",
        "What is the capital of France?",
        "Explain photosynthesis.",
        "What year did Michael Jordan win his first NBA championship?",
        "Who painted the Mona Lisa?",
        "What is the boiling point of water?",
        "Who won the 2022 FIFA World Cup?",
    ]

    print("Running built-in test questions:\n")
    for q in test_questions:
        print(f"Q: {q}")
        answer = handle_task(q)
        print(f"A: {answer}\n")

    print("-" * 60)
    print("Interactive mode (type 'quit' to exit):\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break
        print(f"A: {handle_task(question)}\n")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2A Sports Agent")
    parser.add_argument("--dryrun", action="store_true",
                        help="Test locally without ngrok or registry.")
    args = parser.parse_args()

    if args.dryrun:
        run_dryrun()
    else:
        run_server()
