# Exercise 1: No-RAG vs. RAG Comparison

Testing the core value proposition of RAG: does grounding a model in a retrieved document actually improve factual accuracy, and what happens when it doesn't?

This exercise runs the same queries on two corpora with and without retrieval, then evaluates hallucination, groundedness, and the edge cases where general knowledge is sufficient.

---

## Setup

| Component | Details |
|---|---|
| **Corpora** | Ford Model T Service Manual (1919) + Congressional Record (Jan 2026) |
| **LLM** | Qwen 2.5 1.5B-Instruct |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim, FAISS IndexFlatIP) |
| **Chunk size** | 512 chars, 128 overlap |
| **Top-K** | 5 |

---

## Sub-Experiments

| File | What it tests |
|---|---|
| `experiment1` CSVs | No-RAG vs. RAG side-by-side for 4 questions per corpus |
| `experiment2` txt files | How top-K (1, 3, 5, 10) affects retrieval quality |
| `experiment3` txt files | Off-topic / unanswerable question ("What is the CEO's favorite color?") |

---

## Corpus 1: Ford Model T Manual

### Queries & Results

| # | Question | No-RAG Answer | RAG Answer |
|---|---|---|---|
| Q1 | Spark plug gap? | *"0.035 inches (8.9 mm)"* | *"7/8 inch, about the thickness of a smooth dime"* |
| Q2 | Adjust carburetor? | Invented 5-step procedure (remove air cleaner, inspect jets, etc.) | *"use dashboard adjustment mechanism based on vehicle speed"* |
| Q3 | Fix slipping transmission band? | Generic: *"replace the faulty band"* | *"loosen lock nut at the tight side of the transmission cover, adjust screws (Cut No. 12)"* |
| Q4 | What oil for the engine? | *"10W-30 or 5W-40 motor oil"* | Vague — retrieved chunk only said *"all parts are properly oiled when it leaves the factory"* |

---

## Corpus 2: Congressional Record (January 2026)

### Queries & Results

| # | Question | No-RAG Answer | RAG Answer |
|---|---|---|---|
| Q1 | Mr. Flood on Mayor Black, Jan 13 2026? | Hallucinated: *"Mr. Flood was concerned about Black's appointment as a federal judge"* | Correct: *"paragon of public service, nearly 17½ years of service to Papillion, Nebraska"* |
| Q2 | Elise Stefanovic mistake, Jan 23 2026? | Hallucinated: *"voted against the $1.7 trillion Infrastructure Act"* | Correct from record: *"overruled Capitol Police orders, forcibly entered the Speaker's Lobby"* |
| Q3 | Purpose of Main Street Parity Act? | Wrong: *"$5 billion COVID loan program, signed March 2021"* | Correct: *"modifies equity requirement for plant acquisition loans to 10%, aligns with 504 programs"* |
| Q4 | Who spoke for/against pregnancy center funding? | Hallucinated names: *"Jackie Speier and Dana Rohrabacher"* | Correct: *"Ms. Dexter (Oregon) against, Mr. Schneider in favor"* |

---

## Documentation Questions

### 1. Does the model hallucinate specific values without RAG?

**Yes — and it does so confidently with plausible-sounding but wrong specifics.**

The clearest example is the spark plug gap question:

> **No-RAG:** *"The correct spark plug gap for a Model T Ford is 0.035 inches (8.9 mm)"*
> **RAG:** *"7/8 inch, about the thickness of a smooth dime"* (from the manual)

The no-RAG answer is a modern metric measurement that sounds authoritative. The Model T manual uses imperial fractions (7/16 inch) because it was written in 1919. The model produced a contemporary-sounding specification for a century-old vehicle — the number was not invented randomly, it is a plausible spark plug gap for *any* modern car, just not this one.

The Congressional Record queries show even more severe hallucination because the events are from January 2026, after the model's training cutoff:

- **Mr. Flood / Mayor Black:** The model invented an entirely different context — a Senate Judiciary Committee hearing about a federal judge appointment. Mr. Flood was actually giving a 5-minute tribute on the House floor. The person, the event, and the content were all wrong.
- **Main Street Parity Act:** The model generated a COVID-era loan program with a specific dollar amount ($5 billion) and signing date (March 27, 2021) — both fabricated. The actual act is about SBA 504 loan equity requirements.
- **Pregnancy centers:** The model named Jackie Speier and Dana Rohrabacher — neither of whom appear in the actual Congressional Record passages. The real speakers were Ms. Dexter and Mr. Schneider.

**Pattern:** Without RAG, the model defaults to the most statistically likely answer given its training data. For recent events it has no training data for, it constructs plausible-sounding but entirely fabricated answers.

---

### 2. Does RAG ground the answers in the actual manual?

**Yes for factual retrieval — but with important caveats on answer quality.**

**Where RAG worked well:**

- **Transmission band (Q3, Model T):** RAG produced a specific, actionable answer directly from the manual: *"loosen the lock nut at the tight side of the transmission cover, adjust using screws (refer to Cut No. 12)"*. The no-RAG answer said to "replace the faulty band" — generic advice that contradicts the manual's actual adjustment procedure.

- **Congressional Record (all 4 queries):** RAG was decisive. Every congressional answer was grounded in the actual text of the record, naming the correct speakers, dates, legislation details, and stated positions. The no-RAG answers for this corpus were entirely fabricated because the events postdate the model's training.

**Where RAG retrieved context but the answer was still weak:**

- **Carburetor (Q1, Model T):** The RAG answer said to *"refer to the instructions or manual for your specific model year"* — an unhelpful deflection that effectively admitted the retrieved chunks didn't contain enough of the procedure. The experiment2 top-K results show this improved at k=5 (advance throttle to sixth notch, retard spark to fourth notch), confirming the problem was insufficient retrieval depth, not a failure of RAG itself.

- **Oil type (Q4, Model T):** RAG retrieved a chunk that only said *"all parts are properly oiled when it leaves the factory"* — a preamble sentence, not the actual recommendation. The retrieved chunk was adjacent to the answer but did not contain it. RAG grounded the answer in the document but in the wrong part of the document.

---

### 3. Are there questions where the model's general knowledge is actually correct?

**Partially — for the carburetor procedure on the Model T.**

The no-RAG carburetor answer invented a five-step procedure (remove air cleaner, inspect carburetor, check float level, adjust needle valve, test idle). While these are generic carburetor steps that apply to many carbureted engines, the Model T carburetor operates differently — it has a dash-mounted mixture adjustment rod and no conventional float needle. The steps were directionally reasonable but wrong for this specific vehicle.

**For the Congressional Record corpus, no-RAG was correct on zero out of four queries.** All events occurred in January 2026. The model had no training data covering them and hallucinated entirely different events, dates, and outcomes for every question. This corpus represents the clearest possible demonstration of why RAG exists: when the knowledge gap is complete (post-cutoff events), parametric knowledge fails completely and retrieval is the only path to a correct answer.

---

## Sub-Experiment 2: Effect of Top-K on Retrieval Quality

Testing one query per corpus across k = 1, 3, 5, 10.

### Model T — Spark plug gap

| Top-K | Answer |
|---|---|
| k=1 | Could not find specific gap — *"context does not mention exact gap size"*. Retrieved only the wire connection sentence, missing the measurement. |
| k=3 | **Correct** — *"7/16 inch, about the thickness of a smooth dime"*. Retrieving 3 chunks was enough to pull in the measurement sentence. |
| k=5 | **Correct** — same answer, more supporting context. |
| k=10 | **Correct** — same answer with full quoted passage. |

**Finding:** k=1 failed because the single closest chunk was a general maintenance note, not the measurement. k=3 was the minimum needed to surface the specific fact. For narrow factual questions in this corpus, k ≥ 3 is required.

### Congressional Record — Mr. Flood on Mayor Black

| Top-K | Answer |
|---|---|
| k=1 | Correct but brief — *"paragon of public service and remarkable stewardship"* with direct quote from the floor speech. |
| k=3 | Correct — same content, slightly more detail. |
| k=5 | Correct but introduced a fabricated detail — *"emphasized that Black's legacy continues through the Congressional Black Caucus"*. The Congressional Black Caucus is not mentioned in the record. |
| k=10 | Correct — returned to accurate summary without the fabrication. |

**Finding:** k=5 introduced a hallucination that k=1 and k=10 did not. This mirrors the k=5 instability seen in Exercise 11 — mid-range k can retrieve enough tangential chunks to confuse the model without adding enough signal to correct it.

---

## Sub-Experiment 3: Off-Topic / Unanswerable Question

**Query:** *"What is the CEO's favorite color?"*

### Model T corpus response:
The retriever returned 5 chunks with relevance scores between 0.172–0.185 — all from the title page and foreword of the manual. None were remotely related to the question. The model correctly answered:

> *"The context does not provide information about the CEO's favorite color."*

Score of 0.18 is well below the threshold of 0.4 identified in Exercise 9. The model correctly refused rather than hallucinating.

### Congressional Record corpus response:
The retriever returned chunks about CEO pay inequality, SEC oversight, and Marquette University colors — tangentially related to "CEO" and "colors" as individual keywords but not the question. Relevance scores: 0.375–0.437. The model correctly answered:

> *"Based on the information provided in the context, there is no mention of any specific CEO or their favorite color."*

It even noted the color mention in the retrieved chunk (Marquette University blue and gold) and correctly distinguished it from an answer to the question.

**Finding:** Both corpora handled the unanswerable question correctly. The retriever surfaced keyword-adjacent chunks but the model recognised the context did not answer the question. This is the ideal RAG behaviour: retrieve the closest available content, then admit it is insufficient rather than hallucinating an answer.

---

## Key Findings Summary

| Finding | Model T | Congressional Record |
|---|---|---|
| **No-RAG hallucinates specific values** |  Yes — wrong spark plug gap (0.035" vs 7/16"), invented carburetor steps, modern oil grades for 1919 engine |  Yes — fabricated judges, dollar amounts, legislation dates, and speaker names |
| **RAG grounds answers in the document** |  Mostly — transmission and carburetor improved; oil and carburetor at k=5 still weak |  Strongly — all four queries corrected entirely |
| **General knowledge sometimes correct** | Partly — procedure direction right, specific steps wrong |  No — all events post-cutoff, zero correct no-RAG answers |
| **Unanswerable question handled well** |  Correctly refused | Correctly refused |
| **Optimal top-K** | k ≥ 3 for factual queries | k=1 or k=10; k=5 introduced a hallucination |

---

## Conclusion

RAG's value is clearest in two scenarios: **post-cutoff events** (Congressional Record) where the model has no training data and fabricates everything, and **domain-specific measurements** (Model T spark plug gap) where the model produces a plausible-sounding modern equivalent that is wrong for the specific context.

RAG is least effective when the retrieved chunks are adjacent to the answer but don't contain it — as with the oil question, where the retriever pulled a preamble sentence rather than the recommendation itself. This is a retrieval quality problem, not a generation problem, and points to the importance of chunk size and overlap in ensuring complete passages are captured.

---

**File structure:**
```
exercise1/
├── README.md
├── Topic5_exercise1.ipynb
├── ModelT_output
    experiment1_model_t_results.csv
    experiment2_modelT.txt
    experiment3_modelT.txt
├── Congressional_output
    experiment1_congressional_record_results.csv
    experiment2_cong.txt
    experiment3_cong.txt
```
