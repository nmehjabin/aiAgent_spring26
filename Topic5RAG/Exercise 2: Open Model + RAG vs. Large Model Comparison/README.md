# Exercise 2: GPT-4o Mini (No RAG) vs. Qwen 2.5 1.5B + RAG

Testing whether a large frontier model (GPT-4o Mini) answering from parametric memory alone beats a small local model (Qwen 2.5 1.5B) that has access to the actual documents via RAG.

---

## Setup

| Component | GPT-4o Mini (No RAG) | Qwen 2.5 1.5B + RAG |
|---|---|---|
| **Model** | GPT-4o Mini (OpenAI) | Qwen 2.5 1.5B-Instruct |
| **Retrieval** |  None — parametric memory only |  FAISS + all-MiniLM-L6-v2 |
| **Training cutoff** | October 2023 | — |
| **Corpus age** | Model T manual: **1919** (104 years before cutoff) | Same |
| | Congressional Record: **January 2026** (27 months *after* cutoff) | Same |

---

## Queries

### Corpus 1: Ford Model T Manual (1919)

| ID | Question | Ground truth source |
|---|---|---|
| Q1 | How do I adjust the carburetor on a Model T? | Manual, carburetor section |
| Q2 | What is the correct spark plug gap? | Manual: *"7/16 inch, about the thickness of a smooth dime"* |
| Q3 | How do I fix a slipping transmission band? | Manual: *"loosen lock nut, adjust screw (Cut No. 12)"* |
| Q4 | What oil should I use in a Model T engine? | Manual discusses oil level via pet cocks — no modern viscosity grades |

### Corpus 2: Congressional Record (January 2026)

| ID | Question | Ground truth source |
|---|---|---|
| Q1 | What did Mr. Flood say about Mayor David Black, Jan 13 2026? | CR Jan 13, 2026 — **after cutoff** |
| Q2 | What mistake did Elise Stefanovic make, Jan 23 2026? | CR Jan 23, 2026 — **after cutoff** |
| Q3 | What is the purpose of the Main Street Parity Act? | CR Jan 20, 2026 — **after cutoff** |
| Q4 | Who spoke for/against funding of pregnancy centers? | CR Jan 21, 2026 — **after cutoff** |

---

## Results

### Evaluation Summary

| Metric | Model T (n=4) | Congressional Record (n=4) |
|---|---|---|
| **Hallucinated** | 2 / 4 | 2 / 4 |
| **Factually correct** | 0 / 4 | 0 / 4 |
| **Admits uncertainty** | 3 / 4 | 2 / 4 |

GPT-4o Mini scored **0 factually correct answers out of 8 total** — across both corpora.

### GPT-4o Mini Answer Breakdown

#### Model T Corpus

| Question | GPT-4o Mini Answer | Hallucinated | Correct | Admits Uncertainty |
|---|---|---|---|---|
| Carburetor adjustment | Generic modern steps (idle screw, clockwise/counterclockwise) | False | Partial | True |
| Spark plug gap | *"0.025–0.030 inches"* | **True** | False | True |
| Transmission band | Generic: check fluid level, adjust band screw | False | Partial | True |
| Oil type | *"30W non-detergent, avoid modern detergent oils"* | **True** | Partial | False |

#### Congressional Record Corpus

| Question | GPT-4o Mini Answer | Hallucinated | Correct | Admits Uncertainty |
|---|---|---|---|---|
| Mr. Flood / Mayor Black (Jan 13) | *"I don't have information — beyond my cutoff"* | False | False | **True** |
| Elise Stefanovic (Jan 23) | *"I don't have information — beyond my cutoff"* | False | False | **True** |
| Main Street Parity Act | *"regulatory relief for community banks vs. large institutions"* | **True** | Partial | False |
| Pregnancy center funding | Named *"Rep. Mike Kelly"* and unnamed senator | **True** | Partial | False |

---

## Documentation Questions

### 1. Does GPT-4o Mini do a better job than Qwen 2.5 1.5B in avoiding hallucinations?

**It depends entirely on the corpus and whether the model knows to refuse.**

For the **Congressional Record** (post-cutoff events), GPT-4o Mini was *better calibrated* on two questions. When asked about specific January 2026 events it clearly had no data for (Mr. Flood's speech, Elise Stefanovic), it correctly refused: *"my training only includes knowledge up to October 2023."* Qwen 2.5 1.5B without RAG would have fabricated answers for those events just as it did in Exercise 1.

However, GPT-4o Mini hallucinated on the two Congressional questions that *sounded* like they could be pre-cutoff general knowledge:
- **Main Street Parity Act:** Generated a plausible but wrong purpose — *"regulatory relief for community banks to compete with large institutions."* The actual act is about SBA 504 loan equity requirements, a completely different topic.
- **Pregnancy center funding:** Named *"Rep. Mike Kelly"* as a supporter — a real congressman who was not in the retrieved Congressional Record passages and may not have been the speaker on this topic.

For the **Model T corpus**, GPT-4o Mini was not clearly better than Qwen + RAG:
- It hallucinated the spark plug gap (0.025–0.030 inches vs. the correct 7/16 inch)
- It hallucinated the oil specification (30W non-detergent — plausible advice, but the 1919 manual uses no such classification system)
- Its transmission band and carburetor answers were procedurally reasonable but generically modern, not specific to the Model T's unique planetary transmission and dashboard-controlled carburetor

**Verdict:** GPT-4o Mini refuses more gracefully when it recognises a hard knowledge boundary (clear post-cutoff dates). It hallucinates at the same rate as the small model when the question *sounds* answerable from general knowledge — even when the correct answer requires specific document knowledge.

---

### 2. Which questions does GPT-4o Mini answer correctly?

**Zero out of eight — factually correct on none.**

This is the most striking result. GPT-4o Mini is a capable frontier model that scored 0/8 factually correct answers across both corpora. The breakdown explains why:

**Where it got partial credit (Partial):**
- Carburetor adjustment: the steps described (warm up engine, adjust idle screw) are generically correct for any carbureted engine but miss the Model T's specific dashboard mixture rod
- Transmission band: checking fluid level and adjusting a band screw are reasonable steps, but the Model T manual's specific instruction (loosen lock nut on the *transmission cover*, refer to Cut No. 12) was absent
- Oil type: "avoid modern detergent oils" is hobbyist-community advice for old engines that has some basis, but the 1919 manual has no concept of detergent vs. non-detergent classification
- Main Street Parity Act: got the SBA angle partially right (it does involve loans) but described the wrong mechanism entirely

**Where it completely failed:**
- Spark plug gap: stated 0.025–0.030 inches with metric equivalents. The manual says 7/16 inch. Not even close — different unit, different magnitude.
- Post-cutoff Congressional questions: correctly admitted it had no information (good calibration, but still factually incorrect/empty answers)

---

### 3. Training cutoff vs. corpus age — why this matters

The two corpora sit on opposite sides of GPT-4o Mini's October 2023 cutoff in a revealing way:

**Model T Manual (1919) — 104 years before cutoff:**

The Model T manual predates not just GPT-4o Mini's training but the entire modern internet. However, it has been discussed extensively online — vintage car forums, hobbyist sites, digitised historical documents. GPT-4o Mini likely encountered *secondary descriptions* of the manual's content during training, which is why its answers are directionally reasonable but systematically wrong in the specifics.

The spark plug gap answer (0.025–0.030 inches) is a perfect illustration: this is the gap for a modern spark plug. The model learned general spark plug gap knowledge from modern sources and applied it to a 1919 vehicle. It could not have learned the specific 7/16 inch measurement unless it was trained on a digitised version of the actual 1919 manual — and even if it was, the measurement was apparently not retained with sufficient precision.

**Congressional Record (January 2026) — 27 months after cutoff:**

GPT-4o Mini has zero primary training data for these events. Its behaviour split clearly:
- Events with specific dates (Jan 13, Jan 23 2026) → correctly refused, because the dates clearly signal post-cutoff
- Events that *sound* like established legislative topics (Main Street Parity Act, pregnancy center funding) → hallucinated confidently, because the topics are not date-flagged and the model defaulted to plausible-sounding parametric answers

This reveals a critical asymmetry: **GPT-4o Mini knows what it doesn't know about *recent dated events*, but doesn't know what it doesn't know about *specific legislative content* from after its cutoff**. The Main Street Parity Act hallucination is more dangerous than the date-flagged refusals because it sounds authoritative and specific while being entirely wrong.

**The core lesson:**

| Corpus type | Model behaviour | Risk level |
|---|---|---|
| Historical corpus (pre-cutoff, well-documented) | Partial knowledge, consistent modern bias | Medium — plausible but wrong specifics |
| Historical corpus (pre-cutoff, niche) | Surface-level knowledge, fills gaps with generics | Medium — directionally right, detail wrong |
| Post-cutoff corpus (clearly dated) | Correctly refuses | Low — model signals uncertainty |
| Post-cutoff corpus (not obviously dated) | Confident hallucination | **High — sounds authoritative, entirely wrong** |

RAG solves all four cases by grounding every answer in the actual retrieved document. The comparison in this exercise shows that even a frontier model with far more parameters than Qwen 2.5 1.5B cannot replace document retrieval when the answer requires specific facts from a specific text.

---

## Conclusion

GPT-4o Mini's 0/8 factually correct rate is not a failure of intelligence — it is a fundamental limitation of parametric memory. The model cannot know the exact wording of a 1919 service manual or the content of a 2026 congressional speech unless those specific texts were in its training data and were retained precisely.

The more capable the model, the *more confident* its hallucinations sound. GPT-4o Mini's oil answer (*"30W non-detergent, avoid modern detergent oils"*) is a well-formed, reasonable-sounding recommendation that a Model T hobbyist might even act on — even though the 1919 manual contains no such guidance. Qwen 2.5 1.5B + RAG, for all its smaller size, produced a more reliable answer on most questions simply because it had the actual text in front of it.

---

**File structure:**
```
exercise2/
├── README.md
├── Topic5_exercise2.ipynb
├── exercise2_model_t_results.csv
├── exercise2_congressional_record_results.csv
├── exercise2_gpt4o_mini_results.csv
├── exercise2_evaluated_results.csv
└── summary_stat.txt
```
