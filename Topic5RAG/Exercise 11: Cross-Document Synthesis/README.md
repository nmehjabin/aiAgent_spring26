
# Exercise 11: Cross-Document Synthesis

Testing whether a RAG pipeline can **combine information scattered across multiple chunks** into a single coherent answer, the hardest retrieval task. A question answerable from one passage is easy; a question that requires finding, merging, and reasoning over many passages is where most RAG systems break down.

## File Structure

exercise11/
├── README.md          #results and question answer
├── Topic5_exercise11.ipynb
├── exercise11_heatmap.png
├── exercise11_scores_vs_k.png
├── exercise11_source_diversity.png
├── exercise11_raw_answers.csv
└── exercise11_evaluated_results.csv


## Set up

| Component | Details |
|---|---|
| **Corpus** | Ford Model T Service Manual (1919) |
| **LLM** | Qwen 2.5 1.5B-Instruct |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim, FAISS IndexFlatIP) |
| **Chunk size** | 512 chars, 128 overlap (fixed) |
| **Variable** | Top-K = 3, 5, 10 |
| **Evaluation** | GPT-4o Mini judge, scores 1-5 on 4 dimensions |


## Synthesis Queries

Each query was chosen because the answer **cannot be found in a single chunk** — it requires the retriever to surface multiple relevant passages and the model to merge them.

| ID | Type | Query | Why it's hard |
|---|---|---|---|
| Q1 | Enumeration | What are ALL the maintenance tasks mentioned in the manual? | Maintenance tasks are scattered across many sections |
| Q2 | Comparison | Compare the procedure for adjusting the transmission bands vs. the carburetor. | Two separate procedures in different chapters must be compared |
| Q3 | Aggregation | What tools and supplies does the manual say I need to maintain a Model T? | Tools mentioned in scattered one-line mentions throughout |
| Q4 | Safety summary | Summarize all safety warnings and cautions mentioned in the manual. | Safety warnings distributed across the whole document |
| Q5 | System overview | Explain how the electrical system and ignition system work together. | Requires combining two separate technical sections into one explanation |



## Results 

### Score Heatmap - Query x Top-k
The heatmap below shows all four evaluation dimensions across every query and k value. Green = good (5), red = poor (1).

<img width="2366" height="622" alt="exercise11_heatmap" src="https://github.com/user-attachments/assets/92f54014-1703-4227-8df7-6bb64d79d707" />



> **Key patterns to notice:**
> - Q1 collapses across *all four dimensions* at k=5 — the single worst result in the experiment
> - Q3 is a flat orange row (2.0) across every k and every dimension — retrieval quantity never fixed this query
> - Q5 is the strongest query, scoring 5.0 on three of four dimensions at k=3 and k=10

### Scores vs. Top-K

How mean scores change as k increases from 3 → 5 → 10.


<img width="1335" height="732" alt="image" src="https://github.com/user-attachments/assets/4c92ad9a-fe3b-4106-bee0-47e0f4e3907e" />


> k=5 is frequently the **worst** setting, not a safe middle ground. k=3 forces the model to work with only its strongest matches; k=10 adds enough coverage to recover. k=5 introduces ambiguity without yet resolving it.


### Source Diversity vs. Top-K

Whether increasing k actually pulled in chunks from more source files.

<img width="1185" height="585" alt="image" src="https://github.com/user-attachments/assets/5baacba3-6d2f-461d-9d66-9ab22c2c07fe" />


> For Q3 (aggregation), source diversity plateaued early — increasing k returned more chunks from the *same* sections rather than new ones, explaining why completeness never improved past 2.0.



## Key Findings

### 1. Can the model successfully combine information from multiple chunks?
- **Q5 (system overview)** and **Q2 (comparison)** synthesised well — Q5 scored 5.0 at k=3 and k=10; Q2 held at 4.0 across all k values
-  **Q3 (aggregation)** consistently scored 2.0 on synthesis quality at every k — the model quoted individual chunks rather than merging them
-  **Q1 (enumeration)** was unstable — 4.0 at k=3, collapsed to 2.0 at k=5, recovered to 4.0 at k=10

### 2. Does it miss information that wasn't retrieved?
- Q3 had the worst completeness (2.0 at every k) — the retriever kept returning the same chunks regardless of k, never surfacing the full set of tool mentions
- Q1's completeness hit **1.0 at k=5** — the most severe completeness failure in the experiment
- Q4 (safety warnings) showed context overflow at k=10: completeness *dropped* from 4.0 to 3.0 as extra irrelevant chunks crowded the context

### 3. Does contradictory information in different chunks cause problems?
- **Q3** had a persistent hallucination score of **2.0** at every k — the only query where the model consistently invented content to fill aggregation gaps
- **Q1 at k=5** hit hallucination = **1.0** — when completeness and accuracy collapsed simultaneously, the model hallucinated rather than admitting the gaps
- **Q5** was hallucination-resistant at 5.0 across all k values — bridging electrical and ignition systems did not cause the model to invent connections

### 4. What is the optimal k for synthesis tasks?

There is no single optimal k — it depends on query type:

| Query type | Best k | Reason |
|---|---|---|
| Comparison (Q2) | Any | Stable across 3, 5, 10 |
| System overview (Q5) | 3 or 10 | Avoid k=5 instability |
| Enumeration (Q1) | 10 | Best completeness, acceptable hallucination |
| Safety summary (Q4) | 3 or 5 | k=10 causes context overflow |
| Aggregation (Q3) | None worked | Retrieval strategy needs rethinking |



## Conclusion

RAG handles synthesis well when the question provides a clear structure for the model to fill in — comparisons, system explanations, and overviews all performed reliably. It breaks down on **open-ended aggregation** (collect every mention of X across the whole document), where the retriever cannot guarantee full coverage and the model fills gaps with hallucinated content.

The most actionable finding is that **k=5 was the most unstable setting for this corpus**, producing worse results than both k=3 and k=10 across multiple queries. This suggests a corpus-specific retrieval cliff where mid-range k introduces noise without adding coverage.
