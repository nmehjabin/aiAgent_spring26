# Exercise 3, Task 1: Sequential vs Parallel MMLU Evaluation with Ollama

## Setup

- **Model:** Llama 3.2-1B via Ollama (`llama3.2:1b`)
- **Topics evaluated:** `business_ethics` (100 questions) and `astronomy` (152 questions)
- **Hardware:** Local Ollama server

## Results

### Accuracy

| Topic | Correct | Total | Accuracy |
|---|---|---|---|
| business_ethics | 46–48 | 100 | 46–48% |
| astronomy | 53–56 | 152 | 34.9–36.8% |

Results varied slightly across runs due to sampling, but were consistent overall. Business ethics accuracy (~47%) was notably higher than astronomy (~36%), likely because the 1B model handles factual recall and short-answer ethics questions better than the quantitative/spatial reasoning required for astronomy.

---

## Timing Observations

### Sequential Execution

```bash
time { python eval_business_ethics.py ; python eval_astronomy.py }
```

| Run | Business Ethics | Astronomy | Total Wall Time |
|---|---|---|---|
| Run 1 | 58.0s | 74.5s | ~132.5s |
| Run 2 | 40.2s | 59.8s | ~100.0s |

**Per-question time:** ~0.40s in faster runs.

---

### Parallel Execution

```bash
time { python eval_business_ethics.py & python eval_astronomy.py & wait; }
```

| Run | Business Ethics | Astronomy | Wall Time |
|---|---|---|---|
| Parallel | 76.9s | 103.1s | ~130s |

**Per-question time:** ~0.51s (business) and ~0.68s (astronomy) — noticeably slower than sequential.

---

## Key Observations

### 1. Parallel execution provided no speedup

The total wall-clock time for parallel execution (~130s) was similar to sequential (~100–132s). This was unexpected — parallel execution should in theory halve the time — but makes sense given how Ollama works.

### 2. Why? Ollama queues requests to a single model instance

Ollama runs one model instance and serializes inference requests. When both scripts fire simultaneously, they compete for the same GPU/CPU resources and the same model thread. Rather than truly running in parallel, the requests are internally queued, causing each individual evaluation to slow down (~0.68s/question vs ~0.40s/question).

### 3. Per-question time increased during parallel runs

During sequential execution, each request to Ollama received the full resource budget of the machine. During parallel execution, the two processes interleaved their requests, causing memory contention and slower token generation — reflected in the ~60–70% increase in per-question latency.

### 4. True parallelism would require multiple model instances

To actually benefit from parallel execution, you would need either:
- Two separate Ollama servers (different ports), or
- A model server that supports concurrent batched inference (e.g., vLLM, TGI)

With a single Ollama instance, `time { ... & ... & wait }` effectively just adds scheduling overhead on top of what is still sequential inference.

---

## Conclusion

For single-GPU local inference with Ollama, **sequential execution is as fast or faster than naive parallelism** because the bottleneck is the model server, not the Python client scripts. The shell-level `&` operator achieves process-level parallelism but cannot overcome the serial nature of the underlying inference engine. Meaningful speedup would require a multi-instance or batched inference setup.
