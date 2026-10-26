# RAG Testing Project

A hands-on project for learning how Retrieval-Augmented Generation (RAG) works, built up one version at a time. Each version adds one technique and is measured against a small hand-labelled eval set, so every change has to show a number that went up.

The corpus is `doc.txt`, a gym and health handbook that also contains the member handbook of Iron Haven Fitness, a fictional gym.

## Current status: v0 (naive baseline)

The pipeline so far:

1. **Load:** read `doc.txt`.
2. **Chunk:** split it into fixed-size word chunks (150 words with a 30-word overlap).
3. **Embed:** encode each chunk with [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from `sentence-transformers`.
4. **Retrieve:** rank chunks by cosine similarity to the query and return the top-k.
5. **Answer:** put the top 3 chunks into a prompt and ask Gemini (`gemini-3.5-flash`) to answer using only that context, or reply "I couldn't find that in the documents."
6. **Evaluate:** measure recall@k on a 10-question eval set where each question is labelled with the chunk IDs that answer it, and grade each generated answer as correct, false refusal or wrong.

Baseline results:

- **recall@3 = 0.80** (8 of 10 questions found a correct chunk in the top 3).
- **Answers:** 8 correct, 2 false refusals, 0 wrong. Both false refusals are the two retrieval misses: the answer chunk never reached the model.

These are the numbers later versions have to beat.

## v1: Make it honest

What changed in v1:

- **Citations:** the prompt now requires a citation after every sentence, in the format `[chunk N]`.
- **Structured output:** `answer()` returns a dict with the answer, the retrieved chunk IDs, the cited chunk IDs, invalid citations (cited but never retrieved), uncited retrieved chunks, and a `no_answer` flag. Citations are parsed with a regex.
- **Refusal test set:** 9 questions whose answers are not in `doc.txt`. 3 are fully out of scope (e.g. "who won the last football World Cup?"). 6 are near misses, where the topic is in the doc but the specific answer isn't (e.g. a made-up error code IH-777, protein for a 12-year-old, creatine during pregnancy, bringing a dog, paying a year upfront, public holidays).
- **Grading:** each result is sorted into correct, false refusal, wrong, or correct refusal. A key-word check gives a first label, then every answer was read and checked by hand.
- **Replaced question:** "how much does the Titan plan cost per year?" was planned but dropped, because the model can answer it correctly by multiplying the monthly price (60 × 12 = 720 JOD), so it doesn't test refusal.
- **Normalization:** model output is normalized with `unicodedata` NFKC, because the model inserts narrow no-break spaces that break text matching.
- **New model:** generation moved from Gemini to `openai/gpt-oss-120b` on Groq (free tier). Gemini calls were taking up to ~2 minutes; Groq calls take under a second.

Results:

|                   | v0 (Gemini) | v1 (gpt-oss-120b) |
|-------------------|-------------|------------------|
| correct           | 8 / 10      | 8 / 10           |
| false refusal     | 2           | 1                |
| wrong             | 0           | 1                |
| correct refusals  | n/a         | 9 / 9            |
| invalid citations | n/a         | 0                |

Findings:

- **The "days off" question was answered wrongly.** The chunk with the answer (chunk 36: beginners should take at least one or two full rest days per week) ranked 5th, so with k=3 the model never saw it. Instead of refusing, the model worked out "four days off" from the 3-day training program chunks. A retrieval miss turned into a made-up answer, not a refusal. Chunk 36 mixes sleep tips with rest days, which likely weakens its match. To re-test in v2 and v3.
- **1 false refusal:** "can I bring a friend with me to the gym?" (the doc uses the word "guest").
- **Recall@3 is unchanged from v0**, because v1 did not change retrieval.
- **Caveat:** v1 changed two things at once, the prompt and the model, so the change from v0 can't be credited to either one alone.

## Files

| File | What it is |
|---|---|
| `search.ipynb` | Main notebook: chunking, retrieval, the eval set and recall@k, an interactive query loop, and the RAG `answer()` function with its graded results |
| `search.py` | Script version of the retrieval pipeline with the interactive query loop |
| `emb101.py` | First experiment: embedding a few sentences and ranking them by cosine similarity |
| `doc.txt` | The document being searched |

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

For the answer step, get a free Groq API key from [console.groq.com](https://console.groq.com), then copy `.env.example` to `.env` in the project folder and put your key in it (`.env` is git-ignored):

```
GROQ_API_KEY=your-key-here
```

Open `search.ipynb` and run the cells in order. The interactive search cell loops until you type `q`, so skip it if you only want to run the rest. To run the script version of the retrieval loop instead, use `python search.py`.

## Roadmap

This is the expected plan and may change as the project goes on.

- ✅ **v0: Naive baseline.** Load, chunk at a fixed size, embed, retrieve the dense top-k, put the chunks into a prompt, and answer. Its job is to give a baseline number.
- ✅ **v1: Make it honest.** Add citations (return chunk IDs and show the sources), plus a "not in the docs" refusal. The model must ground every claim in a retrieved chunk.
- **v2: Hybrid recall.** Add BM25 keyword search and combine it with dense search using Reciprocal Rank Fusion. BM25 catches exact tokens such as error codes and names, and dense search catches paraphrases and intent.

  ```python
  def hybrid_recall(query, k_each=50):
      bm25_ids  = bm25_search(query, k=k_each)     # exact tokens, codes, names
      dense_ids = vector_search(query, k=k_each)   # paraphrase, intent
      return rrf([bm25_ids, dense_ids], k=60)[:k_each]  # rank-based fuse
  ```

- **v3: Precision with a reranker.** Recall wide (top 50), then rerank down to the top 5. The recall stage makes sure the answer is somewhere in the candidates, and the reranker makes sure the right 5 end up at the top.
- **v4: Chunking sweep.** Stop guessing the chunk size. Sweep chunk sizes against the eval set and choose one from the recall@20 / precision@5 tradeoff. If precision@5 is still below 0.7, add [Anthropic-style contextual retrieval](https://www.anthropic.com/news/contextual-retrieval).
- **v5: Observability and cost.** Connect Langfuse to trace each query's retrieved chunks, tokens, latency and cost. Add a cache and a cost cap.
