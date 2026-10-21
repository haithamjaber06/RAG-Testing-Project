import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
 

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    result = []
    for start in range(0, len(words), chunk_size - overlap):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        result.append({"chunk_id": len(result), "text": chunk})
        if end > len(words):
            break
    return result
    
with open("doc.txt", encoding= "utf-8") as f:
    text = f.read()
    
chunks = chunk_text(text)
print(f"{len(chunks)} Chunks")

chunk_vector = model.encode([c["text"] for c in chunks])

def retrieve(query, k=3):
    query_vec = model.encode(query)
    scores = []
    for ch_v, ch in zip(chunk_vector, chunks):
        sim = float(cosine_similarity(ch_v, query_vec))
        scores.append({"id": ch["chunk_id"], "score": sim, "chunk": ch})
        
    s_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return s_scores[:k]

def expand_with_neighbors(results, window=1):
    ids = set()
    for r in results:
        center = r["id"]
        for neighbor in range(center - window, center + window + 1):
            if 0 <= neighbor < len(chunks):
                ids.add(neighbor)

    return [chunks[i] for i in sorted(ids)]

while True:
    query = input("\nQuestion (or 'q' to quit): ")
    if query == 'q':
        break
    for r in retrieve(query):
        print(f"\n[{r['id']}] {r["score"]:.3f}\n{r['chunk']['text'][:300]}...")