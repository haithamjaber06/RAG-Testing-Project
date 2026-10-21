import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials.",
    "The cat is sleeping on the sofa.",
    "What's the weather like tomorrow?",
    "My account is locked and I can't sign in.",
    "I can sign in to my account just fine",
    "Error E-4012 on checkout",
]

vectors = model.encode(sentences)

def cosine_similarity(a, b):
     return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
 
query = "my credentials aren't working"
query_vec = model.encode(query)

result = []
for vec, sent in zip(vectors, sentences):
    score = cosine_similarity(vec, query_vec)
    result.append((float(score), sent))

result = sorted(result, reverse=True)
for r in result:
    print(r)