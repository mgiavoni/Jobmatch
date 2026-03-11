from openai import OpenAI
from supabase import create_client
import numpy as np
import ast

SUPABASE_URL = ""
SUPABASE_KEY = ""
OPENAI_API_KEY = ""

client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

question = "What does this job require?"

emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

query_embedding = emb.data[0].embedding

response = supabase.table("document_chunks").select("*").execute()
chunks = response.data

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = []

for chunk in chunks:
    embedding = ast.literal_eval(chunk["embedding"])
    score = cosine_similarity(query_embedding, embedding)
    scores.append((score, chunk["chunk_text"]))

scores.sort(reverse=True)

context = "\n".join([c[1] for c in scores[:3]])

answer = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer based on the context"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
)

print(answer.choices[0].message.content)