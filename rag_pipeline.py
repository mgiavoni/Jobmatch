from pypdf import PdfReader
from openai import OpenAI
from supabase import create_client
import os

client = OpenAI(api_key="")

supabase = create_client(
    "",
    ""
)

def extract_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

def index_pdf(path):

    text = extract_pdf_text(path)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):

        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        supabase.table("document_chunks").insert({
            "chunk_text": chunk,
            "chunk_index": i,
            "embedding": emb
        }).execute()

index_pdf("individual.pdf.pdf")
index_pdf("lideranca.pdf.pdf")