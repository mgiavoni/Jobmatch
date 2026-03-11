from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client

app = FastAPI()

client = OpenAI(api_key="")

supabase = create_client(
    "",
    ""
)

class EvaluationRequest(BaseModel):
    job_description: str
    alignment_transcript: str = ""
    resume: str
    interview_transcript: str = ""
    evaluation_type: str


@app.post("/evaluate")
def evaluate(data: EvaluationRequest):

    # -------------------------
    # CREATE QUERY
    # -------------------------

    query_text = f"""
Resume:
{data.resume}

Interview:
{data.interview_transcript}

Job Description:
{data.job_description}

Manager Alignment:
{data.alignment_transcript}
"""

    # -------------------------
    # CREATE EMBEDDING
    # -------------------------

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    ).data[0].embedding

    # -------------------------
    # VECTOR SEARCH
    # -------------------------

    response = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": 5
        }
    ).execute()

    framework_context = "\n\n".join(
        [c["chunk_text"] for c in response.data]
    )

    # -------------------------
    # PROMPT
    # -------------------------

    prompt = f"""
You are a strict recruiting evaluator.

Use ONLY the information provided.

FRAMEWORK CONTEXT:
{framework_context}

JOB DESCRIPTION:
{data.job_description}

ALIGNMENT TRANSCRIPT:
{data.alignment_transcript}

RESUME:
{data.resume}

INTERVIEW TRANSCRIPT:
{data.interview_transcript}

Evaluate:
- Fit with the role
- Real seniority
- Compare the candidate evidence explicitly against each retrieved framework level before selecting the final level
- Career stability
- Career progression
- Risks
- Potential

Return STRICT JSON:

{{
  "score": number,
  "estimated_seniority": "",
  "singularity_level": "",
  "strengths": [],
  "gaps": [],
  "risks": [],
  "potential": [],
  "recommendation": ""
}}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an expert technical recruiter."},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "evaluation": completion.choices[0].message.content,
        "framework_chunks_used": response.data
    }