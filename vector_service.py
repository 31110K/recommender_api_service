from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from pathlib import Path
from dotenv import load_dotenv

app = FastAPI()

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

model = SentenceTransformer("all-MiniLM-L6-v2")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

if not pinecone_api_key:
    raise RuntimeError("Missing PINECONE_API_KEY. Check server/.env or shell env vars.")

if not pinecone_index_name:
    raise RuntimeError("Missing PINECONE_INDEX_NAME. Check server/.env or shell env vars.")

pc = Pinecone(api_key=pinecone_api_key, environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pc.Index(pinecone_index_name)


class PostVector(BaseModel):
    post_id: str
    title: str
    meta_description: str
    tag: str


class RecommendInput(BaseModel):
    interest_text: str

class InterestData(BaseModel):
    interest_text: str

@app.post("/store")
def store_vector(data: PostVector):

    try:
        text = f"{data.title} {data.meta_description} {data.tag}"


        vector = model.encode(
            text,
            normalize_embeddings=True
        ).tolist()

        index.upsert([{
            "id": data.post_id,
            "values": vector,
            "metadata": {
                "title": data.title,
                "tag": data.tag
            }
        }])

        return {"success": True}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/recommend")
def recommend(data: RecommendInput):


    try:

        vector = model.encode(
            data.interest_text,
            normalize_embeddings=True
        ).tolist()

        results = index.query(
            vector=vector,
            top_k=10,
            include_metadata=False
        )

        recommendations = []

        for match in results.get("matches", []):
            recommendations.append(match["id"])

        return {"recommendations": recommendations}

    except Exception as e:

        return {"recommendations": []}
    


@app.post("/similarPosts")
def similar_posts(data: InterestData):
    interest = data.interest_text
    
    result = index.query(
        vector=model.encode(interest, normalize_embeddings=True).tolist(),
        top_k=6,
        include_metadata=False
    ).get("matches", [])

    similar_posts_ids = [match["id"] for match in result]
    
    return {
        "similar_posts_ids": similar_posts_ids
    }

