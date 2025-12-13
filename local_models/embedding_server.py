from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import logging
from typing import List
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-server")

app = FastAPI(title="Local Embedding Service")

# Load model
MODEL_NAME = "all-MiniLM-L6-v2"
# Use /root/.cache/huggingface which is mounted
# Defaults to /app/weights which is mounted from host ./local_models/weights
cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME", "/app/weights")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_folder

logger.info(f"Loading model: {MODEL_NAME} from {cache_folder}...")
# Ensure directory exists
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder, exist_ok=True)
    
model = SentenceTransformer(MODEL_NAME, cache_folder=cache_folder)
logger.info("Model loaded successfully.")

class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
async def embed(request: EmbedRequest):
    if not request.texts:
        return {"embeddings": []}
    
    try:
        # Generate embeddings
        # convert_to_numpy=True is default, but we need list for JSON
        embeddings = model.encode(request.texts, convert_to_numpy=True).tolist()
        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
