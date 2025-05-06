# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents.executor import execute
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agentic RAG API")

# very permissive—lock down origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_methods=["*"],           
    allow_headers=["*"],           
    allow_credentials=True,
)

class QueryModel(BaseModel):
    query: str


def clean_text(text: str) -> str:
    """
    Clean the model output by removing newlines, tabs, and collapsing whitespace.
    """
    # Remove newline and tab characters
    cleaned = text.replace('\n', ' ').replace('\t', ' ')
    # Collapse multiple spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned

# Wrap executor.execute into rag_model interface
class RAGModel:
    def generate_response(self, query: str):
        # The executor returns the final answer string
        answer = execute(query)
        # Return tuple (response, original question)
        return answer, query


rag_model = RAGModel()
@app.post("/generate-response/")
async def generate_response(query_model: QueryModel):
    # Handle simple greetings without invoking RAG
    greeting = query_model.query.strip().lower()
    if greeting in ("hi", "hello", "hey", "hey there", "good morning", "good afternoon", "good evening"):
        # return a friendly greeting
        return {"response": "Hello! How can I assist you today?", "question": query_model.query}
    
    try:
        
        raw, question = rag_model.generate_response(query_model.query)
        # find the last “Answer:” in the returned text, and keep only what follows
        if "Answer:" in raw:
            answer = raw.split("Answer:", 1)[1]
            
        else:
            answer = raw
        # Remove any trailing 'Question:' fragments
        answer = answer.split("Question:", 1)[0]
        # clean whitespace
        clean = answer.replace("\n", " ").replace("\t"," ")
        clean = " ".join(clean.split())    
        clean += '.'
        return {"response": clean, "question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # Entry point for local testing
# if __name__ == "__main__":
#     # Run with: python api.py
#     uvicorn.run(app, host="0.0.0.0", port=8000)

