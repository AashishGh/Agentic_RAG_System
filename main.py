# main.py (project root)
import uvicorn
from src.api import app

if __name__ == "__main__":
    # Directly run the FastAPI app defined in src/api.py
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
