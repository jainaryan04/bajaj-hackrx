from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import List
import secrets
import os
from dotenv import load_dotenv
from model import ask_model
from openai import OpenAIError

load_dotenv()
app = FastAPI()
API_KEY = os.getenv("API_KEY", "supersecure123")

class RunRequest(BaseModel):
    documents: str 
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=RunResponse)
def run(
    data: RunRequest,
):  
    try:
        answers = ask_model(pdf_url=data.documents, questions=data.questions)
        return {"answers": answers}
    
    except OpenAIError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI service error: The GEMINI_API_KEY environment variable is not set or is invalid. Please check your .env file."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

