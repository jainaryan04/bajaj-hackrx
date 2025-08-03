from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import List
import secrets
import os
import logging
from dotenv import load_dotenv
from model import ask_model
from openai import OpenAIError

load_dotenv()

app = FastAPI()
API_KEY = os.getenv("API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RunRequest(BaseModel):
    documents: str 
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=RunResponse)
def run(
    data: RunRequest,
    authorization: str = Header(...)
):
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid token format. Must be 'Bearer <token>'."
        )

    token = authorization.split(" ")[1]
    if not secrets.compare_digest(token, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid or missing API key"
        )
      
    try:
        logging.info(f"Received document URL: {data.documents}")
        logging.info(f"Received questions: {data.questions}")

        answers = ask_model(pdf_url=data.documents, questions=data.questions)

        logging.info(f"Generated answers: {answers}")

        return {"answers": answers}
    
    except OpenAIError:
        logging.exception("OpenAI API error occurred.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI service error: The GEMINI_API_KEY environment variable is not set or is invalid. Please check your .env file."
        )
    except Exception as e:
        logging.exception("An unexpected error occurred.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
