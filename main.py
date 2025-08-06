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
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    # Use logger for critical startup errors
    raise ValueError("API_KEY environment variable not set.")


class RunRequest(BaseModel):
    documents: str 
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=RunResponse)
async def run(
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
        # --- KEY CHANGE 2: Use the logger instance ---

        answers = await ask_model(file_url=data.documents, questions=data.questions)

        return {"answers": answers}
    
    except OpenAIError:
        # Use the logger for exceptions as well
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            # Corrected the error message typo
            detail="AI service error: An issue occurred with the OpenAI API. Check keys or service status."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
