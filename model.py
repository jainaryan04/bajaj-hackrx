import os
import fitz  
import requests
from dotenv import load_dotenv
from openai import OpenAI

def ask_model(pdf_url, questions):
    load_dotenv()
    api_key=os.getenv("GEMINI_API_KEY")

    response=requests.get(pdf_url)
    if response.status_code!=200:
        raise Exception(f"Failed to download PDF: {response.status_code}")

    tmp_path="temp_downloaded.pdf"
    with open(tmp_path, "wb") as f:
        f.write(response.content)

    with fitz.open(tmp_path) as doc:
        full_text="".join(page.get_text() for page in doc)

    os.remove(tmp_path)

    client=OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    answers=[]
    for q in questions:
        messages=[
            {"role":"system", "content":"You will be given a full document, answer the question asked by the user in a single line."},
            {"role":"user", "content":f"Document:\n{full_text}\n\nQuestion: {q}"}
        ]
        try:
            response=client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages
            )
            answers.append(response.choices[0].message.content.strip())
        except Exception as e:
            answers.append(f"Error: {str(e)}")

    return answers
