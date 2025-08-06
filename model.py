import os
import fitz # PyMuPDF
import gc
import asyncio
import httpx
from dotenv import load_dotenv

import tempfile
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from docx import Document as DocxDocument
from urllib.parse import urlparse, unquote



# --- KEY CHANGE 1: Initialize all reusable objects globally, once on startup ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("FATAL: OPENAI_API_KEY environment variable not set.")

# Reusable models
EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
LLM = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Reusable prompt for final answer generation
FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
      "You are an AI assistant. Answer questions clearly and concisely in one or two sentences."
        " Base your answers strictly on the provided context. If the answer is not explicitly stated"
        " but can be reasonably inferred, do so carefully. If the context does not support even an inferred answer,"
        " reply with: 'The provided document does not have the answer to the question.'"
        "\n\n"
        "Answer in the same formal style as the examples below, using clear insurance-related phrasing and including both word and digit formats where relevant:\n\n"
        "Example 1:\n"
        "Q: What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?\n"
        "A: A grace period of thirty (30) days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.\n\n"
        "Example 2:\n"
        "Q: What is the waiting period for pre-existing diseases (PED) to be covered?\n"
        "A: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.\n"
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# Reusable prompt and chain for query rewriting
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that rewrites user questions to optimize them for retrieving relevant information from insurance or policy documents.

Rewritten queries must:
- Be concise and keyword-rich
- Eliminate conversational or filler words (e.g., "what", "how", "can I")
- Focus only on the core concepts, terms, or conditions present in formal insurance or policy clauses

Your output should resemble compact search queries that align closely with the language used in health insurance policy documents, terms and conditions, or regulatory filings.

Only return the rewritten query without explanations or extra formatting.
"""),
        ("human", "{question}")
    ])
REWRITE_CHAIN = REWRITE_PROMPT | LLM

def extract_text_from_docx(docx_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(docx_bytes)
        tmp.flush()
        doc = DocxDocument(tmp.name)
        full_text = "\n".join([para.text for para in doc.paragraphs])
    os.unlink(tmp.name)
    return full_text

def process_pdf_and_create_retriever(pdf_content: bytes) -> EnsembleRetriever:
    """Synchronous function to handle all CPU-bound processing for a PDF."""
    # --- KEY CHANGE 2: Process PDF from in-memory bytes, not a temp file ---
    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
        full_text = "".join(page.get_text() for page in doc)

    if not full_text.strip():
        raise ValueError("Could not extract text from the PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]

    vectordb = FAISS.from_documents(docs, EMBEDDINGS_MODEL)
    
    dense_retriever = vectordb.as_retriever(search_kwargs={"k": 7})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10
    
    return EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.7, 0.3])


async def get_final_answer(original_question: str, retriever: EnsembleRetriever) -> str:
    """Async pipeline to get an answer for a single question."""
    try:
        # 1. Rewrite the question for better retrieval
        rewritten_question_response = await REWRITE_CHAIN.ainvoke({"question": original_question})
        rewritten_question = rewritten_question_response.content.strip()
        
        # 2. Retrieve relevant documents using the rewritten question
        retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, rewritten_question)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 3. Generate final answer using original question and retrieved context
        final_prompt = FINAL_ANSWER_PROMPT.format_messages(context=context, question=original_question)
        final_response = await LLM.ainvoke(final_prompt)
        return final_response.content.strip()

    except Exception as e:
        return f"[Error answering question: {str(e)}]"


async def ask_model(file_url: str, questions: list[str]) -> list[str]:
    """Main function to orchestrate the RAG pipeline for a PDF or a .pdf.zip archive."""
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            response.raise_for_status()
            file_bytes = response.content
    except httpx.RequestError as e:
        return [f"[Error: Could not access the file at {file_url}]"] * len(questions)

    retriever = None
    parsed_url = urlparse(file_url)
    path = unquote(parsed_url.path).lower() 

                      

    if path.lower().endswith(".pdf"):
        try:
            retriever = await asyncio.to_thread(process_pdf_and_create_retriever, file_bytes)
        except Exception as e:
            return ["[Error: Failed to process the PDF document.]"] * len(questions)
        
    elif path.lower().endswith(".docx"):
        try:
            full_text = await asyncio.to_thread(extract_text_from_docx, file_bytes)
            if not full_text.strip():
                raise ValueError("DOCX file contains no readable text.")

            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]

            vectordb = FAISS.from_documents(docs, EMBEDDINGS_MODEL)
            dense_retriever = vectordb.as_retriever(search_kwargs={"k": 7})
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 10
            retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.7, 0.3])
        except Exception as e:
            return ["[Error: Could not process DOCX document.]"] * len(questions)
        
    elif path.endswith((".png", ".jpg", ".jpeg")):
        return ["The provided document does not have the answer to the question."] * len(questions)
               
             


    else:
        return ["The provided document does not have the answer to the question."] * len(questions)


    # Run all question-answering tasks concurrently
    tasks = [get_final_answer(q, retriever) for q in questions]
    answers = await asyncio.gather(*tasks)

    # Clean up
    del retriever
    gc.collect()

    return answers