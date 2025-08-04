import os
import fitz
import requests
import gc
import psutil
import logging
from fastapi import HTTPException
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
import uuid
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
import asyncio
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import httpx
process = psutil.Process(os.getpid())

def log_memory(stage):
    mem = process.memory_info().rss / 1024**2  # MB
    logging.info(f"[Memory] {stage}: {mem:.2f} MB")



async def ask_model(pdf_url, questions):
    system_prompt = SystemMessagePromptTemplate.from_template(
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
    )


    human_prompt = HumanMessagePromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    log_memory("Start")

    async with httpx.AsyncClient() as client:
        response = await client.get(pdf_url)
        response.raise_for_status()
        content = response.content


    tmp_path = f"temp_{uuid.uuid4().hex}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(content)

    with fitz.open(tmp_path) as doc:
        full_text = "".join(page.get_text() for page in doc)
    os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
    vectordb = FAISS.from_documents(docs, embedding=embeddings)

    dense_retriever = vectordb.as_retriever(search_kwargs={"k": 7})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10
    ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.7, 0.3] 
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key),
        retriever=ensemble_retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": chat_prompt}
    )

    log_memory("Before answering")

    answers = []
    async def get_answer(q):
        try:
            return await asyncio.to_thread(lambda: qa_chain.run(q).strip())
        except Exception as e:
            logging.exception("AI or other error")
            return f"[Error answering question: {str(e)}]"
        
    answer_tasks = [get_answer(q) for q in questions]
    answers = await asyncio.gather(*answer_tasks)

    


    del qa_chain, ensemble_retriever, vectordb, embeddings, docs, full_text,dense_retriever,bm25_retriever
    gc.collect()

    log_memory("After cleanup")

    return answers