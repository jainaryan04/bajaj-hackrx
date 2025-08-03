import os
import fitz
import requests
import gc
import psutil
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

process = psutil.Process(os.getpid())

def log_memory(stage):
    mem = process.memory_info().rss / 1024**2  # MB
    logging.info(f"[Memory] {stage}: {mem:.2f} MB")

def ask_model(pdf_url, questions):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    log_memory("Start")

    response = requests.get(pdf_url)
    response.raise_for_status()

    tmp_path = "temp_downloaded.pdf"
    with open(tmp_path, "wb") as f:
        f.write(response.content)

    with fitz.open(tmp_path) as doc:
        full_text = "".join(page.get_text() for page in doc)
    os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectordb = FAISS.from_documents(docs, embedding=embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )

    log_memory("Before answering")

    answers = []
    for q in questions:
        try:
            answer = qa_chain.run(q).strip()
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error: {str(e)}")

    del qa_chain, retriever, vectordb, embeddings, docs, full_text
    gc.collect()

    log_memory("After cleanup")

    return answers
