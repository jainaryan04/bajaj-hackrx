import os
import fitz
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA


def ask_model(pdf_url, questions):
    load_dotenv()
    api_key=os.getenv("OPENAI_API_KEY")

    # Download PDF
    response = requests.get(pdf_url)
    response.raise_for_status() 

    tmp_path="temp_downloaded.pdf"
    with open(tmp_path, "wb") as f:
        f.write(response.content)

    # Extract text
    with fitz.open(tmp_path) as doc:
        full_text="".join(page.get_text() for page in doc)
    os.remove(tmp_path)

    # Split document into chunks
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs=[Document(page_content=chunk) for chunk in splitter.split_text(full_text)]

    # Embedding model
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    # Create vector store
    vectordb=FAISS.from_documents(docs, embedding=embeddings)

    # Create retriever-based QA chain
    retriever=vectordb.as_retriever(search_kwargs={"k":3})
    qa_chain=RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )

    # Ask each question
    answers=[]
    for q in questions:
        try:
            answer=qa_chain.run(q).strip()
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error: {str(e)}")

    return answers
