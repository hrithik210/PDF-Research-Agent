from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

load_dotenv()

loader = PyPDFLoader('Holiday Calendar-25.pdf')
splitter = RecursiveCharacterTextSplitter(
  chunk_size = 100,
  chunk_overlap = 50,
)

pages = loader.load()
print(pages[0].page_content)

docs = splitter.split_documents(pages)
print("docs : " , docs)


#Emeddings
embeddings = FakeEmbeddings(size = 1536)

db = FAISS.from_documents(docs, embeddings)
print(f"📦 FAISS vector store created with {db.index.ntotal} chunks.")


#initializing llm
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template("""
You are an assistant answering questions based on the following PDF context.

<context>
{context}
</context>

Question: {input}
""")

retriever = db.as_retriever()

qa_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)



print(qa_chain.invoke("What is the holiday calendar for 2025?"))