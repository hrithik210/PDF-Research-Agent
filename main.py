from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import FAISS



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