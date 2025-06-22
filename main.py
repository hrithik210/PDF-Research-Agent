from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = PyPDFLoader('Holiday Calendar-25.pdf')
splitter = RecursiveCharacterTextSplitter(
  chunk_size = 100,
  chunk_overlap = 50,
)

pages = loader.load()

print(pages[0].page_content)
