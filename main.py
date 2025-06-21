from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('Holiday Calendar-25.pdf')

pages = loader.load()

print(pages[0].page_content)
