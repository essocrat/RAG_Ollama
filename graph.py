from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

with open("data/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

chunks=splitter.split_text(text)
print(f"total chunks created: {len(chunks)}")
print(chunks[:2])

embedding_model=OllamaEmbeddings(model="mxbai-embed-large")
metadatas = [{"source": "chapter1.txt", "chunk": i} for i in range(len(chunks))]
vectorstore=FAISS.from_texts(chunks,embedding_model,metadatas=metadatas)

#print("Vector store created with", len(chunks), "chunks.",)
print(vectorstore)

query="What is Retrieval Augumented Generation"

docs=vectorstore.similarity_search(query,k=2)

for i, doc in enumerate(docs):
    print(i,doc.page_content)

#for i, doc in enumerate(docs):
    print(f"\n--- Retrieved Chunk {i+1} ---\n")
    print(doc.page_content)



 