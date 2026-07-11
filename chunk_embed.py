from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1️⃣ Load your document
with open("data/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2️⃣ Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"Total chunks created: {len(chunks)}")

# 3️⃣ Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4️⃣ Store in FAISS
vectorstore = FAISS.from_texts(chunks, embeddings)
vectorstore.save_local("vector_index")

print("✅ Embeddings created and stored successfully!")
