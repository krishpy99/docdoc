from fastapi import FastAPI
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel

class Data(BaseModel):
    text: str

app = FastAPI()

vectors = None
index = None

@app.get("/")
async def root():
    return {"message": "Testing..."}

@app.post("/add")
async def add(data: Data):
    chunks = await chunkdata(data.text)
    text = []
    for i in chunks:
        text.append(i.page_content)
    encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
    vectors = await encoder.encode(text)
    #print(vectors.shape[1])
    print(text)
    return
    index = faiss.IndexFlatL2(vectors.shape[1])
    print(index)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    print(index)
    return {"response": "Indexed successfully"}

@app.post("/ask")
async def ask(prompt):
    prompt_vector = encoder.encode([prompt])
    k = 3
    D, I = index.search(prompt_vector, k)
    context = D[0] + D[1] + D[2]
    return context

async def chunkdata(text):
    custom_text_splitter = RecursiveCharacterTextSplitter(
            # Set custom chunk size
            chunk_size = 300,
            chunk_overlap  = 30,
            # Use length of the text as the size measure
            length_function = len,
            # Use only "\n\n" as the separator
            separators = ['\n\n']
            )

    # Create the chunks
    custom_texts = custom_text_splitter.create_documents([text])
    return custom_texts
