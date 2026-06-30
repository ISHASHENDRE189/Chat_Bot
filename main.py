import os
import ollama
import faiss
import numpy as np
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2")

INPUT_PDF_PATH = "./input/class_10_englist_first_language.pdf"
INDEX_PATH = "pdf_faiss_database_index.faiss"
CHUNKS_PATH = "pdf_chunks_data.npy"
TOP_K = 4

# Read pdf and return text inside pdf
def get_pdf_text(pdf_path):
    full_text = ""
    reader = PdfReader(pdf_path)

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text

    return full_text

# Create chunks from full_text
def get_chunks(input_str: str):
  splitter_obj = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  all_chunks = splitter_obj.split_text(input_str)
  return all_chunks

# create embeddings for chunks 
def create_embeddings(chunks):
  embeddings=[]
  for chunk in chunks:
    response = ollama.embeddings(
      model = "nomic-embed-text",
      prompt = chunk
    )
    embeddings.append(response.embedding)
  return embeddings

# save embeddings
def save_embeddings(embeddings):
  embeddings = np.array(embeddings)
  dim = embeddings.shape[1]
  index = faiss.IndexFlatL2(dim)
  index.add(embeddings)
  faiss.write_index(index, INDEX_PATH)
  print("Faiss data is stored into : ", INDEX_PATH)
  return index

def load_embedding():
  index = faiss.read_index(INDEX_PATH)
  return index

def save_chunks(all_chunks):
  np.save(CHUNKS_PATH, np.array(all_chunks, dtype=object))
  print("Chunk Data is saved to :", CHUNKS_PATH)

def load_chunks():
  chunks_data = np.load(CHUNKS_PATH, allow_pickle = True).tolist()
  return chunks_data

def run_model(input_prompt):
  result = model.invoke(input_prompt)
  return result.content

def get_prompt(index, user_query, all_chunks):
  user_embedding = create_embedding([user_query,])
  user_embedding = np.array(user_embedding)

  # search in index
  distances, top_index = index.search(user_embedding, TOP_K)

  # Combine results 
  context = ""
  for ind in top_index[0]:
    context += all_chunks[ind]

  prompt = f"""
                you are a agent answering user question using context
                USER_QUESTION : {user_query}
                context : {context}
                answer with only result no need to mention context or prompt 
            """
  return prompt

def RAG_PIPELINE():
  # step 1 read pdf data
  pdf_data = get_pdf_text(INPUT_PDF_PATH)

  # if chunks files are not stored create else load
  if not os.path.exists(CHUNKS_PATH):
    print("No chunks found creating new chunk file")
    # create chunk is chunks are not available
    all_chunks = get_chunks(pdf_data)

    # save chunks
    save_chunks(all_chunks)
  else:
    print("Loading chunks insted creating")
    all_chunks = load_chunks()

  # create embedding if INDEX_PATH file is not there yet
  if not as.path.exists(INDEX_PATH):
    print("No embeddings found creating new embeddings")
    all_embeddings = create_embedding(all_chunks)
    
    # save embeddings
    index = save_embeddings(all_embeddings)
  else:
    print("Loading ebeddings insted creating ")
    # load the index
    index = load_embedding()

  while True:
    user_input = input("\n User: ")

    # Get prompt -- create embeddings for user questions, search in index, get top k sentences merged
    prompt_text = get_prompt(index, user_input, all_chunks)

    # Run model and print result
    out_put = run_model(prompt_text)
    print("\n AI : ", output)


RAG_PIPELINE()









