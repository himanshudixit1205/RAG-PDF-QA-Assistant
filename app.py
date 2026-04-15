# Imports
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from typing import List, Tuple, Any

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# Configuration
MODEL_NAME = "llama3:8b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
MAX_CONTEXT_LENGTH = 4000
TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", ".", "!", "?", " ", ""]

# Load models
@st.cache_resource
def load_models() -> Tuple[SentenceTransformer, CrossEncoder, OllamaLLM]:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    llm = OllamaLLM(model=MODEL_NAME)
    return embed_model, cross_encoder, llm

embed_model, cross_encoder, llm = load_models()

# Extract Text from PDF
@st.cache_data
def extract_pdf(pdf_bytes: bytes) -> str:
    text = ""
    doc = fitz.open(stream=pdf_bytes, filetype='pdf') # Fitz is used for extraction, analysis, conversion, and manipulation of PDF and other document formats
    
    for i, page in enumerate(doc):
        text += f"[Page {i+1}]\n" + page.get_text()
        
    doc.close()
    return text

# Chunks
def get_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        separator=TEXT_SPLITTER_SEPARATORS,
        chunk_size = CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

# Embeddings are numerical vector representations of text that capture semantic meaning, allowing systems to find relevant context by measuring similarity (often using cosine similarity) between a user query and stored data.
@st.cache_data
def create_embeddings(chunks: List[str]) -> np.ndarray:
    return embed_model.encode(chunks, normalize_embeddings=True)

# Store in FAISS
@st.cache_resource
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Euclidean distance (L2 distance)
    index.add(np.array(embeddings))
    return index

# Retrieve
def retrieve_chunks(query, index, chunks, k):
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

# Re-Ranking
def rerank(query, retrieved_chunks):
    ranks = cross_encoder.rank(query, retrieved_chunks, top_k=3)
    final_chunks = [retrieved_chunks[r['corpus_id']] for r in ranks]
    return "\n\n".join(final_chunks)

# Context
def check_context(context):
    
    if len(context) > MAX_CONTEXT_LENGTH:
        return context[:MAX_CONTEXT_LENGTH]
    return context

# LLM
def generate_answer(context, query):
    context = check_context(context)
    prompt = f"""
    You are a strict AI assistant.
    
    Answer only from the context.
    Do not add outside knowledge.
    
    Context:
    {context}
    
    Question:
    {query}
    
    If answer is not found, say "Not Found".
    """
    try:
        return llm.invoke(prompt)
    except Exception as e:
        return f"⚠️ Error: Could not connect to Ollama. Make sure it is running.\n\nDetails: {e}"

# UI
st.title("RAG PDF QA System")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # Reset session if new file
    if "filename" in st.session_state and st.session_state.filename != uploaded_file.name:
        st.session_state.clear()

    # Process PDF only if not already processed
    if "index" not in st.session_state:

        with st.spinner("Processing PDF..."):
            text = extract_pdf(uploaded_file.read())

            if not text.strip():
                st.error("No extractable text. PDF might be scanned or image-based.")
                st.stop()

            chunks = get_chunks(text)
            if not chunks:
                st.error("Text extracted but no chunks created. PDF may be too short.")
                st.stop()
                
            embeddings = create_embeddings(chunks)
            index = create_faiss_index(embeddings)

            # Store in session state
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.filename = uploaded_file.name
            st.session_state.text = text

        st.success("PDF processed successfully")

    else:
        index = st.session_state.index
        chunks = st.session_state.chunks
        text = st.session_state.text

    # Summarization
    if st.button("Summarize Document"):
        with st.spinner("Summarizing..."):
            summary_chunks = retrieve_chunks("Summarize the main points", index, chunks, k=TOP_K)
            context = rerank("Summarize the main points", summary_chunks)
            context = check_context(context)

            try:
                summary = llm.invoke(f"Summarize the following document content:\n{context}")
                st.write(summary)
            except Exception:
                st.error("Ollama not running or connection failed.")

    # Query
    query = st.text_input("Ask a question")

    if query:
        # Slider
        k_value = st.slider("Number of chunks to retrieve", 3, 15, 10)

        with st.spinner("Generating answer..."):
            retrieved = retrieve_chunks(query, index, chunks, k=k_value)
            
            if not retrieved:
                st.warning("No relevant chunks found")
                st.stop()
            
            context = rerank(query, retrieved)
            context = check_context(context) 
            answer = generate_answer(context, query)

        st.write("**Answer:**", answer)

        # Sources
        with st.expander("Sources"):
            for i, chunk in enumerate(retrieved[:k_value]):
                st.write(f"Chunk {i+1}: {chunk[:200]} ...")
