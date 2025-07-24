import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM

# --- Configuration ---
MODEL_NAME = "llama3:8b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CROSS_ENCODER_TOP_K = 3
TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", ".", "!", "?", " ", ""]

# --- Cache Resources ---
@st.cache_resource
def load_cross_encoder_model() -> CrossEncoder:
    with st.spinner("Loading Cross-Encoder model..."):
        return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource
def load_ollama_llm(model_name: str) -> OllamaLLM:
    with st.spinner(f"Loading Ollama LLM: {model_name}..."):
        return OllamaLLM(model=model_name)

cross_encoder_model = load_cross_encoder_model()
ollama_llm = load_ollama_llm(MODEL_NAME)

# --- Extract PDF Text ---
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(uploaded_file_bytes: bytes) -> str:
    text = ""
    try:
        doc = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# --- Chunking ---
def split_text_to_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=TEXT_SPLITTER_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

# --- Re-rank Chunks with CrossEncoder ---
def re_rank_cross_encoders(prompt: str, documents: List[str]) -> Tuple[str, List[int]]:
    relevant_text = ""
    relevant_text_ids = []

    try:
        ranks = cross_encoder_model.rank(prompt, documents, top_k=CROSS_ENCODER_TOP_K)
        for rank in ranks:
            doc_id = rank['corpus_id']
            if 0 <= doc_id < len(documents):
                relevant_text_ids.append(doc_id)
                relevant_text += documents[doc_id] + "\n\n"
            else:
                st.warning(f"Invalid document ID returned: {doc_id}")
    except Exception as e:
        st.error(f"Error during reranking: {e}")
        return "", []

    return relevant_text, relevant_text_ids

# --- Streamlit UI ---
st.title("🔍 RAG-PDF QA Assistant ")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file.read())

    if pdf_text.strip() == "":
        st.warning("No text found in PDF!")
    else:
        st.success("PDF text extracted successfully.")
        text_chunks = split_text_to_chunks(pdf_text)

        user_query = st.text_input("Ask a question about this PDF:")

        if user_query:
            relevant_text, _ = re_rank_cross_encoders(user_query, text_chunks)

            if relevant_text.strip() == "":
                st.warning("No relevant context found.")
            else:
                with st.spinner("Getting answer from LLM..."):
                    response = ollama_llm.invoke(f"Context:\n{relevant_text}\n\nQuestion:\n{user_query}")
                    st.markdown("### 🤖 Answer:")
                    st.write(response)
