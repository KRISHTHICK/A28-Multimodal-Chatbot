# Install needed packages:
# pip install streamlit chromadb langchain ollama pdf2image pytesseract pdfplumber
# Also install system packages:
# sudo apt-get install poppler-utils tesseract-ocr

import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import tempfile

# Function to extract text + OCR text
def extract_text_and_ocr_from_pdf(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

        images = convert_from_path(tmp_path)
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                all_text += ocr_text + "\n"

    return all_text

# Cache embedding store
@st.cache_resource
def create_vector_store(all_text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.create_documents([all_text])

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(texts, embedding_model)

# Streamlit UI
st.set_page_config(page_title="Multi-PDF + OCR RAG Chatbot", layout="wide")
st.title("🖼️📄 Multi-PDF RAG Chatbot (Text + Image OCR)")

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Extracting documents..."):
        all_text = extract_text_and_ocr_from_pdf(uploaded_files)
        vector_store = create_vector_store(all_text)
        retriever = vector_store.as_retriever()
        ollama_llm = Ollama(model="llama3")
        rag_chain = RetrievalQA.from_chain_type(llm=ollama_llm, retriever=retriever)

    # Chat interface
    if "history" not in st.session_state:
        st.session_state.history = []

    user_question = st.chat_input("Ask anything about your uploaded PDFs!")

    if user_question:
        with st.spinner("Thinking..."):
            answer = rag_chain.run(user_question)
            st.session_state.history.append((user_question, answer))

    # Show chat history
    for idx, (q, a) in enumerate(st.session_state.history):
        with st.chat_message("user", avatar="👤"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(a)

    # Sidebar history
    st.sidebar.markdown("### Chat History")
    for idx, (q, a) in enumerate(st.session_state.history):
        st.sidebar.markdown(f"**Q:** {q}")
else:
    st.info("👆 Please upload one or more PDFs to get started!")

st.caption("Built with ❤️ using Ollama and Streamlit")
