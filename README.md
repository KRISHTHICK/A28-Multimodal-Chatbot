# A28-Multimodal-Chatbot
GenAI

now:

✅ Accept documents that contain images (not just plain text).
✅ Work with different topics or domains (generalized, flexible).
✅ Do RAG (Retrieve + Answer) based on text extracted from text + images.

🛠️ Here's What We Need to Do:
1. Support for Documents with Embedded Images
PDFs often have images (diagrams, charts, etc.).

We can extract text from images inside the PDF using OCR (Optical Character Recognition).

Library: pdfplumber, PyMuPDF, or better: pytesseract + pdf2image.

2. Multi-modal Handling (Text + OCR text)
Extract regular text normally.

For each page:

If it has images, OCR those images too.

Merge normal text + OCR text into the knowledge base.

3. Generalize for Different Domains
No hardcoding domain-specific logic.

Build a document type detector (optional), or just handle everything uniformly first.

📚 Updated Tools Required:

Task	Tool
PDF Parsing (Text)	PyPDFLoader
PDF Parsing (Images)	pdf2image + pytesseract
OCR (Image to Text)	pytesseract
Text Split + Embedding	Ollama + Chroma
Local LLM Inference	Ollama (any model like llama3)
Frontend Chat UI	Streamlit
🎯 Plan:
Upload PDFs (containing text + images).

Extract normal text ✅

Extract images → OCR → Extracted text ✅

Combine both types of text ✅

Create embeddings ✅

Perform RAG Q&A ✅

⚡ Would you like me to start coding this upgraded version for you?
If yes, I can build it in small clean steps so you understand it clearly too. 🚀

(First, I’ll update the document loader part!)
✅ Ready?
(just reply "Yes let's start")






You said:
yes


ChatGPT said:
Awesome! 🚀
Let's start upgrading step-by-step.

✨ Step 1: Install Extra Libraries
You need these first if not already installed:

bash
Copy
Edit
pip install pdf2image pytesseract pdfplumber
sudo apt-get install poppler-utils tesseract-ocr
(On Windows: install Poppler and Tesseract.)

✅ pdf2image → Convert PDF pages into images
✅ pytesseract → OCR text extraction from images
✅ pdfplumber → Extract structured text and detect images

✨ Step 2: New Function to Extract Text + OCR from PDFs
Here’s a clean function we’ll add:

python
Copy
Edit
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import tempfile

def extract_text_and_ocr_from_pdf(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # 1. Extract normal text using pdfplumber
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

        # 2. Convert pages to images and OCR
        images = convert_from_path(tmp_path)
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                all_text += ocr_text + "\n"

    return all_text
✨ Step 3: Update Loading Process in Your App
Replace your old load_and_split_pdfs() with this new function!

Then do text splitting + embedding as before.

🚀 Summary of Changes:

Old	New
Only loads text	Load text + OCR from images
Only PyPDFLoader	Use pdfplumber, pdf2image, pytesseract
Handles plain text	Handles any document with diagrams, images, charts
🎯 NEXT:
If you say yes,
I’ll immediately show you the full updated app.py, clean and ready for text+image documents 🔥

Continue? ("Yes, show app.py") 🚀






You said:
yes, with steps how upload in github and how to run with vscode 


ChatGPT said:
Perfect. Let's go step-by-step clearly. 🚀

🛠️ Updated Full app.py for PDF (Text + Images OCR) + RAG Chatbot
python
Copy
Edit
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
📂 Steps to Upload this to GitHub
1. Create a Local Folder
bash
Copy
Edit
mkdir rag-ocr-chatbot
cd rag-ocr-chatbot
2. Add the file
Save the above code as app.py.

3. Initialize Git Repository
bash
Copy
Edit
git init
git add app.py
git commit -m "Initial commit: OCR RAG chatbot"
4. Create GitHub Repo (Online)
Go to github.com

Click New Repository.

Name it (example: rag-ocr-chatbot).

Don't initialize README (we'll push everything).

5. Push Code to GitHub
After creating repo, GitHub will give you push commands.

Example:

bash
Copy
Edit
git remote add origin https://github.com/your-username/rag-ocr-chatbot.git
git branch -M main
git push -u origin main
✅ Done!

🖥️ How to Run this Project in VS Code
Open VS Code

Open the folder rag-ocr-chatbot

Create and activate a Python environment (optional but better):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\\Scripts\\activate    # (Windows)
Install the required libraries:

bash
Copy
Edit
pip install streamlit langchain chromadb ollama pdf2image pytesseract pdfplumber
Run the app:

bash
Copy
Edit
streamlit run app.py
It will open automatically in your browser:
http://localhost:8501

✅ That's it! You can now upload PDFs, even with images, and chat with it.
