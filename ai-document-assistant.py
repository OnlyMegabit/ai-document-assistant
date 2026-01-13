import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# --- CACHING FUNCTIONS (For Performance) ---

@st.cache_data
def get_pdf_text(uploaded_files):
    """Extracts text from PDF and caches it to avoid re-reading the file."""
    text = ""
    for file in uploaded_files:
        reader = PdfReader(file)        
        for page in reader.pages:
            text += page.extract_text()        
    return text

@st.cache_resource
def create_vectorstore(chunks, _api_key):
    """Creates the searchable index and keeps it in memory."""
    embeddings = OpenAIEmbeddings(openai_api_key=_api_key)
    return FAISS.from_texts(chunks, embeddings)

# --- WEB INTERFACE SETUP ---

st.set_page_config(page_title="Document AI", page_icon="📂")
st.title("📂 Mini Document Assistant")
st.write("Upload a document and ask questions to get instant AI insights.")

with st.sidebar:
    uploaded_files = st.file_uploader("Upload a PDF for the AI to read", type="pdf", accept_multiple_files=True)

query = st.text_input("Ask a question about the document:")

# --- MAIN LOGIC ---

if uploaded_files and query:
    # Use your new secret key here
    api_key = "PASTE API KEY HERE"

    # 1. Extract Text (Cached)
    with st.spinner(f"Reading {len(uploaded_files)} documents..."):
        for uploaded_file in uploaded_files:
            # Call your cached function for each individual file
            text = get_pdf_text(uploaded_files)
        

    # 2. Split Text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # 3. UI: Progress Bar
    progress_text = "Searching document for answers..."
    my_bar = st.progress(0, text=progress_text)

    try:
        # 4. Create/Load Vector Store (Cached)
        vectorstore = create_vectorstore(chunks, api_key)

        # 5. Connect to AI
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        # Smooth progress bar simulation
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)

        # 6. Get the Answer
        response = qa_chain.run(query)
        
        # UI Polish
        my_bar.empty()
        st.success("Analysis Complete!", icon="✅")
        
        st.write("### AI Response:")
        st.info(response)

    except Exception as e:
        my_bar.empty()
        st.error(f"An error occurred: {e}")