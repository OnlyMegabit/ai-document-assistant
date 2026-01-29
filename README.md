# AI Document Assistant

An AI-powered tool that allows users to upload PDFs and ask questions about their content using LangChain and OpenAI.

## Features
* **PDF Text Extraction:** Reads multiple PDF files simultaneously.
* **Smart Search:** Uses FAISS vector storage for fast information retrieval.
* **AI Insight:** Powered by OpenAI's GPT-3.5-Turbo.

##  Setup Instructions
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Add your OpenAI Key in the .py file.
4. Run the app: `streamlit run your_filename.py`.

##  Tech Stack
* **Streamlit** (UI Framework)
* **LangChain** (AI Orchestration)
* **OpenAI** (LLM)
* **FAISS** (Vector Database)