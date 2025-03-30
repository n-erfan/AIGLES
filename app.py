import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import os
import json

# Sidebar with contents
with st.sidebar:
    st.title("🦅 AIGLES Governance Chat App")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://https://platform.openai.com/docs/models) LLM Model
                
                ''')
    
    add_vertical_space(5)
    st.write("Made by Nafis Erfan")

# Configuration
PERSIST_DIR = "vectorstores"
PROCESSED_DOCS_FILE = "processed_docs.json"

# Initialize directories and files
os.makedirs(PERSIST_DIR, exist_ok=True)
if not os.path.exists(PROCESSED_DOCS_FILE):
    with open(PROCESSED_DOCS_FILE, "w") as f:
        json.dump([], f)

# Load processed documents registry
with open(PROCESSED_DOCS_FILE, "r") as f:
    processed_docs = json.load(f)

def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    
    store_name = pdf.name[:-4]
    persist_path = os.path.join(PERSIST_DIR, store_name)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(persist_path)

    # Update processed docs registry
    if store_name not in processed_docs:
        processed_docs.append(store_name)
        with open(PROCESSED_DOCS_FILE, "w") as f:
            json.dump(processed_docs, f)
    
    return vector_store

def load_vector_stores():
    vector_stores = []
    for store_name in processed_docs:
        persist_path = os.path.join(PERSIST_DIR, store_name)
        if os.path.exists(persist_path):
            vector_store = FAISS.load_local(
                persist_path,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
            vector_stores.append(vector_store)
    return vector_stores

def main():
    load_dotenv()
    st.header("Chat with your Governance Policy interactively.")

    # Initialize chat history and vector store in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List to store chat messages

    if "current_vector_store" not in st.session_state:
        st.session_state.current_vector_store = None

    # Upload a pdf
    pdf = st.file_uploader("Upload new policy document PDF", type='pdf')
    if pdf is not None:
        st.session_state.current_vector_store = process_pdf(pdf)
        underlined_name = '\u0332'.join(pdf.name)
        st.success(f"Processed {underlined_name} successfully! You can now ask questions from it.")

    # Display chat history from session state
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Accept user query
    query = st.chat_input("Ask questions about this document.")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Try to answer from existing knowledge
        all_vector_stores = load_vector_stores()
        if st.session_state.current_vector_store:
            all_vector_stores.append(st.session_state.current_vector_store)

        if all_vector_stores:
            # Search across all vector stores
            combined_docs = []
            for vs in all_vector_stores:
                combined_docs.extend(vs.similarity_search(query, k=2))

            if combined_docs:
                llm = ChatOpenAI(model_name='gpt-3.5-turbo')
                chain = load_qa_chain(llm, chain_type="stuff")
                
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=combined_docs, question=query)
                    print(cb)
            else:
                response = "I couldn't find relevant information in existing documents. Please upload a new PDF with this information."
        else:
             response = "No documents available. Please upload a PDF first."
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display the chat
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == '__main__':
    main()