import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

##load the groq api
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    """ 
    Answer the questions based on the context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
 
 """
)

import os

def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector database from pdf..."):
            # 1. Embeddings
            st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")

            # 2. Load PDFs  ✅ USE ABSOLUTE / RAW PATH
            data_path = r"E:\Lang chain projects\1-Q&A chatbot\1.3-RAG Document QA\research_papers"
            st.session_state.loader = PyPDFDirectoryLoader(data_path)
            st.session_state.docs = st.session_state.loader.load()

            # Debug: show how many docs loaded
            st.write("Docs loaded:", len(st.session_state.docs))

            if not st.session_state.docs:
                st.error(f"No PDFs found in folder: {data_path}")
                return

            # 3. Split into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )

            # Debug: show how many chunks created
            st.write("Chunks created:", len(st.session_state.final_documents))

            if not st.session_state.final_documents:
                st.error("No text chunks created from the documents. PDFs may be scanned or empty.")
                return

            # 4. Create FAISS index (this is where error was before)
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )

        st.success("Database is Ready! ✅")



        
        
## UI streamlit
st.title("RAG QA-Reasearch papers")

user_prompt = st.text_input("Enter your query from research paper")

if st.button("Create Document Embedding"):
    create_vector_embedding()



import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Create/ Refresh document embeddinh' ")
    else:
       ##build chains
       document_chain = create_stuff_documents_chain(llm,prompt)
       retriever= st.session_state.vectors.as_retriever()
       retrieval_chain = create_retrieval_chain(retriever,document_chain)

       start = time.process_time()
       response = retrieval_chain.invoke({'input':user_prompt})
       elapsed=time.process_time()-start
       st.subheader("Answer")
       st.write(response['answer'])
       st.caption(f"Response time: {elapsed:.2f} seconds")
    
    # document similarity search(show retrieved chunks)

       with st.expander("Document similarity Search (retrieved chunks)"):
        for i , doc in enumerate(response["context"],start=1):
            source = doc.metadata.get("source","Unknown source")
            page = doc.metadata.get("page","Unknown page")

            st.markdown(f"**Chunk {i} — Source: ** `{source}`,**Page:** `{page}`")
            st.write(doc.page_content)
            st.markdown("---")

    
