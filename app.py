
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

st.markdown(
    "<h2 style='text-align: center;'>AI QueryBot [Sushant Kumar(VIT VELLORE, 21BCI0321)]</h2>",
    unsafe_allow_html=True
)

# ----------- 3. Enhanced prompt -----------
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert AI assistant. Answer ONLY with information from the context provided below.
    If the answer is not in the context, say "I'm sorry, the answer is not available in the provided document."
    Be concise, clear, and cite the relevant section if possible.

    <context>
    {context}
    </context>
    Question: {input}
    """
)
# ------------------------------------------

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

def is_out_of_context(answer):
    out_of_context_keywords = [
        "i'm sorry", "i don't know", "not sure", "out of context", "invalid", "there is no mention",
        " no mention"
    ]
    for keyword in out_of_context_keywords:
        if keyword in answer.lower():
            return True
    return False

def initialize_vector_db(pdf_file):
    """Initialize vector database with the uploaded PDF file"""
    if "vector_store" not in st.session_state:
        try:
            with st.spinner("Loading and processing PDF..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(pdf_file.read())
                    pdf_file_path = temp_file.name

                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name='BAAI/bge-small-en-v1.5',
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )

                # ----------- 1. Improved chunking -----------
                st.session_state.loader = PyPDFLoader(pdf_file_path)
                st.session_state.text_document_from_pdf = st.session_state.loader.load()
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,  # Larger chunks for more context
                    chunk_overlap=300  # More overlap for continuity
                )
                st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(
                    st.session_state.text_document_from_pdf
                )
                # ---------------------------------------------

                st.session_state.vector_store = Chroma.from_documents(
                    st.session_state.final_document_chunks,
                    st.session_state.embeddings
                )
                os.unlink(pdf_file_path)

            st.success("PDF has been successfully loaded and processed!")
            return True

        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return False

    return True

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload section
pdf_input_from_user = st.file_uploader("Please Upload your PDF file", type=['pdf'])

if pdf_input_from_user is not None:
    if st.button("Process PDF"):
        if initialize_vector_db(pdf_input_from_user):
            st.success("Ready to chat about your PDF!")

# Chat interface (only show if PDF is processed)
if "vector_store" in st.session_state:
    st.subheader("💬 Chat History")
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Bot:** {message['content']}")
        st.write("---")

    user_prompt = st.text_input("Enter Your Question:", key="user_input")

    if st.button('Send Message'):
        if user_prompt:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            if "vector_store" in st.session_state:
                with st.spinner("Searching for answer..."):
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    
                    # ----------- 2. Retrieve more chunks -----------
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve 5 relevant chunks
                    # ------------------------------------------------
                    
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    response = retrieval_chain.invoke({'input': user_prompt})

                    if is_out_of_context(response['answer']):
                        bot_response = "I'm sorry, the answer is not available in the provided document."
                    else:
                        bot_response = response['answer']

                    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            else:
                error_msg = "Vector database not initialized. Please restart the application."
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            st.experimental_rerun()
        else:
            st.error('Please enter your question')
else:
    st.info("Please upload a PDF file to start chatting!")
