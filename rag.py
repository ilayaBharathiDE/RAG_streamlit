import streamlit as st
from PyPDF2 import PdfReader  # for PDF read
from langchain.text_splitter import RecursiveCharacterTextSplitter  # to split multiple chunks
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
import time
import torch

# Add custom CSS for styling
st.markdown("""
    <style>
        .response-container {
            background-color: #e8f0fe;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #4CAF50;
            color: #333;
        }
        .left-aligned-message {
            text-align: left;  /* Align text to the left */
            margin-top: 30px;   /* Adjust the space above */
            margin-left: 90px;   /* Adjusted left margin for better alignment */
            font-size: 20px;     /* Increase font size */
            color: black;        /* Ensure color is black */
        }
        .sub-header-message {
            text-align: left;  /* Align text to the left */
            margin-top: 0px;   /* Adjust the space above */
            margin-left: 260px;   /* Adjusted left margin for better alignment */
            font-size: 18px;     /* Increase font size */
            color: black;        /* Ensure color is black */
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("✨ Ilaya's RAG : PDF Assistance ✨")
    st.markdown("<div class='sub-header-message '> 🤖 (AI Powered) </div>", unsafe_allow_html=True)
    
    # Sidebar for file upload
    st.sidebar.header("Upload PDF")
    pdf = st.sidebar.file_uploader("📥 Drag and drop your PDF file here!", type='pdf')
    # Additional information in the sidebar
    st.sidebar.subheader("Model Information")
    st.sidebar.write("🦙 **LLM**: Llama2")
    st.sidebar.write("🔍 **Embedding Model**: HuggingFace")
    st.sidebar.write("📦 **Vector DB**: FAISS")

    # Check if a PDF is uploaded
    if pdf is not None:
        st.sidebar.success("Your file has been successfully uploaded!")  # Success message after upload
        
        # Show the query input field after upload
        query = st.text_input("💬 Ask anything about your PDF", key="query_input")

        if query:  # Proceed only if there is a query
            with st.spinner("Processing your PDF..."):
                reader = PdfReader(pdf)

               # extracting the text from PDF
                text = ""                               
                for page in reader.pages:
                    text += page.extract_text()

               #Spliting the extracted text from PDF into multiple chuncks
                split_chunks = RecursiveCharacterTextSplitter(       
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )

                chunks = split_chunks.split_text(text=text)

                # Check if GPU is available
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Initialize embeddings with only model name
                # Using Hugging face to convert chunks to embedded value
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 

                # Print whether GPU is being used
                if device == 'cuda':
                    st.write("🟢 Using GPU for processing!")
                else:
                    st.write("🔴 Using CPU for processing.")

                # Use FAISS to create vector store
                start_time = time.time()  # Start timing the response

                # using FAISS to store embedded data
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)

                # Here we are starting Q&A
                # our large language model is meta's llama2
                docs = vector_store.similarity_search(query=query, k=2)  # Reduced k for quicker response
                llm = Ollama(model='llama2', temperature=0.5)
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)
                end_time = time.time()  # End timing the response

                response_time = end_time - start_time  # Calculate response time

                # Display the response with custom styling
                st.markdown(f"<div class='response-container'>{response}</div>", unsafe_allow_html=True)
                st.write(f"⏱️ Response Time: {response_time:.2f} seconds")
    
    else:
        # Left-aligned message for no PDF uploaded with emojis
        st.markdown("<div class='left-aligned-message'>📄 Please upload a PDF file to start asking questions! 📝</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
