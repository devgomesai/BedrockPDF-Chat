import boto3
import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# AWS Setup
region_name = "us-east-1"
s3_client = boto3.client("s3", region_name=region_name)
bedrock_client = boto3.client("bedrock-runtime", region_name=region_name)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# Environment variable for bucket
BUCKET_NAME = os.getenv("BUCKET_NAME")
if not BUCKET_NAME:
    st.error("❌ Environment variable `BUCKET_NAME` is not set!")
    st.stop()

# Generate UUID
def get_unique_id():
    return str(uuid.uuid4())

# Split text using recursive splitter
def split_text(pages, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

# Create FAISS vector store and upload to S3
def create_vector_store(request_id, documents):
    try:
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        file_name = f"{request_id}.bin"
        folder_path = "/tmp"
        full_path = os.path.join(folder_path, file_name)

        vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

        # Upload files to S3
        s3_client.upload_file(Filename=full_path + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
        s3_client.upload_file(Filename=full_path + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

        return True
    except Exception as e:
        st.error(f"❌ Failed to create vector store or upload to S3: {e}")
        return False

# Streamlit UI
def main():
    st.title("📄 Admin Panel: Chat with PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        request_id = get_unique_id()
        st.info(f"Request ID: `{request_id}`")

        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()
        st.success(f"✅ Loaded {len(pages)} pages from PDF.")

        # Split text into chunks
        splitted_docs = split_text(pages)
        st.write(f"🔹 Total Chunks: {len(splitted_docs)}")
        st.code(splitted_docs[0].page_content[:500], language="markdown")  # Show snippet of first chunk

        st.write("⏳ Creating the Vector Store...")
        if create_vector_store(request_id, splitted_docs):
            st.success("🎉 PDF processed and vector store uploaded successfully!")
        else:
            st.error("❌ Something went wrong. Check logs.")

if __name__ == "__main__":
    main()
