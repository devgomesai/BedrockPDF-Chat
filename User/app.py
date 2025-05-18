import os
import uuid
import boto3
import datetime
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
region_name = "us-east-1"
BUCKET_NAME = os.getenv("BUCKET_NAME")
folder_path = "/tmp"

# Boto3 Clients
s3_client = boto3.client("s3", region_name=region_name)
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=region_name)
dynamodb = boto3.resource("dynamodb", region_name=region_name)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock_client
)

# Load FAISS index from S3
def load_index():
    try:
        s3_client.download_file(BUCKET_NAME, "my_faiss.faiss", os.path.join(folder_path, "my_faiss.faiss"))
        s3_client.download_file(BUCKET_NAME, "my_faiss.pkl", os.path.join(folder_path, "my_faiss.pkl"))
        return True
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {e}")
        return False

# Get Bedrock LLM
def get_llm():
    return Bedrock(
        model_id="anthropic.claude-v2:1",
        client=bedrock_client,
        model_kwargs={"max_tokens_to_sample": 512}
    )

# RAG response
def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to provide a concise answer to the question.
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    response = qa({"query": question})
    return response["result"]

# Save to DynamoDB
def save_chat_to_dynamo(question, answer):
    try:
        table = dynamodb.Table("ChatHistory")
        table.put_item(
            Item={
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": answer,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        st.error(f"❌ Error saving to DynamoDB: {e}")

# Streamlit App
def main():
    st.title("💬 Chat with PDF (Client)")
    st.subheader("Powered by Bedrock, FAISS, and RAG")

    if not BUCKET_NAME:
        st.error("❌ Environment variable `BUCKET_NAME` not set.")
        return

    with st.spinner("Loading FAISS index from S3..."):
        if not load_index():
            return

    st.write("📂 Loaded files from `/tmp`:")
    st.code("\n".join(os.listdir(folder_path)))

    # Load FAISS
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.success("✅ FAISS Index is ready!")
    question = st.text_input("❓ Ask a question about your PDF")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Querying Bedrock model..."):
            llm = get_llm()
            answer = get_response(llm, faiss_index, question)
            st.markdown("### 📌 Answer:")
            st.write(answer)
            save_chat_to_dynamo(question, answer)
            st.success("✅ Answer saved to DynamoDB")

if __name__ == "__main__":
    main()
