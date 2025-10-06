# ingest.py
import boto3
import pandas as pd
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Direct Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id="amazon.titan-embed-text-v2:0"
)

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def load_documents():
    docs = []
    # Add PDFs
    for pdf in [f for f in os.listdir("data") if f.endswith(".pdf")]:
        text = extract_pdf_text(f"data/{pdf}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append({
                "text": chunk,
                "metadata": {"source": pdf, "type": "manual"}
            })
    
    # Add CSV records as structured text
    df = pd.read_csv("data/synthetic_maintenance_records.csv", sep=";")
    for _, row in df.iterrows():
        text = (
            f"Equipment: {row['Equipment ID']} | Date: {row['Date of Inspection']} | "
            f"Pos: {row['Pos']} | Component: {row['Component/Function']} | "
            f"Type: {row['Inspection Type']} | Worker: {row['Fieldworker Name']} | "
            f"Notes: {row['Notes']}"
        )
        docs.append({
            "text": text,
            "metadata": {
                "source": "maintenance_log",
                "equipment_id": row["Equipment ID"],
                "pos": row["Pos"],
                "date": row["Date of Inspection"],
                "worker": row["Fieldworker Name"]
            }
        })
    return docs

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    
    print("Creating FAISS index...")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local("faiss_index")
    print("✅ FAISS index saved to ./faiss_index")