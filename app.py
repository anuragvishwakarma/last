# app.py
import streamlit as st
from agents import coordinator

st.set_page_config(page_title="Field Support Assistant", layout="wide")
st.title("🔧 Field Support & Maintenance Workflow Assistant")
st.caption("Powered by Amazon Bedrock + FAISS + Multi-Agent RAG")

query = st.text_input(
    "Ask about maintenance, safety, or next actions (e.g., 'When is next safety valve check for HCK_EQ003?')",
    placeholder="Type your query..."
)

if query:
    with st.spinner("🔍 Consulting manuals, logs, and workflow rules..."):
        answer = coordinator(query)
    st.markdown(answer)