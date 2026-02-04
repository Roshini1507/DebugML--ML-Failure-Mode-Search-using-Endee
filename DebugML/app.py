
import streamlit as st
from search import search_failures
from rag import generate_rag_answer

st.title("🧠 ML Failure Mode Search")

query = st.text_input("Describe your ML issue")

if st.button("Search") and query:
    results = search_failures(query)

    st.subheader("🔍 Similar Past Failures")
    for r in results:
        st.write(r["metadata"]["text"])
        st.caption("Fix: " + r["metadata"]["fix"])

    st.subheader("🤖 RAG Insight")
    st.write(generate_rag_answer(query, results))
