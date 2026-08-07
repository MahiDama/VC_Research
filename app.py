import streamlit as st
from typing import List, Dict, Any, Tuple

# --- Backend Imports ---
from retriever import retrieve
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VC Analyst Co-Pilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

# --- CACHED BACKEND LOGIC ---
@st.cache_resource
def load_llm_and_rewriter():
    llm = ChatOllama(model="llama3", temperature=0)
    history_condenser_template = """Given the chat history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the chat history.
Chat History: {chat_history}
Follow-up Question: {question}
Standalone Question:"""
    history_condenser_prompt = ChatPromptTemplate.from_template(history_condenser_template)
    question_rewriter_chain = history_condenser_prompt | llm | StrOutputParser()
    return llm, question_rewriter_chain

llm, question_rewriter_chain = load_llm_and_rewriter()

def create_where_filter(query: str) -> Dict[str, Any]:
    conditions, query_lower = [], query.lower()
    known_companies = ["anthropic", "cursor"]
    found_companies = [c for c in known_companies if c in query_lower]
    if len(found_companies) == 1: conditions.append({"company": found_companies[0].title()})
    if "founder" in query_lower: conditions.append({"document_type": "founder_profile"})
    if "memo" in query_lower or "investment" in query_lower: conditions.append({"document_type": "investment_memo"})
    if not conditions: return {}
    if len(conditions) == 1: return conditions[0]
    return {"$and": conditions}

# --- UI RENDERING ---

with st.sidebar:
    st.header("Retrieved Context")
    if st.session_state.retrieved_chunks:
        for i, chunk in enumerate(st.session_state.retrieved_chunks):
            source_file = chunk['metadata'].get('source', 'Unknown Source')
            with st.expander(f"Chunk {i+1}: {source_file}", expanded=i == 0):
                st.markdown(f"**Source:** `{source_file}`"); st.info(chunk['text'])
    else: st.info("Context from your knowledge base will appear here.")

st.title("📈 VC Analyst Co-Pilot")
st.caption("Your AI-powered research assistant for venture capital.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "details" in message:
            with st.expander("Show Details"):
                st.subheader("Sources"); [st.info(s) for s in message["details"]["sources"]]
                st.subheader("Query Details")
                st.code(f"Rephrased Question: {message['details']['standalone_question']}\nFilter Applied: {message['details']['where_filter']}", language="text")

if prompt := st.chat_input("Ask about companies, founders, or market trends..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                if st.session_state.chat_history:
                    history_str = "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in st.session_state.chat_history])
                    standalone_question = question_rewriter_chain.invoke({"chat_history": history_str, "question": prompt})
                else: standalone_question = prompt
                
                where_filter = create_where_filter(standalone_question)
                retrieved_chunks = retrieve(query=standalone_question, top_k=5, where_filter=where_filter)
                st.session_state.retrieved_chunks = retrieved_chunks

                if not retrieved_chunks:
                    response_text, sources = "I could not find any relevant documents to answer your question.", []
                else:
                    context_str = "\n\n---\n\n".join([c['text'] for c in retrieved_chunks])
                    unique_sources = {c['metadata'].get('source', 'Unknown').replace('.md','').replace('_',' ').title() for c in retrieved_chunks}
                    sources = sorted(list(unique_sources))
                    rag_template = "Answer the question based ONLY on the following context.\n\nContext:\n{context}\n\nQuestion: {question}"
                    rag_prompt = ChatPromptTemplate.from_template(rag_template)
                    rag_chain = rag_prompt | llm | StrOutputParser()
                    response_text = rag_chain.invoke({"context": context_str, "question": standalone_question})
            
            st.markdown(response_text)
            response_details = {"sources": sources, "standalone_question": standalone_question, "where_filter": where_filter if where_filter else "None"}
            with st.expander("Show Details"):
                st.subheader("Sources"); [st.info(s) for s in sources] if sources else st.warning("No sources were retrieved.")
                st.subheader("Query Details"); st.code(f"Rephrased Question: {response_details['standalone_question']}\nFilter Applied: {response_details['where_filter']}", language="text")
            
            st.session_state.messages.append({"role": "assistant", "content": response_text, "details": response_details})
            st.session_state.chat_history.append((prompt, response_text))
            if len(st.session_state.chat_history) > 3: st.session_state.chat_history = st.session_state.chat_history[-3:]
        
        except Exception as e:
            # --- THIS IS THE NEW ERROR HANDLING BLOCK ---
            st.error(f"An error occurred: {e}")
            # Optionally log the full traceback to the terminal for debugging
            import traceback
            traceback.print_exc()

    st.rerun()