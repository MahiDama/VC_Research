from retriever import retrieve
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any, Tuple

# --- CONFIGURATION ---
LLM_MODEL = "llama3"
CHAT_HISTORY_LENGTH = 6

# 1. Initialize the Language Model (LLM)
llm = ChatOllama(model=LLM_MODEL, temperature=0)

# 2. --- REWRITTEN and CORRECTED create_where_filter function ---
def create_where_filter(query: str) -> Dict[str, Any]:
    """
    Parses the query to create a ChromaDB metadata filter.
    If multiple company entities are found, it applies NO company filter,
    allowing the vector search to retrieve documents for all of them.
    """
    conditions = []
    query_lower = query.lower()

    # Define known filterable entities
    known_companies = ["anthropic", "cursor"]
    found_companies = [company for company in known_companies if company in query_lower]

    # If only one company is found, add a filter for it.
    # If multiple are found (comparison), add no filter for company.
    if len(found_companies) == 1:
        # Use .title() to match the casing in your metadata, e.g., "Anthropic"
        conditions.append({"company": found_companies[0].title()})

    # You can still filter by document type even in a comparison
    if "founder" in query_lower:
        conditions.append({"document_type": "founder_profile"})
    if "memo" in query_lower or "investment" in query_lower:
        conditions.append({"document_type": "investment_memo"})

    if not conditions:
        print("Generated filter: {}")
        return {}
    if len(conditions) == 1:
        print(f"Generated filter: {conditions[0]}")
        return conditions[0]
    
    filter_dict = {"$and": conditions}
    print(f"Generated filter: {filter_dict}")
    return filter_dict

# --- NO CHANGES TO THE REST OF THE FILE ---

# 3. Prompt Template to Condense Chat History and a New Question
history_condenser_template = """
Given the chat history and a follow-up question, rephrase the follow-up question
to be a standalone question that can be understood without the chat history.

Chat History:
{chat_history}

Follow-up Question: {question}
Standalone Question:
"""
history_condenser_prompt = ChatPromptTemplate.from_template(history_condenser_template)
question_rewriter_chain = history_condenser_prompt | llm | StrOutputParser()

# 4. Main RAG Prompt Template
rag_template = """
You are an expert investment analyst assistant.
Answer the question based ONLY on the following context.
If the context does not contain the answer, state that clearly.

Context:
{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

# 5. Main RAG Logic
def get_rag_response(question: str, chat_history: List[Tuple[str, str]]):
    if chat_history:
        history_str = "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in chat_history])
        standalone_question = question_rewriter_chain.invoke({
            "chat_history": history_str,
            "question": question
        })
        print(f"Standalone Question: {standalone_question}")
    else:
        standalone_question = question

    where_filter = create_where_filter(standalone_question)

    retrieved_chunks = retrieve(
        query=standalone_question,
        top_k=5,  # Increased top_k to get more context for comparisons
        where_filter=where_filter
    )

    if not retrieved_chunks:
        return {"answer": "I could not find any relevant documents to answer your question.", "sources": []}

    context_str = "\n\n---\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    unique_sources = set()
    for chunk in retrieved_chunks:
        source_file = chunk['metadata'].get('source', 'Unknown Source')
        display_name = source_file.replace('.md', '').replace('_', ' ').title()
        unique_sources.add(display_name)

    rag_chain = RunnablePassthrough.assign(context=lambda x: context_str) | rag_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"question": standalone_question})
    return {"answer": answer, "sources": list(unique_sources)}

# 6. Interactive Chat Loop
if __name__ == "__main__":
    print("Welcome to the RAG-powered Chat Analyst. Type 'exit' to end.")
    chat_history = []
    while True:
        user_input = input("\n> ")
        if user_input.lower() == 'exit':
            break
        response = get_rag_response(user_input, chat_history)
        print("\nAssistant:")
        print(response["answer"])
        if response["sources"]:
            print("\nSources:")
            for source in response["sources"]:
                print(f"- {source}")
        chat_history.append((user_input, response["answer"]))
        if len(chat_history) > CHAT_HISTORY_LENGTH // 2:
            chat_history = chat_history[-(CHAT_HISTORY_LENGTH // 2):]