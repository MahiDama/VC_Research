# VC Copilot

VC Copilot is a Streamlit-based research assistant for venture capital and startup analysis. It uses a retrieval-augmented generation (RAG) workflow to search a local knowledge base of company notes and answer questions about founders, market trends, investment memos, and related context.

## Features

- Chat-based interface for VC research questions
- Retrieval from a persisted ChromaDB vector store
- Context-aware prompts using local LLM responses from Ollama
- Sidebar view of retrieved context chunks for transparency
- Designed for quick exploration of startup and market knowledge

## Project Structure

- `app.py` — main Streamlit application
- `retriever.py` — ChromaDB retrieval logic
- `companies/` — source markdown documents for research context
- `chroma_db/` and `.chromadb/` — persisted vector database files
- `rag_query.py` — optional helper script for querying the knowledge base

## Prerequisites

Before running the app, make sure you have:

- Python 3.10+ installed
- Ollama installed and running locally
- A model available in Ollama (the app currently uses `llama3`)

If needed, pull the model with:

```bash
ollama pull llama3
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install streamlit chromadb langchain-ollama langchain-core
```

## Running the App

From the project root, start the app with:

```bash
streamlit run app.py
```

The app will open in your browser and let you ask questions about the documents stored in the workspace.

## Notes

- The app expects a Chroma collection named `vc_research` to already exist in the local database.
- The knowledge base is populated from the markdown files under `companies/` and the persisted vector database.
- If you encounter issues with Ollama, verify that the service is running and that the selected model is available.

## License

This project is provided for local research and experimentation purposes.
