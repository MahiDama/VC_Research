from pathlib import Path
import chromadb
from typing import Dict, Any, Optional

# --- Connect to Chroma using the persisted workspace database ---
workspace_root = Path(__file__).resolve().parent
chroma_path = workspace_root / ".chromadb"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))

# --- Load your collection WITHOUT specifying an embedding function ---
# This will make Chroma use the 'default' function that was persisted
# when the collection was created, resolving the error.
collection = chroma_client.get_collection(name="vc_research")

# --- MODIFIED retrieve function (the logic here is still correct) ---
def retrieve(query: str, top_k: int = 5, where_filter: Optional[Dict[str, Any]] = None):
    """
    Retrieves chunks from ChromaDB, now with an optional metadata filter.

    Args:
        query (str): The question to search for.
        top_k (int): The number of results to return.
        where_filter (Optional[Dict]): A dictionary for metadata filtering.
    """
    if where_filter is None:
        where_filter = {}

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter
    )

    retrieved_chunks = []
    if results and results.get("documents"):
        for i in range(len(results["documents"][0])):
            retrieved_chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "id": results["ids"][0][i]
            })

    return retrieved_chunks

# Optional test block
if __name__ == "__main__":
    print("--- Testing retrieval with a filter for company=Cursor ---")
    query = "What are the biggest risks?"
    chunks = retrieve(query, where_filter={"company": "Cursor"})

    if chunks:
        for chunk in chunks:
            print("\n---")
            print("Source Metadata:", chunk["metadata"])
            print(chunk["text"][:200] + "...")
    else:
        print("No chunks found for the test query.")