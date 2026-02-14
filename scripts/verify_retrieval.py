import sys
import logging
from app.state import AppState
from app.config import AppConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_retrieval():
    logger.info("Initializing AppState...")
    config = AppConfig.get()
    state = AppState(tenant_id="default")
    
    # Load index (mocks loading, actually just sets up retrievers mostly)
    # in Streamlit app, ensure_index_loaded is called.
    # verification script might need to mock some things if Qdrant isn't running? 
    # But Qdrant IS running.
    
    logger.info("Ensuring retrievers...")
    # This should trigger ensure_retrievers -> get_qdrant_client -> TenantAwareVectorRetriever
    if not state.ensure_index_loaded():
        logger.error("Failed to load index/retrievers!")
        return

    retriever = state.vector_retriever
    if not retriever:
        logger.error("Vector retriever is None!")
        return

    logger.info(f"Retriever type: {type(retriever)}")
    
    # Test Query
    query = "procedure"
    logger.info(f"Querying for: '{query}'")
    
    from llama_index.core import QueryBundle
    nodes = retriever.retrieve(QueryBundle(query))
    
    logger.info(f"Retrieved {len(nodes)} nodes.")
    
    if not nodes:
        logger.warning("No nodes retrieved. Index might be empty or connection failed.")
        return

    for i, node in enumerate(nodes[:3]):
        logger.info(f"[{i}] Score: {node.score:.4f}")
        logger.info(f"    Text: {node.node.get_content()[:100]}...")
        logger.info(f"    Metadata: {node.node.metadata}")
        
        # Verify Score Range
        if not (0.0 <= node.score <= 1.0):
             logger.error(f"Score {node.score} out of range [0, 1]")

    # Verify Filtering
    logger.info("Testing doc_type filter (REGULATION)...")
    filtered_nodes = retriever.retrieve_filtered(query, doc_type_filter=["REGULATION"])
    logger.info(f"Retrieved {len(filtered_nodes)} filtered nodes.")
    for node in filtered_nodes:
        dtype = node.node.metadata.get("doc_type", "UNKNOWN")
        if dtype != "REGULATION":
            logger.error(f"Filter failed! Found doc_type: {dtype}")

    logger.info("Verification Complete.")

if __name__ == "__main__":
    verify_retrieval()
