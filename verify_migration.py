import sys
from app.indexing import build_index_from_library_parallel
from app.vector_store import get_qdrant_client, ensure_collection
from app.config import AppConfig

def verify_migration():
    print("Starting verification...")
    
    # 1. Run rebuild (this uses Qdrant now)
    print("Running build_index_from_library_parallel...")
    nodes, index, report = build_index_from_library_parallel(clear_gemini_cache=False)
    
    print(f"Rebuild complete. Processed {len(nodes)} nodes.")
    print(f"Report: {report}")
    
    # 2. Check Qdrant directly
    print("Checking Qdrant collection...")
    client = get_qdrant_client()
    name = ensure_collection(client)
    info = client.get_collection(name)
    
    print(f"Collection: {name}")
    print(f"Points count: {info.points_count}")
    print(f"Vectors count: {info.vectors_count}")
    
    if info.points_count > 0:
        print("SUCCESS: Qdrant has points.")
        # Check payload of one point
        points, _ = client.scroll(collection_name=name, limit=1, with_payload=True)
        if points:
            print(f"Sample payload: {points[0].payload}")
            if "tenant_id" in points[0].payload and "source" in points[0].payload:
                print("SUCCESS: Payload has tenant_id and source.")
            else:
                print("FAILURE: Payload missing required fields.")
    else:
        print("WARNING: Qdrant is empty (might be expected if no docs).")

if __name__ == "__main__":
    verify_migration()
