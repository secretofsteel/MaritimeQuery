from app.vector_store import get_qdrant_client, ensure_collection
try:
    client = get_qdrant_client()
    name = ensure_collection(client)
    collection_info = client.get_collection(name)
    print(f"Collection Name: {name}")
    print(f"Vectors Count: {collection_info.vectors_count}")
    print(f"Status: {collection_info.status}")
    print(f"Vector Config: {collection_info.config.params.vectors}")
except Exception as e:
    print(f"Verification Failed: {e}")
