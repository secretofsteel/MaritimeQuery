"""
Deduplicate ChromaDB by chunk_id (keep only latest), then recover nodes.
"""

import chromadb
from pathlib import Path
import pickle
from llama_index.core.schema import TextNode
from collections import defaultdict

# Connect to ChromaDB
chroma_path = Path("C:/MADASS/data/chroma_db")
client = chromadb.PersistentClient(path=str(chroma_path))
collection = client.get_collection("maritime_docs")

print(f"ChromaDB has {collection.count()} vectors before deduplication")

# Get all documents
result = collection.get(
    include=['documents', 'metadatas', 'embeddings']
)

print(f"Retrieved {len(result['ids'])} entries")

# Group by source + chunk index to find duplicates
entries_by_key = defaultdict(list)

for i, doc_id in enumerate(result['ids']):
    metadata = result['metadatas'][i]
    
    # Create unique key from source + chunk identifier
    source = metadata.get('source', '')
    
    # Try to get a chunk identifier
    chunk_id = metadata.get('chunk_id') or metadata.get('node_id') or doc_id
    
    # Key is source:chunk_id
    key = f"{source}:{chunk_id}"
    
    entries_by_key[key].append({
        'id': doc_id,
        'text': result['documents'][i],
        'metadata': metadata,
        'embedding': result['embeddings'][i] if result['embeddings'] is not None else None,
        'index': i
    })

print(f"Found {len(entries_by_key)} unique chunk keys")

# For each key, keep only the first entry (or you could use last if you want newest)
unique_entries = []
duplicate_ids = []

for key, entries in entries_by_key.items():
    if len(entries) > 1:
        print(f"  Duplicate: {key} has {len(entries)} copies")
        # Keep first, mark rest for deletion
        unique_entries.append(entries[0])
        for entry in entries[1:]:
            duplicate_ids.append(entry['id'])
    else:
        unique_entries.append(entries[0])

print(f"\nWill keep {len(unique_entries)} unique entries")
print(f"Will delete {len(duplicate_ids)} duplicates")

if duplicate_ids:
    print("\nDeleting duplicates from ChromaDB...")
    collection.delete(ids=duplicate_ids)
    print(f"✅ Deleted {len(duplicate_ids)} duplicate vectors")

print(f"\nChromaDB now has {collection.count()} vectors")

# Now reconstruct nodes from the unique entries
nodes = []

for entry in unique_entries:
    node = TextNode(
        id_=entry['id'],
        text=entry['text'],
        metadata=entry['metadata'],
        embedding=entry['embedding']
    )
    nodes.append(node)

print(f"Reconstructed {len(nodes)} nodes")

# Save to pickle
cache_path = Path("C:/MADASS/data/cache/nodes_cache.pkl")
with open(cache_path, 'wb') as f:
    pickle.dump(nodes, f)

print(f"\n✅ Saved {len(nodes)} nodes to {cache_path}")
print("✅ ChromaDB deduplicated")
print("✅ Restart Streamlit to load the recovered nodes")