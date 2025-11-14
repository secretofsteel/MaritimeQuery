# Hierarchical Retrieval Integration Example

## Quick Start

The hierarchical retrieval system is designed to work **automatically** with minimal integration. Here's how to use it:

## Option 1: Standalone Usage (For Testing)

```python
from app.config import AppConfig
from app.state import AppState
from app.query import classify_retrieval_strategy, retrieve_hierarchical, format_hierarchical_context
from app.indexing import load_document_trees

# Initialize
app_state = AppState()
app_state.ensure_index_loaded()

# Your query
query = "What is the ballast discharge procedure?"

# Step 1: Classify the query strategy
strategy = classify_retrieval_strategy(query)
print(f"Query strategy: {strategy}")

# Step 2: If section_level, use hierarchical retrieval
if strategy == "section_level":
    # Retrieve complete sections
    nodes = retrieve_hierarchical(query, app_state, top_sections=2)
    print(f"Retrieved {len(nodes)} chunks from {2} sections")

    # Load trees for formatting
    config = AppConfig.get()
    trees_path = config.paths.cache_dir / "document_trees.json"
    trees = load_document_trees(trees_path)

    # Format as hierarchical markdown
    context = format_hierarchical_context(nodes, trees)
    print("\nHierarchical Context:")
    print(context)
else:
    # Use existing chunk-level retrieval
    print("Using chunk-level retrieval for specific facts")
```

## Option 2: Integration into Existing Query Flow

If you want to integrate hierarchical retrieval into the existing `query_with_confidence` function, add this logic:

```python
from app.query import query_with_confidence, classify_retrieval_strategy, retrieve_hierarchical
from app.state import AppState

def query_with_hierarchical(
    app_state: AppState,
    query_text: str,
    enable_hierarchical: bool = True,
    **kwargs
):
    """
    Enhanced query function with hierarchical retrieval support.

    Args:
        app_state: Application state
        query_text: User query
        enable_hierarchical: Enable hierarchical retrieval for procedural queries
        **kwargs: Additional arguments passed to query_with_confidence

    Returns:
        Query result dict with answer, sources, confidence, etc.
    """
    # Classify strategy if hierarchical is enabled
    if enable_hierarchical:
        strategy = classify_retrieval_strategy(query_text)

        if strategy == "section_level":
            # Use hierarchical retrieval
            nodes = retrieve_hierarchical(query_text, app_state, top_sections=2)

            # Check if we got enough context (fallback to chunks if < 500 tokens)
            total_text = "".join([n.node.text for n in nodes])
            if len(total_text) < 500:
                # Fallback to chunk-level
                return query_with_confidence(
                    app_state=app_state,
                    query_text=query_text,
                    **kwargs
                )

            # Build context from hierarchical sections
            from app.indexing import load_document_trees
            from app.query import format_hierarchical_context

            config = AppConfig.get()
            trees_path = config.paths.cache_dir / "document_trees.json"
            trees = load_document_trees(trees_path)

            hierarchical_context = format_hierarchical_context(nodes, trees)

            # You would need to modify query_with_confidence to accept pre-built context
            # For now, this is a conceptual example
            print(f"✅ Using hierarchical retrieval ({len(nodes)} chunks)")

    # Use existing query_with_confidence
    return query_with_confidence(
        app_state=app_state,
        query_text=query_text,
        **kwargs
    )
```

## Option 3: Minimal Integration (Recommended for Production)

For production use, the simplest approach is to make hierarchical retrieval **opt-in** via a parameter:

```python
from app.state import AppState
from app.query import query_with_confidence, classify_retrieval_strategy, retrieve_hierarchical

# In your main query handler
def handle_query(query_text: str, use_hierarchical: bool = True):
    """
    Handle user query with optional hierarchical retrieval.

    Args:
        query_text: User question
        use_hierarchical: Enable hierarchical retrieval for procedures
    """
    app_state = AppState()
    app_state.ensure_index_loaded()

    # Optional: Override retrieval based on strategy
    if use_hierarchical:
        strategy = classify_retrieval_strategy(query_text)

        if strategy == "section_level":
            print("🔍 Detected procedural query - using hierarchical retrieval")
            # You can pass custom retrievers or pre-fetched nodes
            # to query_with_confidence here

    # Standard query (may incorporate hierarchical internally)
    result = query_with_confidence(
        app_state=app_state,
        query_text=query_text,
        retriever_type="hybrid",
        use_conversation_context=True
    )

    return result
```

## Complete Example with Comparison

Here's a complete example showing chunk-level vs section-level retrieval:

```python
from app.config import AppConfig
from app.state import AppState
from app.query import (
    classify_retrieval_strategy,
    retrieve_hierarchical,
    query_with_confidence,
)
from app.indexing import load_document_trees

def compare_retrieval_strategies(query: str):
    """Compare chunk-level vs hierarchical retrieval."""

    # Initialize
    app_state = AppState()
    app_state.ensure_index_loaded()

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    # Classify
    strategy = classify_retrieval_strategy(query)
    print(f"📊 Detected Strategy: {strategy}\n")

    # Method 1: Chunk-Level (Standard)
    print("🔍 Method 1: Chunk-Level Retrieval")
    print("-" * 40)

    app_state.ensure_retrievers()
    vector_results = app_state.vector_retriever.retrieve(query)
    bm25_results = app_state.bm25_retriever.retrieve(query)

    from app.query import reciprocal_rank_fusion
    chunk_nodes = reciprocal_rank_fusion(vector_results, bm25_results, k=60, top_k=10)

    print(f"Retrieved: {len(chunk_nodes)} chunks")
    for i, node in enumerate(chunk_nodes[:3], 1):
        print(f"  {i}. {node.node.metadata.get('source')} - {node.node.metadata.get('section')}")

    # Method 2: Hierarchical (if procedural)
    if strategy == "section_level":
        print("\n🔍 Method 2: Hierarchical Retrieval")
        print("-" * 40)

        hierarchical_nodes = retrieve_hierarchical(query, app_state, top_sections=2)

        print(f"Retrieved: {len(hierarchical_nodes)} chunks from complete sections")

        # Group by section
        from collections import defaultdict
        by_section = defaultdict(list)
        for node in hierarchical_nodes:
            section_id = node.node.metadata.get('section_id', 'unknown')
            by_section[section_id].append(node)

        print(f"Sections covered: {len(by_section)}")
        for section_id, nodes in by_section.items():
            print(f"  - Section {section_id}: {len(nodes)} chunks")

        # Show formatted context
        config = AppConfig.get()
        trees_path = config.paths.cache_dir / "document_trees.json"
        trees = load_document_trees(trees_path)

        from app.query import format_hierarchical_context
        formatted = format_hierarchical_context(hierarchical_nodes, trees)

        print("\n📄 Formatted Hierarchical Context (preview):")
        print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
    else:
        print("\n⚠️  Not a procedural query - hierarchical retrieval not applicable")

    print(f"\n{'='*80}\n")

# Example usage
if __name__ == "__main__":
    # Test queries
    queries = [
        "What is the ballast discharge procedure?",  # Procedural
        "What is the discharge temperature limit?",  # Specific fact
        "How do I handle ice navigation?",           # Procedural
    ]

    for query in queries:
        compare_retrieval_strategies(query)
```

## Integration Checklist

Before using hierarchical retrieval in production:

- [ ] **Rebuild the index** with hierarchical support
  - Via UI: Admin → "Full Rebuild (Parallel)"
  - Via code: `build_index_from_library_parallel()`

- [ ] **Verify document trees exist**
  ```python
  from pathlib import Path
  trees_path = Path("data/document_trees.json")
  assert trees_path.exists(), "Trees not found! Rebuild index."
  ```

- [ ] **Test strategy classification**
  ```python
  from app.query import classify_retrieval_strategy

  assert classify_retrieval_strategy("What is the procedure?") == "section_level"
  assert classify_retrieval_strategy("What is the limit?") == "chunk_level"
  ```

- [ ] **Run the test suite**
  ```bash
  python test_hierarchical.py
  ```

- [ ] **Test with real queries** from your domain

- [ ] **Monitor performance** (hierarchical should be similar or faster)

- [ ] **Set up fallback logic** for edge cases (missing trees, empty sections, etc.)

## Production Configuration

For production deployment, consider these settings:

```python
# config.py or settings
HIERARCHICAL_RETRIEVAL_CONFIG = {
    "enabled": True,
    "top_sections": 2,  # Max sections to retrieve (token budget)
    "min_section_tokens": 500,  # Fallback to chunks if section < this
    "max_section_tokens": 10000,  # Skip sections larger than this
    "strategy_classification_timeout": 2.0,  # Seconds
    "fallback_on_error": True,  # Use chunk-level if hierarchical fails
}
```

## Monitoring and Logging

Add logging to track retrieval strategy usage:

```python
import logging

logger = logging.getLogger(__name__)

def query_with_monitoring(query_text: str):
    strategy = classify_retrieval_strategy(query_text)

    logger.info(
        "Query strategy",
        extra={
            "query": query_text[:100],
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
        }
    )

    if strategy == "section_level":
        nodes = retrieve_hierarchical(query_text, app_state)
        logger.info(
            "Hierarchical retrieval",
            extra={
                "chunks_retrieved": len(nodes),
                "sections_covered": len(set(n.node.metadata.get('section_id') for n in nodes)),
            }
        )

    # Continue with query...
```

## Next Steps

1. **Rebuild your index** to generate document trees
2. **Run the test suite** to validate the implementation
3. **Test with domain-specific queries** from your maritime documents
4. **Monitor performance** and tune parameters as needed
5. **Integrate into your application** using one of the patterns above

For complete details, see [HIERARCHICAL_RETRIEVAL.md](HIERARCHICAL_RETRIEVAL.md).
