"""Test script for hierarchical retrieval system."""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import AppConfig
from app.extraction import build_document_tree, gemini_extract_record
from app.indexing import (
    build_document_tree,
    load_document_trees,
    save_document_trees,
    map_chunks_to_tree_sections,
)
from app.query import classify_retrieval_strategy, retrieve_hierarchical
from app.state import AppState
from app.logger import LOGGER


def test_tree_building():
    """Test document tree building from a sample document."""
    print("\n" + "=" * 80)
    print("TEST 1: Document Tree Building")
    print("=" * 80)

    # Check if there are any documents to test with
    config = AppConfig.get()
    docs_path = config.paths.docs_path

    if not docs_path.exists() or not list(docs_path.glob("*")):
        print("❌ No documents found in docs_path. Please add documents first.")
        return False

    # Get first document
    test_doc = next(docs_path.glob("*"), None)
    if not test_doc:
        print("❌ No test document found")
        return False

    print(f"📄 Testing with document: {test_doc.name}")

    # Extract with Gemini (or load from cache)
    try:
        meta = gemini_extract_record(test_doc)
        if "parse_error" in meta:
            print(f"❌ Extraction error: {meta.get('parse_error')}")
            return False

        # Build tree
        doc_id = test_doc.stem
        tree = build_document_tree(meta, doc_id)

        print(f"\n✅ Tree built successfully:")
        print(f"   Doc ID: {tree['doc_id']}")
        print(f"   Doc Type: {tree['doc_type']}")
        print(f"   Title: {tree['title']}")
        print(f"   Sections: {len(tree['sections'])} top-level sections")

        # Print tree structure
        def print_section(section, indent=0):
            prefix = "  " * indent
            section_id = section.get("section_id", "?")
            title = section.get("title", "Untitled")
            level = section.get("level", 0)
            chunk_count = len(section.get("chunk_ids", []))
            child_count = len(section.get("children", []))

            print(f"{prefix}├─ [{section_id}] {title} (level={level}, chunks={chunk_count}, children={child_count})")

            for child in section.get("children", []):
                print_section(child, indent + 1)

        print("\n📊 Tree Structure:")
        for section in tree["sections"]:
            print_section(section)

        return True

    except Exception as exc:
        print(f"❌ Error during tree building: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_classification():
    """Test query strategy classification."""
    print("\n" + "=" * 80)
    print("TEST 2: Query Strategy Classification")
    print("=" * 80)

    test_queries = [
        ("What is the discharge temperature limit?", "chunk_level"),
        ("What is the ballast discharge procedure?", "section_level"),
        ("How do I handle ice navigation?", "section_level"),
        ("What PPE is required for deck work?", "chunk_level"),
        ("What are the steps for drug testing?", "section_level"),
    ]

    results = []
    for query, expected in test_queries:
        try:
            strategy = classify_retrieval_strategy(query)
            match = "✅" if strategy == expected else "❌"
            print(f"{match} '{query}' → {strategy} (expected: {expected})")
            results.append(strategy == expected)
        except Exception as exc:
            print(f"❌ Error classifying '{query}': {exc}")
            results.append(False)

    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")

    return success_rate >= 60  # 60% threshold


def test_hierarchical_retrieval():
    """Test hierarchical retrieval with a procedural query."""
    print("\n" + "=" * 80)
    print("TEST 3: Hierarchical Retrieval")
    print("=" * 80)

    # Load app state
    app_state = AppState()

    # Try to load cached index
    if not app_state.ensure_index_loaded():
        print("❌ No index found. Please build the index first using the main app.")
        print("   Run: python -m streamlit run app/main.py")
        print("   Then use the Admin panel to build/sync the index.")
        return False

    print(f"✅ Loaded index with {len(app_state.nodes)} chunks")

    # Check if document trees exist
    config = AppConfig.get()
    trees_path = config.paths.cache_dir / "document_trees.json"

    if not trees_path.exists():
        print("❌ No document trees found. Please rebuild the index with hierarchical support.")
        print("   The index needs to be rebuilt after implementing hierarchical retrieval.")
        return False

    trees = load_document_trees(trees_path)
    print(f"✅ Loaded {len(trees)} document trees")

    # Test with a procedural query
    test_query = "What is the procedure for ballast discharge?"

    try:
        print(f"\n🔍 Testing query: '{test_query}'")

        # Classify strategy
        strategy = classify_retrieval_strategy(test_query)
        print(f"   Strategy: {strategy}")

        if strategy == "section_level":
            # Use hierarchical retrieval
            nodes = retrieve_hierarchical(test_query, app_state, top_sections=2)
            print(f"\n✅ Retrieved {len(nodes)} chunks via hierarchical retrieval")

            # Check if results have section_ids
            chunks_with_sections = sum(
                1 for n in nodes if n.node.metadata.get("section_id")
            )
            print(f"   Chunks with section_ids: {chunks_with_sections}/{len(nodes)}")

            if len(nodes) > 0:
                # Show sample
                print("\n📄 Sample chunk metadata:")
                sample = nodes[0]
                metadata = sample.node.metadata
                print(f"   Source: {metadata.get('source', 'Unknown')}")
                print(f"   Section: {metadata.get('section', 'N/A')}")
                print(f"   Section ID: {metadata.get('section_id', 'N/A')}")
                print(f"   Score: {sample.score:.3f}")

                return True
            else:
                print("❌ No chunks retrieved")
                return False
        else:
            print(f"⚠️  Strategy was {strategy}, not section_level. Cannot test hierarchical retrieval.")
            return False

    except Exception as exc:
        print(f"❌ Error during hierarchical retrieval: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_tree_persistence():
    """Test saving and loading document trees."""
    print("\n" + "=" * 80)
    print("TEST 4: Tree Persistence")
    print("=" * 80)

    config = AppConfig.get()
    trees_path = config.paths.cache_dir / "document_trees_test.json"

    # Create test tree
    test_tree = {
        "doc_id": "test_doc",
        "doc_type": "Procedure",
        "title": "Test Procedure",
        "sections": [
            {
                "section_id": "1",
                "title": "Introduction",
                "level": 1,
                "parent_id": None,
                "chunk_ids": ["chunk_001", "chunk_002"],
                "children": [
                    {
                        "section_id": "1.1",
                        "title": "Purpose",
                        "level": 2,
                        "parent_id": "1",
                        "chunk_ids": ["chunk_003"],
                        "children": [],
                    }
                ],
            }
        ],
    }

    try:
        # Save
        save_document_trees([test_tree], trees_path)
        print("✅ Saved test tree")

        # Load
        loaded_trees = load_document_trees(trees_path)
        print(f"✅ Loaded {len(loaded_trees)} trees")

        # Verify
        if len(loaded_trees) == 1:
            loaded = loaded_trees[0]
            if (
                loaded["doc_id"] == "test_doc"
                and loaded["sections"][0]["section_id"] == "1"
                and loaded["sections"][0]["children"][0]["section_id"] == "1.1"
            ):
                print("✅ Tree structure matches")

                # Cleanup
                trees_path.unlink()
                print("✅ Cleanup successful")

                return True
            else:
                print("❌ Tree structure mismatch")
                return False
        else:
            print("❌ Wrong number of trees loaded")
            return False

    except Exception as exc:
        print(f"❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("HIERARCHICAL RETRIEVAL SYSTEM - TEST SUITE")
    print("=" * 80)

    tests = [
        ("Tree Building", test_tree_building),
        ("Strategy Classification", test_strategy_classification),
        ("Tree Persistence", test_tree_persistence),
        ("Hierarchical Retrieval", test_hierarchical_retrieval),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as exc:
            print(f"\n❌ Test '{name}' failed with exception: {exc}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
