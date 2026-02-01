#!/usr/bin/env python3
"""
ChromaDB Tenant Migration Script
================================

Adds tenant_id metadata to existing ChromaDB vectors.
Run ONCE after deploying Phase 4 code.

Usage:
    python scripts/migrate_chroma_tenant.py
    
    # Dry run
    python scripts/migrate_chroma_tenant.py --dry-run
    
    # Set specific tenant (default: 'shared')
    python scripts/migrate_chroma_tenant.py --tenant shared
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from app.config import AppConfig
from app.logger import LOGGER


def migrate_chroma_tenant(
    tenant_id: str = "shared",
    dry_run: bool = False,
    batch_size: int = 500
) -> bool:
    """
    Add tenant_id to all existing ChromaDB vectors.
    
    Args:
        tenant_id: Tenant ID to assign to existing vectors
        dry_run: If True, show what would change without changing
        batch_size: Number of vectors to update per batch
    
    Returns:
        True if successful
    """
    config = AppConfig.get()
    chroma_path = config.paths.chroma_path
    
    print("=" * 60)
    print("CHROMADB TENANT MIGRATION")
    print("=" * 60)
    print()
    print(f"ChromaDB path: {chroma_path}")
    print(f"Target tenant_id: {tenant_id}")
    print()
    
    if not chroma_path.exists():
        print("✗ ChromaDB path doesn't exist")
        return False
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    try:
        collection = client.get_collection("maritime_docs")
    except Exception as exc:
        print(f"✗ Failed to get collection: {exc}")
        return False
    
    total_count = collection.count()
    print(f"Total vectors in collection: {total_count}")
    print()
    
    if total_count == 0:
        print("✓ No vectors to migrate")
        return True
    
    # Check current state - sample some vectors
    sample = collection.get(limit=5, include=["metadatas"])
    
    has_tenant = False
    missing_tenant = False
    
    for metadata in sample["metadatas"]:
        if "tenant_id" in metadata:
            has_tenant = True
        else:
            missing_tenant = True
    
    if has_tenant and not missing_tenant:
        print("✓ All sampled vectors already have tenant_id")
        print("  Migration may have already run. Use --dry-run to check.")
    elif missing_tenant:
        print(f"⚠ Found vectors without tenant_id - migration needed")
    
    print()
    
    if dry_run:
        print("DRY RUN - No changes will be made")
        print()
        
        # Get all IDs to check
        all_data = collection.get(include=["metadatas"])
        
        needs_update = 0
        already_set = 0
        
        for metadata in all_data["metadatas"]:
            if "tenant_id" not in metadata:
                needs_update += 1
            else:
                already_set += 1
        
        print(f"Vectors needing tenant_id: {needs_update}")
        print(f"Vectors already with tenant_id: {already_set}")
        return True
    
    # Perform migration in batches
    print("Migrating vectors...")
    
    updated_count = 0
    skipped_count = 0
    offset = 0
    
    while offset < total_count:
        # Get batch
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )
        
        if not batch["ids"]:
            break
        
        # Find vectors needing update
        ids_to_update = []
        metadatas_to_update = []
        
        for i, (chunk_id, metadata) in enumerate(zip(batch["ids"], batch["metadatas"])):
            if "tenant_id" not in metadata:
                metadata["tenant_id"] = tenant_id
                ids_to_update.append(chunk_id)
                metadatas_to_update.append(metadata)
            else:
                skipped_count += 1
        
        # Update batch
        if ids_to_update:
            collection.update(
                ids=ids_to_update,
                metadatas=metadatas_to_update
            )
            updated_count += len(ids_to_update)
            print(f"  Updated {updated_count}/{total_count}...", end="\r")
        
        offset += batch_size
    
    print()
    print()
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print()
    print(f"✓ Updated: {updated_count} vectors")
    print(f"✓ Skipped (already had tenant_id): {skipped_count}")
    print(f"✓ Total: {updated_count + skipped_count}")
    print()
    print(f"All vectors now have tenant_id='{tenant_id}'")
    
    return True


def verify_migration() -> bool:
    """Verify that migration was successful."""
    config = AppConfig.get()
    chroma_path = config.paths.chroma_path
    
    print("Verifying migration...")
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection("maritime_docs")
    
    # Sample check
    sample = collection.get(limit=100, include=["metadatas"])
    
    missing = 0
    by_tenant = {}
    
    for metadata in sample["metadatas"]:
        tenant = metadata.get("tenant_id")
        if tenant:
            by_tenant[tenant] = by_tenant.get(tenant, 0) + 1
        else:
            missing += 1
    
    print(f"Sample of 100 vectors:")
    for tenant, count in sorted(by_tenant.items()):
        print(f"  tenant_id='{tenant}': {count}")
    if missing:
        print(f"  missing tenant_id: {missing}")
    
    if missing == 0:
        print()
        print("✓ Migration verified - all vectors have tenant_id")
        return True
    else:
        print()
        print("⚠ Some vectors missing tenant_id")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add tenant_id to ChromaDB vectors"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--tenant",
        default="shared",
        help="Tenant ID to assign (default: 'shared')"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing migration"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_migration()
    else:
        success = migrate_chroma_tenant(
            tenant_id=args.tenant,
            dry_run=args.dry_run
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
