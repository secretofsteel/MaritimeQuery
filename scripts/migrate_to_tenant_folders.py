#!/usr/bin/env python3
"""
Migrate flat docs/ folder to per-tenant subfolders.

This script:
1. Reads the existing gemini_extract_cache.jsonl to determine each file's tenant_id
2. Reads config/users.yaml to ensure folders exist for ALL known tenants
3. Moves files from data/docs/ into data/docs/{tenant_id}/
4. Splits the single JSONL into per-tenant JSONL files
5. Backs up the legacy JSONL (does not delete it)

Usage:
    cd /opt/MaritimeQuery   (or your project root)
    python scripts/migrate_to_tenant_folders.py

    # Dry run (shows what would happen without moving anything):
    python scripts/migrate_to_tenant_folders.py --dry-run

Safe to run multiple times — skips files already in subfolders.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Resolve project root (script is in scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
CACHE_DIR = DATA_DIR / "cache"
LEGACY_JSONL = CACHE_DIR / "gemini_extract_cache.jsonl"
USERS_YAML = PROJECT_ROOT / "config" / "users.yaml"


def load_users_yaml() -> list[str]:
    """Extract tenant_ids from users.yaml."""
    try:
        import yaml
    except ImportError:
        print("WARNING: PyYAML not installed, cannot read users.yaml")
        print("  Install with: pip install pyyaml")
        return []

    if not USERS_YAML.exists():
        print(f"WARNING: {USERS_YAML} not found")
        return []

    with open(USERS_YAML) as f:
        data = yaml.safe_load(f)

    tenants = set()
    usernames = data.get("credentials", {}).get("usernames", {})
    for username, user_data in usernames.items():
        tid = user_data.get("tenant_id")
        if tid:
            tenants.add(tid)

    return sorted(tenants)


def load_legacy_jsonl() -> dict[str, dict]:
    """Load the legacy single JSONL cache."""
    if not LEGACY_JSONL.exists():
        return {}

    records = {}
    for line in LEGACY_JSONL.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            record = json.loads(line)
            filename = record.get("filename")
            if filename:
                records[filename] = record
        except json.JSONDecodeError:
            continue
    return records


def migrate(dry_run: bool = False):
    print("=" * 60)
    print("Per-Tenant Folder Migration")
    print("=" * 60)

    if dry_run:
        print("\n🔍 DRY RUN — no files will be moved\n")

    # Step 1: Read tenant assignments from JSONL
    cached_records = load_legacy_jsonl()
    file_tenants: dict[str, str] = {}
    records_by_tenant: dict[str, list] = {}

    for filename, record in cached_records.items():
        tenant = record.get("tenant_id", "shared")
        file_tenants[filename] = tenant
        records_by_tenant.setdefault(tenant, []).append(record)

    print(f"📄 JSONL: {len(cached_records)} records")
    for tid, recs in sorted(records_by_tenant.items()):
        print(f"   {tid}: {len(recs)} files")

    # Step 2: Get all known tenants from YAML
    yaml_tenants = load_users_yaml()
    print(f"\n👥 Users YAML: {len(yaml_tenants)} tenants: {yaml_tenants}")

    # Combine tenants from both sources + always include 'shared'
    all_tenants = sorted(set(["shared"] + yaml_tenants + list(records_by_tenant.keys())))
    print(f"📁 Will create folders for: {all_tenants}")

    # Step 3: Create tenant subdirectories
    print("\n--- Creating tenant folders ---")
    for tid in all_tenants:
        tenant_dir = DOCS_DIR / tid
        if tenant_dir.exists():
            print(f"  ✓ {tid}/ (already exists)")
        else:
            if not dry_run:
                tenant_dir.mkdir(parents=True, exist_ok=True)
            print(f"  + {tid}/ (created)")

    # Step 4: Move files into tenant subfolders
    print("\n--- Moving files ---")
    files_in_base = [f for f in DOCS_DIR.glob("*") if f.is_file()]

    if not files_in_base:
        print("  No files in base docs/ directory — nothing to move")
    else:
        moved = 0
        orphaned = 0
        for file_path in sorted(files_in_base):
            tenant = file_tenants.get(file_path.name, "shared")
            target_dir = DOCS_DIR / tenant
            target_path = target_dir / file_path.name

            if target_path.exists():
                print(f"  ⚠ {file_path.name} → {tenant}/ (SKIPPED, already exists)")
                continue

            if file_path.name not in file_tenants:
                orphaned += 1
                print(f"  ? {file_path.name} → shared/ (no JSONL entry, defaulting)")
            else:
                print(f"  → {file_path.name} → {tenant}/")

            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(target_path))
            moved += 1

        print(f"\n  Moved: {moved} files ({orphaned} orphaned → shared/)")

    # Step 5: Split JSONL into per-tenant files
    if cached_records:
        print("\n--- Splitting JSONL cache ---")
        for tenant, records in sorted(records_by_tenant.items()):
            tenant_jsonl = CACHE_DIR / f"gemini_extract_cache_{tenant}.jsonl"
            if tenant_jsonl.exists():
                # Merge with existing (in case of partial migration)
                existing = {}
                for line in tenant_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
                    try:
                        rec = json.loads(line)
                        fn = rec.get("filename")
                        if fn:
                            existing[fn] = rec
                    except json.JSONDecodeError:
                        continue
                
                for rec in records:
                    existing[rec["filename"]] = rec
                
                if not dry_run:
                    with open(tenant_jsonl, "w", encoding="utf-8") as f:
                        for rec in existing.values():
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  ✏ {tenant_jsonl.name}: {len(existing)} records (merged)")
            else:
                if not dry_run:
                    with open(tenant_jsonl, "w", encoding="utf-8") as f:
                        for record in records:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"  + {tenant_jsonl.name}: {len(records)} records")

        # Backup legacy JSONL
        backup = LEGACY_JSONL.with_suffix(".jsonl.bak")
        if not dry_run:
            if backup.exists():
                print(f"\n  ⚠ Backup already exists: {backup.name}")
            else:
                LEGACY_JSONL.rename(backup)
                print(f"\n  📦 Legacy JSONL backed up to {backup.name}")
        else:
            print(f"\n  📦 Would back up legacy JSONL to {backup.name}")
    else:
        print("\n--- No legacy JSONL found, skipping split ---")

    # Step 6: Handle sync_cache.json → per-tenant
    legacy_sync = CACHE_DIR / "sync_cache.json"
    if legacy_sync.exists():
        print("\n--- Migrating sync cache ---")
        # The old sync cache has a flat files_hash.
        # We can't reliably split it by tenant without the JSONL mapping.
        # Safest approach: just rename it so a fresh sync builds per-tenant caches.
        backup_sync = legacy_sync.with_suffix(".json.bak")
        if not dry_run:
            if not backup_sync.exists():
                legacy_sync.rename(backup_sync)
                print(f"  📦 sync_cache.json backed up to {backup_sync.name}")
                print("  ℹ  Fresh per-tenant sync caches will be created on next sync")
            else:
                print(f"  ⚠ Sync cache backup already exists")
        else:
            print(f"  Would back up sync_cache.json to {backup_sync.name}")

    # Summary
    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN complete — no changes made")
        print("Run without --dry-run to execute migration")
    else:
        print("Migration complete!")
        print("\nNext steps:")
        print("  1. Restart the application: systemctl restart maritime-rag")
        print("  2. Verify in Admin Panel that documents appear correctly")
        print("  3. Run a sync for each tenant to rebuild sync caches")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate to per-tenant folder structure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")
    args = parser.parse_args()

    migrate(dry_run=args.dry_run)
