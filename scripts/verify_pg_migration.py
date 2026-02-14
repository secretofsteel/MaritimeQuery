import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app
from app.config import AppConfig
from api.dependencies import get_current_user

async def mock_get_current_user():
    return {"sub": "test_user", "tenant_id": "astra", "role": "user"}

def verify_migration():
    print("Verifying API endpoints...")
    
    # Override auth dependency
    app.dependency_overrides[get_current_user] = mock_get_current_user
    
    with TestClient(app) as client:
        # 1. Check System Status
        print("\nChecking System Status...")
        resp = client.get("/api/v1/health")  # Health check
        print(f"Health: {resp.status_code} - {resp.json()}")
        
        resp = client.get("/api/v1/system/status")
        if resp.status_code == 200:
            data = resp.json()
            print(f"System Status: PG Connected={data.get('pg_connected')}, Tables={data.get('pg_tables')}")
            if not data.get('pg_connected'):
                print("FAIL: PostgreSQL not connected in system status")
                sys.exit(1)
        else:
            print(f"FAIL: System status endpoint returned {resp.status_code}")
            sys.exit(1)
            
        # 2. Check Sessions (listing)
        print("\nChecking Sessions List...")
        # valid tenant_id header might be needed if the app enforces it
        # The current implementation of SessionManager takes tenant_id from constructor.
        # The API dependency likely gets it from request header or defaults.
        # Let's assume default or 'astra' if we can force it.
        # Getting AppState to see how it's configured.
        
        # In api/dependencies.py (assumed), get_current_tenant might check header 'X-Tenant-ID'
        headers = {"X-Tenant-ID": "astra"} 
        resp = client.get("/api/v1/sessions", headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            sessions = data.get("sessions", [])
            print(f"Sessions found: {len(sessions)}")
            if len(sessions) > 0:
                print(f"First session: {sessions[0].get('session_id')} - {sessions[0].get('title')}")
            else:
                print("WARNING: No sessions found (might be expected if testing on empty tenant)")
        else:
            print(f"FAIL: Sessions endpoint returned {resp.status_code}")
            print(resp.text)
            sys.exit(1)

    print("\nVerification Successful!")

if __name__ == "__main__":
    verify_migration()
