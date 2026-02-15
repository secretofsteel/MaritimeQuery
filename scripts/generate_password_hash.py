#!/usr/bin/env python3
"""
Password Hash Generator for Maritime RAG Authentication
=======================================================

Generates bcrypt password hashes for use in config/users.yaml.

Usage:
    python scripts/generate_password_hash.py
    
    # Or with password as argument (less secure - visible in shell history)
    python scripts/generate_password_hash.py "mypassword"

The generated hash can be copied directly into users.yaml.
"""

import sys
import getpass

def generate_hash(password: str) -> str:
    """Generate bcrypt hash for a password."""
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def main():
    print("=" * 50)
    print("Maritime RAG Password Hash Generator")
    print("=" * 50)
    print()
    
    # Get password
    if len(sys.argv) > 1:
        password = sys.argv[1]
        print("Using password from command line argument")
        print("(Note: This is visible in shell history)")
    else:
        password = getpass.getpass("Enter password to hash: ")
        confirm = getpass.getpass("Confirm password: ")
        
        if password != confirm:
            print("\nError: Passwords don't match!")
            sys.exit(1)
    
    if len(password) < 6:
        print("\nWarning: Password is very short (< 6 characters)")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Generate hash
    password_hash = generate_hash(password)
    
    print()
    print("Generated hash:")
    print("-" * 50)
    print(password_hash)
    print("-" * 50)
    print()
    print("Copy this hash into config/users.yaml under the")
    print("appropriate user's 'password' field.")
    print()
    
    # Also generate a cookie key if needed
    print("=" * 50)
    print("Bonus: Cookie Secret Key")
    print("=" * 50)
    print()
    print("If you need a new cookie key, here's a secure one:")
    print()
    
    import secrets
    cookie_key = secrets.token_hex(16)
    print(cookie_key)
    print()
    print("Use this for the 'key' field under 'cookie' in users.yaml")
    print()


if __name__ == "__main__":
    main()
