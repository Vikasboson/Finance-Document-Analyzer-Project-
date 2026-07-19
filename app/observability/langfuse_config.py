"""
langfuse_config.py
------------------
Reads Langfuse credentials from .env.example (same file as
AWS credentials) and creates the singleton client.


"""

import os
from dotenv import load_dotenv

# Same file the rest of the project uses for AWS creds
load_dotenv(".env.example")

_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "")
_HOST   = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

langfuse = None

if _PUBLIC and _SECRET:
    try:
        from langfuse import Langfuse
        langfuse = Langfuse(
            public_key=_PUBLIC,
            secret_key=_SECRET,
            host=_HOST,
        )
        print(f"[Langfuse] ✅  Connected → {_HOST}")
    except Exception as exc:
        print(f"[Langfuse] ⚠️  Init failed ({exc}), observability disabled.")
else:
    print("[Langfuse] ⚠️  Keys not set in .env.example — observability disabled.")


def is_enabled() -> bool:
    return langfuse is not None


def flush():
    """Push buffered events. Call on app shutdown."""
    if langfuse:
        langfuse.flush()