"""
MongoDB connection and helper utilities powered by Motor (async driver).

DNS workaround
──────────────
If the MONGO_URI uses the +srv scheme, the driver relies on the system DNS
to resolve the SRV record. On some networks the local DNS resolver blocks or
times-out on mongodb.net lookups. This module detects that scheme and
resolves the SRV + TXT records manually using dnspython pointed at Google
Public DNS (8.8.8.8), then builds a plain mongodb:// URI that the driver
can connect to without any further DNS lookups.
"""

import os
import re
from motor.motor_asyncio import AsyncIOMotorClient

client: AsyncIOMotorClient = None  # type: ignore
_db_connected: bool = False


def is_db_connected() -> bool:
    """Return True when the last startup ping succeeded."""
    return _db_connected


def _resolve_srv_uri(srv_uri: str) -> str:
    """
    Convert a mongodb+srv:// URI to a plain mongodb:// URI by resolving
    the SRV and TXT records via Google Public DNS (8.8.8.8).

    Falls back to the original URI if resolution fails for any reason.
    """
    try:
        import dns.resolver

        # Build a resolver that uses Google DNS, ignoring the system resolver
        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = ["8.8.8.8", "8.8.4.4"]
        resolver.timeout = 8
        resolver.lifetime = 10

        # Pull credentials and hostname out of the +srv URI
        # mongodb+srv://user:pass@hostname/dbname?options
        m = re.match(
            r"mongodb\+srv://([^@]+)@([^/?]+)(.*)",
            srv_uri,
            re.IGNORECASE,
        )
        if not m:
            return srv_uri

        credentials, hostname, rest = m.group(1), m.group(2), m.group(3)

        # 1. Resolve SRV records → list of host:port
        srv_name = f"_mongodb._tcp.{hostname}"
        srv_answers = resolver.resolve(srv_name, "SRV")
        hosts = []
        for r in srv_answers:
            h = str(r.target).rstrip(".")
            hosts.append(f"{h}:{r.port}")

        if not hosts:
            return srv_uri

        # 2. Resolve TXT records → extra options (authSource, replicaSet …)
        txt_options = ""
        try:
            txt_answers = resolver.resolve(hostname, "TXT")
            for r in txt_answers:
                txt_options = b"".join(r.strings).decode()
                break
        except Exception:
            pass

        # 3. Build a plain mongodb:// URI
        hosts_str = ",".join(hosts)
        # Always add ssl=true for Atlas
        base_opts = "ssl=true"
        if txt_options:
            base_opts += "&" + txt_options

        # Preserve any explicit query string from the original URI
        extra = ""
        if "?" in rest:
            extra = "&" + rest.split("?", 1)[1]

        plain_uri = (
            f"mongodb://{credentials}@{hosts_str}/"
            f"?{base_opts}{extra}"
        )
        print(f"[DATABASE] Resolved +srv -> plain URI (via Google DNS): {hosts_str}")
        return plain_uri

    except Exception as exc:
        print(f"[DATABASE] SRV resolution failed ({exc}), using original URI.")
        return srv_uri


async def connect_db() -> None:
    """Open the MongoDB connection. Call from FastAPI startup event."""
    global client, _db_connected

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")

    # If the URI uses +srv and our system DNS might be broken, resolve manually
    if mongo_uri.lower().startswith("mongodb+srv://"):
        mongo_uri = _resolve_srv_uri(mongo_uri)

    # Only enable TLS when connecting to Atlas (+srv already resolved above)
    # or when the URI explicitly opts in with ssl=true / tls=true.
    # Forcing tls=True on a plain localhost URI breaks the connection with
    # SSL_WRONG_VERSION_NUMBER or similar TLS handshake errors.
    _uri_lower = mongo_uri.lower()
    _use_tls = (
        "ssl=true" in _uri_lower
        or "tls=true" in _uri_lower
        or _uri_lower.startswith("mongodb+srv://")
    )

    _tls_kwargs = {"tls": True, "tlsAllowInvalidCertificates": False} if _use_tls else {}

    client = AsyncIOMotorClient(
        mongo_uri,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=15000,
        **_tls_kwargs,
    )

    try:
        await client.admin.command("ping")
        _db_connected = True
        print("[DATABASE] MongoDB connected successfully.")
        # Ensure unique indexes on the users collection so concurrent
        # registrations can never produce duplicate email accounts.
        await _setup_user_indexes()
    except Exception as exc:
        _db_connected = False
        print(f"[DATABASE] MongoDB connection failed: {exc}")
        print("   Check: Atlas IP whitelist, cluster is not paused, network access.")


async def _setup_user_indexes() -> None:
    """Create unique indexes on the users collection. Safe to call repeatedly."""
    try:
        users = client[get_db_name()]["users"]
        await users.create_index("email",   unique=True, name="users_email_unique")
        await users.create_index("user_id", unique=True, name="users_user_id_unique")
    except Exception as exc:
        print(f"[DATABASE] Could not create user indexes: {exc}")


def get_db_name() -> str:
    return os.getenv("MONGO_DB_NAME", "ai_bg_remover")


async def close_db() -> None:
    """Close the MongoDB connection. Call from FastAPI shutdown event."""
    global _db_connected
    if client:
        client.close()
    _db_connected = False


def get_collection(name: str):
    """Return a Motor collection by name."""
    if client is None:
        raise RuntimeError("Database client is not initialised.")
    return client[get_db_name()][name]
