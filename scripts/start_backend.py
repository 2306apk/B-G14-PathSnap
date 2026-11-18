#!/usr/bin/env python3
"""Startup script for PathSnap Map-Matching Backend API server.

Binds to 0.0.0.0 by default to allow cross-machine LAN and ZeroTier connections.
Prints clear connection URLs for the Windows frontend machine.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure paths
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import uvicorn
from pathsnap.api import get_local_ip_addresses


def main():
    parser = argparse.ArgumentParser(description="PathSnap Backend API Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Host IP to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot reloading")
    args = parser.parse_args()

    net_info = get_local_ip_addresses()
    lan_ip = net_info.get("lan", "127.0.0.1")
    zt_ip = net_info.get("zerotier")

    print("=" * 76)
    print("🚀  PathSnap Map-Matching Backend API Starting...")
    print("=" * 76)
    print(f"• Local Hostname    : {net_info.get('hostname')}")
    print(f"• Bound Host        : {args.host}")
    print(f"• Port              : {args.port}")
    print(f"• Local Machine URL : http://127.0.0.1:{args.port}")
    print(f"• Mode 1 (Same Wi-Fi/Hotspot) LAN URL   : http://{lan_ip}:{args.port}")
    if zt_ip:
        print(f"• Mode 2 (ZeroTier Network) URL         : http://{zt_ip}:{args.port}")
    else:
        print(f"• Mode 2 (ZeroTier Network) URL         : (ZeroTier interface not detected)")
    print("-" * 76)
    print("📌  FRONTEND CONFIGURATION INSTRUCTIONS FOR TEAMMATE (WINDOWS):")
    print(f"    In frontend/.env.local (or Windows environment), set:")
    print(f"    VITE_API_BASE_URL=http://{lan_ip}:{args.port}")
    print(f"    (Or use ZeroTier IP if connected via ZeroTier)")
    print("=" * 76)

    uvicorn.run(
        "pathsnap.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

