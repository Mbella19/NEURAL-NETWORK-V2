#!/usr/bin/env python3
"""
Launch the real-time training visualization dashboard.

This script starts both:
1. FastAPI backend (WebSocket server on port 8000)
2. Next.js frontend (development server on port 3000)

Usage:
    python scripts/start_dashboard.py

    # Or with custom ports:
    python scripts/start_dashboard.py --backend-port 8000 --frontend-port 3000

Requirements:
    - Python packages: fastapi, uvicorn, websockets
    - Node.js and npm installed
    - Run `cd frontend && npm install` first
"""

import argparse
import subprocess
import sys
import os
import signal
import time
from pathlib import Path
from typing import List, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def check_dependencies():
    """Check that required dependencies are installed."""
    errors = []

    # Check Python dependencies
    try:
        import fastapi
        import uvicorn
    except ImportError as e:
        errors.append(f"Missing Python package: {e.name}. Run: pip install fastapi uvicorn websockets")

    # Check Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append("Node.js is not installed")
    except FileNotFoundError:
        errors.append("Node.js is not installed")

    # Check npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append("npm is not installed")
    except FileNotFoundError:
        errors.append("npm is not installed")

    # Check if frontend dependencies are installed
    frontend_dir = get_project_root() / "frontend"
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        errors.append(f"Frontend dependencies not installed. Run: cd {frontend_dir} && npm install")

    if errors:
        print("Dependency check failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def start_backend(port: int = 8000) -> subprocess.Popen:
    """Start the FastAPI backend server."""
    project_root = get_project_root()

    print(f"Starting backend server on port {port}...")

    # Use uvicorn to run the server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "visualization.server:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload",
        "--log-level", "info",
    ]

    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(project_root)},
    )

    return process


def start_frontend(port: int = 3000) -> subprocess.Popen:
    """Start the Next.js frontend dev server."""
    frontend_dir = get_project_root() / "frontend"

    print(f"Starting frontend server on port {port}...")

    # Start Next.js dev server
    cmd = ["npm", "run", "dev", "--", "-p", str(port)]

    process = subprocess.Popen(
        cmd,
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
    )

    return process


def stream_output(process: subprocess.Popen, prefix: str):
    """Stream process output with a prefix."""
    for line in iter(process.stdout.readline, b''):
        if line:
            print(f"[{prefix}] {line.decode().rstrip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Start the training visualization dashboard"
    )
    parser.add_argument(
        "--backend-port", type=int, default=8000,
        help="Backend server port (default: 8000)"
    )
    parser.add_argument(
        "--frontend-port", type=int, default=3000,
        help="Frontend server port (default: 3000)"
    )
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip dependency checks"
    )

    args = parser.parse_args()

    if not args.skip_checks:
        check_dependencies()

    processes: List[subprocess.Popen] = []

    def cleanup(signum=None, frame=None):
        """Clean up processes on exit."""
        print("\nShutting down...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Start backend
        backend_proc = start_backend(args.backend_port)
        processes.append(backend_proc)

        # Wait a bit for backend to start
        time.sleep(2)

        # Start frontend
        frontend_proc = start_frontend(args.frontend_port)
        processes.append(frontend_proc)

        print("\n" + "=" * 60)
        print("Dashboard is starting...")
        print(f"  Backend:  http://localhost:{args.backend_port}")
        print(f"  Frontend: http://localhost:{args.frontend_port}")
        print("=" * 60)
        print("\nPress Ctrl+C to stop\n")

        # Stream output from both processes
        import threading

        def stream_backend():
            stream_output(backend_proc, "BACKEND")

        def stream_frontend():
            stream_output(frontend_proc, "FRONTEND")

        backend_thread = threading.Thread(target=stream_backend, daemon=True)
        frontend_thread = threading.Thread(target=stream_frontend, daemon=True)

        backend_thread.start()
        frontend_thread.start()

        # Wait for processes
        while True:
            # Check if either process has died
            backend_status = backend_proc.poll()
            frontend_status = frontend_proc.poll()

            if backend_status is not None:
                print(f"Backend exited with code {backend_status}")
                cleanup()

            if frontend_status is not None:
                print(f"Frontend exited with code {frontend_status}")
                cleanup()

            time.sleep(1)

    except Exception as e:
        print(f"Error: {e}")
        cleanup()


if __name__ == "__main__":
    main()
