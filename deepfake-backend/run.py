"""
run.py
======
Convenience entrypoint. Run with:

    python run.py                    # development (auto-reload)
    python run.py --env production   # production (4 workers, no reload)
"""

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceForge Detection Lab API")
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8000)
    parser.add_argument("--env",     choices=["development", "production"], default="development")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    is_prod = args.env == "production"

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not is_prod,
        workers=args.workers or (4 if is_prod else 1),
        log_level="info",
        access_log=True,
    )
