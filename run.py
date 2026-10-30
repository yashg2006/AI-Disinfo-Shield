"""
ModelSentinel — AI Model Weight Steganography & Malware Scanner
Entry point: starts the FastAPI server with uvicorn.
"""

import uvicorn


def main():
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
