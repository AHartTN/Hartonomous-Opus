#!/usr/bin/env python3
"""
OpenAI-compatible RAG Gateway - Modular Entry Point
"""
import sys
import os

# Ensure the package can be imported
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Run the modular OpenAI Gateway application"""
    from openai_gateway.main import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8700, log_level="info")

if __name__ == "__main__":
    main()
