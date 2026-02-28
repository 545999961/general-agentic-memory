# Web Usage Guide

This guide describes how to use the `gam-afs` Web interface for building and exploring memories.

## Installation

To use the Web platform, ensure you have installed the optional web dependencies:
```bash
pip install -e ".[web]"
```

## Running the Web Platform

Start the Web interface by running the following script:
```bash
python examples/run_web.py
```

### Custom Configuration

Customize the server address, port, and LLM backend:
```bash
python examples/run_web.py \
    --host 127.0.0.1 --port 8080 \
    --model gpt-4o \
    --api-base https://api.openai.com/v1 \
    --api-key sk-your-api-key
```

## Key Features

- **Pipeline Builder**: Upload documents for automatic intelligent chunking and memory organization.
- **Incremental Addition**: Append new documents to existing GAMs.
- **GAM Browser**: Explore memory nodes through an intuitive tree structure.
- **Intelligent QA**: Interactive chat interface for querying GAM knowledge bases.
- **Session Management**: Manage outputs and load historical sessions.

---
