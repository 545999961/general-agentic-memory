# CLI Usage Guide

This guide provides examples and detailed information on using the `gam-afs` command-line tools.

## Installation and Setup

Ensure you have installed the package:
```bash
pip install -e ".[all]"
```

Configure environment variables for convenience. GAM Agent and Chat Agent can be configured independently:
```bash
# GAM Agent (memory building)
export GAM_API_KEY="sk-your-api-key"
export GAM_MODEL="gpt-4o-mini"
export GAM_API_BASE="https://api.openai.com/v1"

# Chat Agent (Q&A) — falls back to GAM Agent config when not set
export GAM_CHAT_API_KEY="sk-your-chat-api-key"
export GAM_CHAT_MODEL="gpt-4o"
export GAM_CHAT_API_BASE="https://api.openai.com/v1"
```

## Commands Overview

The primary commands are `gam-add` (for building memory) and `gam-request` (for querying memory). Use the `--type text/video` flag to switch between modalities.

### `gam-add` — Building Memory

#### Text Mode (`--type text`)

```bash
# Add a single file
gam-add --type text --gam-dir ./my_gam --input paper.pdf

# Add multiple files
gam-add --type text --gam-dir ./my_gam --input paper1.pdf --input paper2.txt

# Add direct text content
gam-add --type text --gam-dir ./my_gam --content "Some text to memorize"

# Disable chunking (treat input as a single memory)
gam-add --type text --gam-dir ./my_gam --input paper.pdf --no-chunking
```

#### Video Mode (`--type video`)

```bash
# Build memory from a video directory (should contain video.mp4 + optional subtitles.srt)
gam-add --type video --gam-dir ./my_video_gam --input ./video_dir

# Use specific models
gam-add --type video --gam-dir ./my_video_gam --input ./video_dir \
    --model gpt-4o --api-key sk-xxx
```

### `gam-request` — Querying Memory

#### Text Mode (`--type text`)

```bash
# Basic query
gam-request --type text --gam-dir ./my_gam --question "What is the main conclusion?"

# Custom system prompt + JSON output
gam-request --type text --gam-dir ./my_gam -q "Summarize the paper" \
    --system-prompt "You are a research assistant." --json
```

#### Video Mode (`--type video`)

```bash
# Video query
gam-request --type video --gam-dir ./my_video_gam --question "What happens in the video?"

# Use a specific multimodal vision model
gam-request --type video --gam-dir ./my_video_gam -q "Describe the scene" \
    --video-model gpt-4o
```

## CLI Reference

### Common Parameters (`gam-add` / `gam-request`):

| Parameter | Default | Description |
|---|---|---|
| `--type, -t` | *(Required)* | GAM type: `text` or `video` |
| `--gam-dir` | *(Required)* | Path to GAM directory (created automatically) |
| `--model` | `gpt-4o-mini` | GAM Agent LLM model name (Env: `GAM_MODEL`) |
| `--api-base` | `https://api.openai.com/v1` | GAM Agent API base URL (Env: `GAM_API_BASE`) |
| `--api-key` | — | GAM Agent API key (Env: `GAM_API_KEY` or `OPENAI_API_KEY`) |
| `--max-tokens` | `4096` | Maximum tokens to generate |
| `--temperature` | `0.3` | Sampling temperature |
| `--verbose / --no-verbose` | `--verbose` | Print detailed logs |

### `gam-add --type text` Specific:

| Parameter | Description |
|---|---|
| `--input, -i` | Path to input files (can be repeated) |
| `--content, -c` | Input text content directly (can be repeated) |
| `--context` | Optional context/description for the input |
| `--chunking / --no-chunking` | Enable intelligent chunking (default: on) |
| `--output-dir` | Directory to save raw chunks |
| `--force-reorganize` | Force reorganization of the directory structure |
| `--memory-workers` | Parallel memory generation workers (default: 4) |

### `gam-add --type video` Specific:

| Parameter | Description |
|---|---|
| `--input, -i` | Path to video directory (must contain `video.mp4`) |
| `--segmentor-model` | LLM model for segment descriptions |
| `--segmentor-api-base` | API base for the segmentor model |
| `--segmentor-api-key` | API key for the segmentor model |
| `--caption-subtitles` | Include subtitles in descriptions (default: on) |

### `gam-request` Specific:

| Parameter | Description |
|---|---|
| `--question, -q` | User question (Required) |
| `--system-prompt, -s` | Custom system prompt |
| `--max-iter` | Maximum iterations for the exploratory agent (default: 10) |
| `--json` | Output result in JSON format |
| `--chat-model` | Chat Agent LLM model name (Env: `GAM_CHAT_MODEL`, default: same as `--model`) |
| `--chat-api-base` | Chat Agent API base URL (Env: `GAM_CHAT_API_BASE`, default: same as `--api-base`) |
| `--chat-api-key` | Chat Agent API key (Env: `GAM_CHAT_API_KEY`, default: same as `--api-key`) |

### `gam-request --type video` Specific:

| Parameter | Description |
|---|---|
| `--video-model` | Multimodal model name for visual analysis |
| `--video-api-base` | API base for the video model |
| `--video-api-key` | API key for the video model |
| `--video-fps` | Sampling rate for video frames (default: 1.0) |
| `--video-max-resolution` | Max resolution for frames (default: 480) |

---
