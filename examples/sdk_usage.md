# SDK Usage Guide

This guide provides examples and detailed information on using the `gam-afs` Python SDK.

## Workflow API (Recommended)

`Workflow` is the easiest way to interact with GAM. Simply use `from gam import Workflow` to manage your memory workflows.

### Text Memory

```python
from gam import Workflow

# Initialize Text Workflow
wf = Workflow("text", gam_dir="./my_gam", model="gpt-4o-mini", api_key="sk-xxx")

# Build memory (Add files or direct text)
wf.add(input_file="paper.pdf")
wf.add(input_file=["paper1.pdf", "paper2.txt"])
wf.add(content="Some text to memorize")

# Query memory
result = wf.request("What is the main conclusion?")
print(result.answer)
print(result.sources)
```

### Video Memory

```python
from gam import Workflow

# Initialize Video Workflow
wf = Workflow(
    "video",
    gam_dir="./my_video_gam",
    model="gpt-4o",
    api_key="sk-xxx",
    # Optional: use a different model for segment descriptions
    # segmentor_model="gpt-4o-mini",
)

# Build video memory (Input directory should contain video.mp4 + optional subtitles.srt)
wf.add(input_path="./video_dir")

# Query video memory (Supports text tools + video frame inspection)
result = wf.request("What happens in the video?")
print(result.answer)
```

### Workflow Parameters Reference

**GAM Agent (memory building) configuration:**

| Parameter | Default | Description |
|---|---|---|
| `workflow_type` | `"text"` | Type of workflow: `"text"` or `"video"` |
| `gam_dir` | — | Path to the GAM directory (created automatically) |
| `model` | `gpt-4o-mini` | GAM Agent LLM model name (Env: `GAM_MODEL`) |
| `api_base` | `https://api.openai.com/v1` | GAM Agent API base URL (Env: `GAM_API_BASE`) |
| `api_key` | — | GAM Agent API key (Env: `GAM_API_KEY` or `OPENAI_API_KEY`) |
| `max_tokens` | `4096` | Maximum tokens to generate |
| `temperature` | `0.3` | Sampling temperature |
| `verbose` | `True` | Whether to print detailed logs |

**Chat Agent (Q&A) configuration** — falls back to GAM Agent config when not set:

| Parameter | Default | Description |
|---|---|---|
| `chat_model` | same as `model` | Chat Agent LLM model name (Env: `GAM_CHAT_MODEL`) |
| `chat_api_base` | same as `api_base` | Chat Agent API base URL (Env: `GAM_CHAT_API_BASE`) |
| `chat_api_key` | same as `api_key` | Chat Agent API key (Env: `GAM_CHAT_API_KEY`) |

**Text-Specific**: `use_chunking` (bool), `memory_workers` (int), `max_iterations` (int)

**Video-Specific**: `segmentor_model`, `video_model`, `video_fps`, `video_max_resolution`, `max_iterations`

---

## Low-level Python API (Advanced)

For fine-grained control, you can interact with individual components.

### Building Memory

```python
from pathlib import Path
from gam import (
    TextGAMAgent, GAMTree,
    OpenAIGenerator, OpenAIGeneratorConfig,
    LocalWorkspace,
)

# 1. Initialize LLM Generator
config = OpenAIGeneratorConfig(
    model_name="gpt-4o-mini",
    api_key="sk-your-api-key",
    base_url="https://api.openai.com/v1",
    max_tokens=4096,
    temperature=0.3,
)
generator = OpenAIGenerator(config)

# 2. Initialize Workspace and Tree
workspace = LocalWorkspace(root_path="./my_gam")
tree = GAMTree.create_empty(Path("./my_gam"), name="my_memory")

# 3. Create Agent and add content
agent = TextGAMAgent(generator, tree, workspace, use_chunking=True)
result = agent.add(input_file=Path("paper.pdf"))
```

### Querying Memory

```python
from pathlib import Path
from gam import (
    TextChatAgent, GAMTree,
    OpenAIGenerator, OpenAIGeneratorConfig,
    LocalWorkspace,
)

config = OpenAIGeneratorConfig(
    model_name="gpt-4o-mini",
    api_key="sk-your-api-key",
    base_url="https://api.openai.com/v1",
    max_tokens=4096,
)
generator = OpenAIGenerator(config)

workspace = LocalWorkspace(root_path="./my_gam")
tree = GAMTree.from_disk(Path("./my_gam"), workspace)

agent = TextChatAgent(generator, tree, workspace=workspace)
result = agent.request(
    system_prompt="You are a helpful research assistant.",
    user_prompt="What is the main conclusion of the paper?",
)
print(result.answer)
```

## SDK API Reference

### Workflow (Primary Entrypoint)

Initialized via `Workflow("text", ...)` or `Workflow("video", ...)`.

| Method | Description |
|---|---|
| `add(...)` | Add files/text/video to GAM knowledge base |
| `request(question, ...)` | Run exploratory QA on the GAM |
| `chat(question, ...)` | Alias for `request()` |
| `get_tree_view(path, depth)` | Retrieve current directory structure |
| `get_skill_prompt()` | Get skill description (for external agent injection) |
| `reload_tree()` | Reload tree structure from disk |

### TextGAMAgent
Builds text memory, handles chunking, memory generation, and organization.

| Method | Description |
|---|---|
| `add(input_file, content, ...)` | Add file/text to GAM |
| `chunk(text, context, ...)` | Intelligent text chunking |
| `organize_from_memorized_chunks(...)` | Organize GAM structure from existing chunks |
| `get_tree_view()` | Retrieve current directory structure |

### TextChatAgent
Interactive agent for querying text GAM.

| Method | Description |
|---|---|
| `request(system_prompt, user_prompt, ...)` | Execute full request with custom prompts |
| `chat(question, ...)` | Simplified query interface |

### VideoGAMAgent
Builds video memory, handles segmenting, description, and summarization.

| Method | Description |
|---|---|
| `add(input_path, ...)` | Build Video GAM from directory |

### VideoChatAgent
Querying agent for video GAM, supporting multimodal tools.

| Method | Description |
|---|---|
| `request(system_prompt, user_prompt, ...)` | Execute full request with custom prompts |

---
