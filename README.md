# general-agentic-memory
A general memory system for agents, powered by deep-research


<h5 align="center"> 🎉 If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

**General Agentic Memory (GAM)** provides a next-generation memory framework for AI agents, combining long-term retention with dynamic reasoning. Following the Just-in-Time (JIT) principle, it preserves full contextual fidelity offline while performing deep research online to build adaptive, high-utility context. With its dual-agent architecture—Memorizer and Researcher—GAM integrates structured memory with iterative retrieval and reflection, achieving state-of-the-art performance across LoCoMo, HotpotQA, LongBench v2, and LongCodeBench benchmarks.

- **Paper**: 
- **Website**: 
- **Documentation**: 
- **YouTube Video**: 

<span id='features'/>

## ✨Key Features

* 🧠 Just-in-Time (JIT) Memory Optimization
</br> Unlike conventional Ahead-of-Time (AOT) systems, GAM performs intensive Memory Deep Research at runtime, dynamically retrieving and synthesizing high-utility context to meet real-time agent needs.

* 🔍 Dual-Agent Architecture: Memorizer & Researcher
</br> A cooperative framework where the Memorizer constructs structured memory from raw sessions, and the Researcher performs iterative retrieval, reflection, and summarization to deliver precise, adaptive context.

* 🚀 Superior Performance Across Benchmarks
</br> Achieves state-of-the-art results on LoCoMo, HotpotQA, LongBench v2, and LongCodeBench, surpassing prior systems such as A-MEM, Mem0, and MemoryOS in both F1 and BLEU-1 metrics.

* 🧩 Modular & Extensible Design
</br> Built to support flexible plug-ins for memory construction, retrieval strategies, and reasoning tools—facilitating easy integration into multi-agent frameworks or standalone LLM deployments.

* 🌐 Cross-Model Compatibility
</br> Compatible with leading LLMs such as GPT-5, GPT-4o-mini, and Qwen2.5, supporting both cloud-based and local deployments for research or production environments.

<span id='news'/>

## 📣 Latest News


## 📑 Table of Contents

* <a href='#features'>✨ Features</a>
* <a href='#news'>🔥 News</a>
* <a href='#structure'> 📁Project Structure</a>
* <a href='#pypi-mode'>🎯 Quick Start</a>
* <a href='#todo'>☑️ Todo List</a>
* <a href='#reproduce'>🔬 How to Reproduce the Results in the Paper </a>
* <a href='#doc'>📖 Documentation </a>
* <a href='#cite'>🌟 Cite</a>
* <a href='#community'>🤝 Join the Community</a>




<span id='structure'/>

## 🏗️	System Architecture
![logo](./assets/GAM-memory.png)



## 🏗️ Project Structure

```
general-agentic-memory/
├── gam/                          # 核心 GAM 包
│   ├── __init__.py              # 包初始化文件
│   ├── agents.py                # 智能代理实现 (MemoryAgent, DeepResearchAgent)
│   ├── llm_call.py              # LLM 调用接口 (BaseLLM, OpenRouter, HuggingFace)
│   ├── prompts.py               # 提示词模板
│   ├── retrieval.py             # 检索功能 (BM25Sessions)
│   └── utils.py                 # 工具函数 (文本处理、分块等)
├── examples/                     # 使用示例
│   └── quickstart/              # 快速开始示例
│       ├── basic_usage.py       # 基础使用示例
│       └── model_usage.py       # 模型选择示例
├── eval/                        # 基准测试评估
│   ├── hotpotqa/                # HotpotQA 基准测试
│   │   └── hotpotqa.py
│   ├── locomo/                  # LoCoMo 基准测试
│   │   ├── locomoqa.py
│   │   └── locomo_eval.py
│   ├── longbenchv2/             # LongBench v2 基准测试
│   │   └── longbenchqa.py
│   └── longcodebench/           # LongCodeBench 基准测试
│       └── longcodebenchqa.py
├── assets/                      # 资源文件
│   └── GAM-memory.png
├── docs/                        # 文档目录
├── setup.py                     # 安装配置
├── pyproject.toml              # 现代项目配置
├── requirements.txt             # 依赖列表
└── README.md                   # 项目说明
```


<span id='pypi-mode'/>

## 📖GAM Getting Started

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/general-agentic-memory.git
cd general-agentic-memory

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 💡 Quick Start

```python
from gam import MemoryAgent, OpenRouterModel, build_session_chunks_from_text

# Initialize LLM
llm = OpenRouterModel(
    model="gpt-4o-mini",
    api_key="your-api-key"
)

# Create memory agent
memory_agent = MemoryAgent(llm)

# Process long text
long_text = "Your long document content here..."
sessions = build_session_chunks_from_text(long_text, max_tokens=2000)

# Build memory
memory_agent.run_memory_agent(sessions=sessions)
final_memory = memory_agent.get_memory_with_abstracts()

print("Memory Events:", final_memory['events'])
print("Overall Abstract:", final_memory['abstract'])
```

### 📚 Complete Examples

For detailed examples and advanced usage, check out:
- [`examples/quickstart/basic_usage.py`](./examples/quickstart/basic_usage.py) - Complete workflow examples with long text processing and deep research
- [`examples/quickstart/model_usage.py`](./examples/quickstart/model_usage.py) - Model selection and configuration examples


<span id='todo'/>

## ☑️ Todo List


Have ideas or suggestions? Contributions are welcome! Please feel free to submit issues or pull requests! 🚀

<span id='doc'/>

## 📖 Documentation

A more detailed documentation is coming soon 🚀, and we will update in the Documentation page.

<span id='cite'/>

## 📣 Citation
**If you find this project useful, please consider citing our paper:**



<span id='related'/>



<span id='community'/>

## 🎯 Contact us


## 🌟 Star History



## Disclaimer
