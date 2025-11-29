#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM Basic Usage Example

这个示例展示了如何使用 GAM 框架进行基本的记忆构建和问答。
演示了记忆构建、检索和研究的完整流程。
"""

import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from gam import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
    DenseRetriever,
    DenseRetrieverConfig,
)


def basic_memory_example():
    """基础记忆构建示例"""
    print("=== 基础记忆构建示例 ===\n")
    
    # 1. 配置并创建 Generator
    gen_config = OpenAIGeneratorConfig(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量读取
        temperature=0.3,
        max_tokens=256
    )
    generator = OpenAIGenerator(gen_config)
    
    # 2. 创建存储
    memory_store = InMemoryMemoryStore()
    page_store = InMemoryPageStore()
    
    # 3. 创建 MemoryAgent
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store
    )
    
    # 4. 准备要记忆的文本（模拟长文档）
    documents = [
        """人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
        机器学习是 AI 的一个子集，使计算机能够在不被明确编程的情况下学习。""",
        
        """深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的工作方式。
        自然语言处理（NLP）是 AI 的另一个重要分支，专注于使计算机能够理解、解释和生成人类语言。""",
        
        """计算机视觉是 AI 的另一个关键领域，致力于使计算机能够"看到"和理解视觉信息。
        强化学习是一种机器学习方法，通过与环境的交互来学习最优的行为策略。""",
        
        """神经网络是深度学习的基础，由相互连接的节点（神经元）组成。
        卷积神经网络（CNN）特别适用于图像处理任务，而循环神经网络（RNN）擅长处理序列数据。""",
        
        """Transformer 架构的引入彻底改变了自然语言处理领域，为 GPT 和 BERT 等大型语言模型奠定了基础。"""
    ]
    
    # 5. 逐个记忆文档
    print(f"正在记忆 {len(documents)} 个文档...")
    for i, doc in enumerate(documents, 1):
        print(f"  记忆文档 {i}/{len(documents)}...")
        memory_agent.memorize(doc)
    
    # 6. 查看记忆状态
    memory_state = memory_agent.load()
    print(f"\n✅ 成功构建记忆:")
    print(f"  - 记忆摘要数: {len(memory_state.abstracts)}")
        
    return memory_agent, memory_store, page_store




def research_example(memory_store, page_store):
    """基于记忆的研究示例"""
    print("\n=== 基于记忆的研究示例 ===\n")
    
    # 1. 配置并创建 Generator
    gen_config = OpenAIGeneratorConfig(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
        max_tokens=2048
    )
    generator = OpenAIGenerator(gen_config)
    
    # 2. 创建检索器
    retriever_config = DenseRetrieverConfig(
        model_path="BAAI/bge-base-en-v1.5",
        top_k=5
    )
    retriever = DenseRetriever(
        config=retriever_config,
        memory_store=memory_store,
        page_store=page_store
    )
    
    # 3. 创建 ResearchAgent
    research_agent = ResearchAgent(
        generator=generator,
        retriever=retriever
    )
    
    # 4. 进行研究
    question = "机器学习和深度学习有什么关键区别？"
    print(f"研究问题: {question}\n")
    
    result = research_agent.research(question=question, top_k=3)
    
    # 5. 显示结果
    print(f"✅ 研究完成:")
    print(f"  - 迭代次数: {len(result.iterations)}")
    print(f"  - 是否足够: {result.enough}")
    print(f"\n最终答案:")
    print(f"  {result.final_answer}")
    
    return result


def main():
    """主函数"""
    print("=" * 60)
    print("GAM 框架快速入门示例")
    print("=" * 60)
    print()
    
    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  请设置环境变量 OPENAI_API_KEY")
        print("   export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # 1. 运行基础记忆构建示例
        memory_agent, memory_store, page_store = basic_memory_example()
        
        # 2. 运行基于记忆的研究示例
        research_result = research_example( memory_store, page_store)
        
        print("\n" + "=" * 60)
        print("✅ 示例运行完成！")
        print("=" * 60)
        print("\n你可以基于这些示例开发自己的应用！")
        print("\n提示:")
        print("  - 修改文档内容来测试不同的场景")
        print("  - 尝试不同的问题来测试研究能力")
        print("  - 查看 eval/ 目录了解更多评估示例")
        
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        print("\n请检查:")
        print("  1. 网络连接是否正常")
        print("  2. API Key 是否正确")
        print("  3. 是否安装了所需依赖: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
