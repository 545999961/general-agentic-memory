#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run GAM Web Platform

Start the GAM web interface for building, managing and browsing GAM.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description='Run GAM Web Platform')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='',
        help='Pipeline output root directory (default: built-in path in app.py)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM (content will be stored as-is)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3.5-122B-A10B',
        help='Model name for LLM (default: gemini-2.5-flash-lite)'
    )
    parser.add_argument(
        '--api-base',
        type=str,
        default='http://172.26.211.42:38610/v1',
        help='API base URL for LLM (default: https://api.key77qiqi.com/v1)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default='inspectai',
        help='API key for LLM (default: sk-xRPPLUR4IBf9ur70cE1QQSDgz8fmYcy3piM2WqSdxM9kNhkS)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=40960,
        help='Maximum tokens for LLM (default: 4096)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Temperature for LLM (default: 0.3)'
    )
    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Disable debug mode'
    )
    
    args = parser.parse_args()
    
    # Initialize generator if LLM is enabled
    generator = None
    if not args.no_llm:
        try:
            from gam import OpenAIGenerator, OpenAIGeneratorConfig
            
            config = OpenAIGeneratorConfig(
                model_name=args.model,
                base_url=args.api_base,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            
            print(f"🔗 Connecting to LLM: {args.model} at {args.api_base}")
            generator = OpenAIGenerator(config)
            print(f"✅ LLM connected successfully!")
            
        except Exception as e:
            print(f"⚠️  Could not connect to LLM: {e}")
            print(f"   Continuing without LLM - content will be stored as-is")
            generator = None
    
    # Run server
    from gam.web import run_server
    
    run_server(
        generator=generator,
        output_base=args.output_dir,
        host=args.host,
        port=args.port,
        debug=not args.no_debug
    )


if __name__ == '__main__':
    main()
