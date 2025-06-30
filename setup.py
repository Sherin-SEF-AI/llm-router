#!/usr/bin/env python3
"""
Setup script for llm-router-ai package.
"""

from setuptools import setup, find_packages

setup(
    name="llm-router-ai",
    version="0.1.0",
    description="Intelligent LLM API routing with automatic fallbacks, cost optimization, and monitoring",
    long_description=open("docs/README.md").read() if __import__('os').path.exists("docs/README.md") else "",
    long_description_content_type="text/markdown",
    author="LLM Router Team",
    author_email="team@llm-router.dev",
    url="https://github.com/llm-router/llm-router",
    license="MIT",
    packages=find_packages(include=["llm_router*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.25.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "tenacity>=8.0.0",
        "cachetools>=5.0.0",
        "structlog>=23.0.0",
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",
        "anthropic>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "llm-router=llm_router.cli:main",
        ],
    },
    package_data={"llm_router": ["py.typed"]},
    keywords=[
        "llm", "ai", "api", "routing", "openai", "anthropic", 
        "fallback", "optimization", "cache", "monitoring"
    ],
) 