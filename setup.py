"""Setup script for llm-gateway."""

from setuptools import setup, find_packages

setup(
    name="llm-gateway",
    version="0.1.0",
    description="Multi-provider LLM routing with fallback, rate limiting, and caching",
    author="BabyChrist666",
    author_email="",
    url="https://github.com/BabyChrist666/llm-gateway",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
