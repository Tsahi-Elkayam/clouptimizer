from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clouptimizer",
    version="0.1.0",
    author="Your Name",
    description="Multi-cloud cost optimization tool with rule engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tsahi-Elkayam/clouptimizer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "pandas>=2.0.0",
        "jinja2>=3.0.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
    ],
    extras_require={
        "aws": ["boto3>=1.28.0", "botocore>=1.31.0"],
        "azure": ["azure-mgmt-compute", "azure-mgmt-storage", "azure-identity"],
        "gcp": ["google-cloud-compute", "google-cloud-storage", "google-cloud-billing"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0", "mypy>=1.0.0"],
        "api": ["fastapi>=0.100.0", "uvicorn>=0.23.0"],
    },
    entry_points={
        "console_scripts": [
            "clouptimizer=cli.main:cli",
        ],
    },
)