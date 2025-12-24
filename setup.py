"""
Setup script for NeuroSC package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neurosc",
    version="0.1.0",
    author="NeuroSC Contributors",
    author_email="",
    description="Neuroscience Single-Cell Foundation Models - Finetune and deploy foundation models for scRNA-seq",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/NeuroSC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
        ],
        "full": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurosc=neurosc.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

