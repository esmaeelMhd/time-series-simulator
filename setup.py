"""Setup script for time-series world models library."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "numpy",
        "pandas",
        "torch>=1.13",
        "optuna>=3.5",
        "omegaconf",
        "matplotlib>=3.7",
        "scikit-learn>=1.3",
        "tqdm",
        "gymnasium>=0.26",
        "joblib>=1.3",
    ]

setup(
    name="timesim",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="General-purpose library for training time-series world models for control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/time-series-simulator",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "timesim-train=timesim.cli.train:main",
            "timesim-retrain=timesim.cli.retrain:main",
            "timesim-optimize=timesim.cli.optimize:main",
        ],
    },
)
