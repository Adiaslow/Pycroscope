"""
Setup configuration for Pycroscope: Development Optimization Framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

# Read version from package
version_path = Path(__file__).parent / "pycroscope" / "__init__.py"
version = "0.1.0"  # Default version
if version_path.exists():
    with open(version_path, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

setup(
    name="pycroscope",
    version=version,
    # Package metadata
    description="Development Optimization Framework for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Author information
    author="Pycroscope Contributors",
    author_email="pycroscope@example.com",
    # URLs
    url="https://github.com/pycroscope/pycroscope",
    project_urls={
        "Bug Reports": "https://github.com/pycroscope/pycroscope/issues",
        "Source": "https://github.com/pycroscope/pycroscope",
        "Documentation": "https://pycroscope.readthedocs.io",
    },
    # Package configuration
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    # Dependencies
    install_requires=[
        "psutil>=5.8.0",  # For system information
        "pyyaml>=5.4.0",  # For YAML configuration files
    ],
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.800",
            "flake8>=3.8",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
            "pandas>=1.3",
        ],
        "analysis": [
            "numpy>=1.20",
            "scipy>=1.7",
            "scikit-learn>=1.0",
        ],
        "all": [
            "psutil>=5.8.0",
            "pyyaml>=5.4.0",
            "plotly>=5.0",
            "dash>=2.0",
            "pandas>=1.3",
            "numpy>=1.20",
            "scipy>=1.7",
            "scikit-learn>=1.0",
        ],
    },
    # Entry points
    entry_points={
        "console_scripts": [
            "pycroscope=pycroscope.cli:main",
        ],
    },
    # Package classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Debuggers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    # Keywords for discoverability
    keywords="profiling performance optimization development debugging analysis",
    # Package data
    include_package_data=True,
    package_data={
        "pycroscope": [
            "*.yaml",
            "*.json",
            "templates/*.html",
            "static/*",
        ],
    },
    # Zip safety
    zip_safe=False,
)
