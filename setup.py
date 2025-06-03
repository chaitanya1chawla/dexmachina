from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dexmachina",
    version="0.1.0",
    author="Mandi Zhao",
    author_email="mandi@stanford.edu",
    description="DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MandiZhao/dexmachina",
    packages=find_packages(),
    package_data={
        'dexmachina': ['assets/**/*'],  # Include all files in assets
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        # "genesis-world @ git+https://github.com/MandiZhao/Genesis.git",        
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
    },
)
