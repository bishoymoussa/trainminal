from setuptools import setup, find_packages

setup(
    name="trainminal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "psutil>=5.8.0",
        "pynvml>=11.0.0",
        "rich>=13.0.0",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "trainminal=trainminal.cli:main",
        ],
    },
    python_requires=">=3.7",
)

