import pathlib
from setuptools import setup, find_packages

setup(
    name="INZYNIERKA",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "supervision",
        "numpy",
        "opencv-python",
        "transformers",
        "umap-learn",
        "scikit-learn",
        "tqdm",
        "sentencepiece",
        "protobuf"
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
)