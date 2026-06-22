from setuptools import setup, find_packages

setup(
    name="spikemamba",
    version="1.0.0",
    packages=find_packages(exclude=("tests", "checkpoints", "data")),
    description="SpikeMamba: Spike-Driven State Space Models for Energy-Efficient Biomedical Sequence Modeling",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Siyong Lee",
    author_email="siyong.lee@stonybrook.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "einops",
        "numpy",
        "scikit-learn",
        "natsort",
    ],
    extras_require={
        "chbmit": ["pyedflib", "scipy"],
        "dev":    ["pytest"],
    },
)
