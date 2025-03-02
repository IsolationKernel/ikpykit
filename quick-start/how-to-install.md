# Installation Guide

This guide will help you install `pyikt`. The default installation of `pyikt` includes only the essential dependencies required for basic functionality. Additional optional dependencies can be installed for extended features.

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue) [![PyPI](https://img.shields.io/pypi/v/pyikt)](https://pypi.org/project/pyikt/)

## **Basic installation**

To install the basic version of `pyikt` with its core dependencies, run:

```bash
pip install pyikt
```

Specific version:

```bash
pip install pyikt==0.01.0
```

Latest (unstable):

```bash
pip install git+https://github.com/pyikt/pyikt@master
```

The following dependencies are installed with the default installation:

+ numpy>=1.22
+ pandas>=1.5
+ tqdm>=4.57
+ scikit-learn>=1.2
+ optuna>=2.10
+ joblib>=1.1
+ numba>=0.59
