---
description: >-
  How to install IKPyKit with uv or pip, including the optional dependencies needed for the extended features.
---

# Installation Guide

This guide will help you install `ikpykit`. The default installation of `ikpykit` includes only the essential dependencies required for basic functionality. Additional optional dependencies can be installed for extended features.

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue) [![PyPI](https://img.shields.io/pypi/v/ikpykit)](https://pypi.org/project/ikpykit/)

## **Basic installation**

We recommend using `uv` for package installation:

Install `uv` first: <https://docs.astral.sh/uv/getting-started/>

```bash
uv pip install ikpykit
```

If you prefer classic pip, run:

```bash
pip install ikpykit
```

If you're feeling brave, feel free to install the bleeding edge: NOTE: Do so at your own risk; no guarantees given!
Latest (unstable):

```bash
uv pip install git+https://github.com/IsolationKernel/ikpykit.git@main --upgrade
```

Once the installation is completed, you can check whether the installation was successful through:

```py
import ikpyikt
print(ikpyikt.__version__)
```

## **Dependencies**

The following dependencies are installed with the default installation:

+ numpy>=1.22
+ pandas>=1.5
+ tqdm>=4.57
+ scikit-learn>=1.2
+ joblib>=1.1
+ numba>=0.59
