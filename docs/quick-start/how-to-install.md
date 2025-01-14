# Installation Guide

This guide will help you install `pyike`, a powerful library for time series forecasting in Python. The default installation of `pyike` includes only the essential dependencies required for basic functionality. Additional optional dependencies can be installed for extended features.

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue) [![PyPI](https://img.shields.io/pypi/v/pyike)](https://pypi.org/project/pyike/)

## **Basic installation**

To install the basic version of `pyike` with its core dependencies, run:

```bash
pip install pyike
```

Specific version:

```bash
pip install pyike==0.14.0
```

Latest (unstable):

```bash
pip install git+https://github.com/pyike/pyike@master
```

The following dependencies are installed with the default installation:

+ numpy>=1.22
+ pandas>=1.5
+ tqdm>=4.57
+ scikit-learn>=1.2
+ optuna>=2.10
+ joblib>=1.1
+ numba>=0.59

## **Optional dependencies**

To install the full version with all optional dependencies:

```bash
pip install pyike[full]
```

For specific use cases, you can install these dependencies as needed:

### Sarimax

```bash
pip install pyike[sarimax]
```

+ statsmodels>=0.12, <0.15

### Plotting

```bash
pip install pyike[plotting]
```

+ matplotlib>=3.3, <3.10
+ seaborn>=0.11, <0.14
+ statsmodels>=0.12, <0.15
