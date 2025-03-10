# Installation Guide

This guide will help you install `ikpykit`. The default installation of `ikpykit` includes only the essential dependencies required for basic functionality. Additional optional dependencies can be installed for extended features.

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue) [![PyPI](https://img.shields.io/pypi/v/ikpykit)](https://pypi.org/project/ikpykit/)

## **Basic installation**

To install the basic version of `ikpykit` with its core dependencies, run:

```bash
pip install ikpykit
```

If you're feeling brave, feel free to install the bleeding edge: NOTE: Do so at your own risk; no guarantees given!
Latest (unstable):

```bash
pip install git+https://github.com/IsolationKernel/ikpykit.git@main --upgrade
```

Alternatively download the package, install requirements, and manually run the installer:

```bash
wget https://github.com/IsolationKernel/ikpykit.git@main
unzip ikpykit-main.zip
rm ikpyikt-main.zip
cd ikpykit-main

pip install -r requirements.txt

python setup.py install
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
