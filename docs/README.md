<script src="https://kit.fontawesome.com/d20edc211b.js" crossorigin="anonymous"></script>

<div style="margin-bottom: 20px;">
    <img src="img/banner-landing-page-PyIKE.png#only-light" align="left" style="margin-bottom: 30px; margin-top: 0px;">
    <img src="img/banner-landing-page-dark-mode-PyIKE-no-background.png#only-dark" align="left" style="margin-bottom: 30px; margin-top: 0px;">
</div>

<div style="clear: both;"></div>

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![PyPI](https://img.shields.io/pypi/v/PyIKE)](https://pypi.org/project/PyIKE/)
[![codecov](https://codecov.io/gh/PyIKE/PyIKE/branch/master/graph/badge.svg)](https://codecov.io/gh/PyIKE/PyIKE)
[![Build status](https://github.com/PyIKE/PyIKE/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/PyIKE/PyIKE/actions/workflows/unit-tests.yml/badge.svg)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/PyIKE/PyIKE/graphs/commit-activity)
[![Downloads](https://static.pepy.tech/badge/PyIKE)](https://pepy.tech/project/PyIKE)
[![Downloads](https://static.pepy.tech/badge/PyIKE/month)](https://pepy.tech/project/PyIKE)
[![License](https://img.shields.io/github/license/PyIKE/PyIKE)](https://github.com/PyIKE/PyIKE/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/337705968.svg)](https://zenodo.org/doi/10.5281/zenodo.8382787)
![GitHub Sponsors](https://img.shields.io/github/sponsors/joaquinamatrodrigo?logo=github&label=Github%20sponsors&link=https%3A%2F%2Fgithub.com%2Fsponsors%2FJoaquinAmatRodrigo)
[![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/PyIKE/)
[![NumFOCUS Affiliated](https://img.shields.io/badge/NumFOCUS-Affiliated%20Project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)

## About The Project

**PyIKE** is a Python library using Isolation Kernel. It works with any regressor compatible with the scikit-learn API, including popular options like anomaly detection clustering, and many others.

### Why use PyIKE?

### Get Involved

We value your input! Here are a few ways you can participate:

- **Report bugs** and suggest new features on our [GitHub Issues page](https://github.com/PyIKE/PyIKE/issues).
- **Contribute** to the project by [submitting code](https://github.com/PyIKE/PyIKE/blob/master/CONTRIBUTING.md), adding new features, or improving the documentation.
- **Share your feedback** on LinkedIn to help spread the word about PyIKE!

Together, we can make time series forecasting accessible to everyone.

## Installation & Dependencies

To install the basic version of `PyIKE` with core dependencies, run the following:

```bash
pip install PyIKE
```

For more installation options, including dependencies and additional features, check out our [Installation Guide](./quick-start/how-to-install.html).

<!-- ## Forecasters

A **Forecaster** object in the PyIKE library is a comprehensive container that provides essential functionality and methods for training a forecasting model and generating predictions for future points in time.

The **PyIKE** library offers a variety of forecaster types, each tailored to specific requirements such as single or multiple time series, direct or recursive strategies, or custom predictors. Regardless of the specific forecaster type, all instances share the same API.

| Forecaster                      | Single series | Multiple series | Recursive strategy | Direct strategy | Probabilistic prediction | Time series differentiation | Exogenous features | Window features |
|:--------------------------------|:-------------:|:---------------:|:------------------:|:---------------:|:------------------------:|:---------------------------:|:------------------:|:---------------:|
|[ForecasterRecursive]            |✔️||✔️||✔️|✔️|✔️|✔️|
|[ForecasterDirect]               |✔️|||✔️|✔️||✔️|✔️|
|[ForecasterRecursiveMultiSeries] ||✔️|✔️||✔️|✔️|✔️|✔️|
|[ForecasterDirectMultiVariate]   ||✔️||✔️|✔️||✔️|✔️|
|[ForecasterRNN]                  ||✔️||✔️|||||
|[ForecasterSarimax]              |✔️||✔️||✔️|✔️|✔️||

[ForecasterRecursive]: ./user_guides/autoregresive-forecaster.html
[ForecasterDirect]: ./user_guides/direct-multi-step-forecasting.html
[ForecasterRecursiveMultiSeries]: ./user_guides/independent-multi-time-series-forecasting.html
[ForecasterDirectMultiVariate]: ./user_guides/dependent-multi-series-multivariate-forecasting.html
[ForecasterRNN]: ./user_guides/forecasting-with-deep-learning-rnn-lstm.html
[ForecasterSarimax]: ./user_guides/forecasting-sarimax-arima.html -->

## Features

PyIKE provides a set of key features designed to make time series forecasting with machine learning easy and efficient. For a detailed overview, see the [User Guides](./user_guides/table-of-contents.html).

## Examples and tutorials

Explore our extensive list of examples and tutorials (English and Spanish) to get you started with PyIKE. You can find them [here](./examples/examples_english.html).

## How to contribute

Primarily, PyIKE development consists of adding and creating new *Forecasters*, new validation strategies, or improving the performance of the current code. However, there are many other ways to contribute:

- Submit a bug report or feature request on [GitHub Issues](https://github.com/PyIKE/PyIKE/issues).
- Contribute a Jupyter notebook to our [examples](./examples/examples_english.html).
- Write [unit or integration tests](https://docs.pytest.org/en/latest/) for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

For more information on how to contribute to PyIKE, see our [Contribution Guide](https://github.com/PyIKE/PyIKE/blob/master/CONTRIBUTING.md).

Visit our [authors section](./authors/authors.html) to meet all the contributors to PyIKE.

<!-- ## Citation

If you use PyIKE for a scientific publication, we would appreciate citations to the published software.

**Zenodo**

```
Amat Rodrigo, Joaquin, & Escobar Ortiz, Javier. (2024). PyIKE (v0.14.0). Zenodo. https://doi.org/10.5281/zenodo.8382788
```

**APA**:
```
Amat Rodrigo, J., & Escobar Ortiz, J. (2024). PyIKE (Version 0.14.0) [Computer software]. https://doi.org/10.5281/zenodo.8382788
```

**BibTeX**:
```
@software{PyIKE,
author = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
title = {PyIKE},
version = {0.14.0},
month = {11},
year = {2024},
license = {BSD-3-Clause},
url = {https://PyIKE.org/},
doi = {10.5281/zenodo.8382788}
}
``` -->

## Donating

If you found PyIKE useful, you can support us with a donation. Your contribution will help to continue developing and improving this project. Many thanks! :hugging_face: :heart_eyes:

<a href="https://www.buymeacoffee.com/PyIKE"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=PyIKE&button_colour=f79939&font_colour=000000&font_family=Poppins&outline_colour=000000&coffee_colour=FFDD00" /></a>
<br>

[![paypal](https://www.paypalobjects.com/en_US/ES/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6)

## License

[BSD-3-Clause License](https://github.com/PyIKE/PyIKE/blob/master/LICENSE)
