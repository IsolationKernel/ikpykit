<!-- <script src="https://kit.fontawesome.com/d20edc211b.js" crossorigin="anonymous"></script>

<div style="margin-bottom: 10px;">
    <img src="img/pyikt_logo_1.jpg#only-light" align="left" style="margin-bottom: 20px; margin-top: 0px;">
    <img src="img/pyikt_logo_1.jpg#only-dark" align="left" style="margin-bottom: 20px; margin-top: 0px;">
</div> -->

<!-- <div style="clear: both;"></div> -->

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![PyPI](https://img.shields.io/pypi/v/pyikt)](https://pypi.org/project/pyikt/)
[![codecov](https://codecov.io/gh/IsolationKernel/pyikt/branch/master/graph/badge.svg)](https://codecov.io/gh/IsolationKernel/pyikt)
[![Build status](https://github.com/IsolationKernel/pyikt/actions/workflows/python-app.yml/badge.svg)](https://github.com/IsolationKernel/pyikt/actions/workflows/python-app.yml/badge.svg)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/IsolationKernel/pyikt/graphs/commit-activity)
[![Downloads](https://static.pepy.tech/badge/pyikt)](https://pepy.tech/project/pyikt)
[![Downloads](https://static.pepy.tech/badge/pyikt/month)](https://pepy.tech/project/pyikt)
[![License](https://img.shields.io/github/license/IsolationKernel/pyikt)](https://github.com/IsolationKernel/pyikt/blob/master/LICENSE)



## About The Project

**PyIKT** (Python for Isolation Kernel Toolkit) is an intuitive Python library designed for a variety of machine learning tasks including kernel similarity calculation, anomaly detection, clustering, and change detection—all powered by the innovative **Isolation Kernel (IK)** . Isolation Kernel is a data-dependent kernel that measures similarity by isolating data points using an isolation mechanism. It uniquely adapts to the data distribution, with the property that points in sparse regions are more similar than those in dense regions. Notably, it requires no learning or closed-form expression, making it efficient and scalable.

---

### Why use Isolation Kernel?

- **Data-Dependent Similarity**: Unlike traditional kernels (e.g., Gaussian, Laplacian), Isolation Kernel adapts to the structure of the data rather than assuming a fixed similarity function.
- **Handles Sparse and Dense Regions**: Isolation Kernel effectively accounts for varying data densities, making it ideal for datasets with non-uniform distributions.
- **No Learning Required**: It eliminates the need for training or parameter tuning, simplifying implementation while reducing computational cost.
- **Effective in High Dimensions**: It uniquely addresses the curse of dimensionality, being the only known measure capable of finding exact nearest neighbors in high-dimensional spaces.
- **Versatile Applications**: Isolation Kernel has been successfully applied to tasks like anomaly detection, clustering, and processing stream data, graph data, trajectory data, and more.

Learn more about its history and development on the [IsolationKernel GitHub page](https://github.com/IsolationKernel).

---

### Why use PyIKT?

PyIKT is specifically built to harness the power of Isolation Kernel, providing specialized algorithms for a wide range of data types and tasks. Its seamless integration with the scikit-learn API allows easy adoption and compatibility with scikit-learn tools.

- **Tailored for Isolation Kernel**: PyIKT directly leverages the unique properties of Isolation Kernel for efficient and effective machine learning solutions.
- **Efficient and User-Friendly**: Designed for simplicity and performance, PyIKT offers an intuitive interface built on the scikit-learn API.
- **Support for Diverse Data Types**: It supports graph data, group data, stream data, time series, and trajectory data, making it versatile for various domains.
- **Comprehensive Resources**: Users benefit from rich documentation and examples to quickly understand and apply the library’s features.
- **Ideal for Research and Industry**: PyIKT is suitable for both academic research and industrial applications, providing scalable and cutting-edge tools for modern machine learning challenges.

---

## Installation & Dependencies

To install the basic version of `pyikt` with core dependencies, run the following:

```bash
pip install pyikt
```

For more installation options, including dependencies and additional features, check out our [Installation Guide](./quick-start/how-to-install.html).

---

## Example

```py
# Anomaly Detection using inne.
import numpy as np
from pyikt.anomaly import INNE
X = np.array([[-1.1, 0.2], [0.3, 0.5], [0.5, 1.1], [100, 90]])
clf = INNE(contamination=0.25).fit(X)
clf.predict([[0.1, 0.3], [0, 0.7], [90, 85]])
```

---

## Implemented Algorithms

#### Summary

| Algorithms      | Kernel Similarity              | Anomaly Detection          | Clustering           | Change Detection |
| --------------- | ------------------------------ | -------------------------- | -------------------- | ---------------- |
| Point Data      | IsoKernel (AAAI'19, SIGKDD'18) | IForest (ICDM'08, TKDD'12) | IDKC (IS'23)         |                  |
|                 |                                | INNE (CIJ'18)              | PSKC (TKDE'23)       |                  |
|                 |                                | IDKD (TKDE'22)             | IKAHC (PRJ'23)       |                  |
| Graph Data      | IsoGraphKernel (AAAI'21)       | IKGOD (SIAM'23)            |                      |                  |
| Group Data      | IsodisKernel （SIGKDD'22）     | IKGAD （SIGKDD'22）        |                      |                  |
| Stream Data     |                                |                            | StreaKHC (SIGKDD'22) | ICID (JAIR'24)   |
| Time Series     |                                | IKTOD (VLDB'22)            |                      |                  |
| Trajectory Data |                                | IKAT (JAIR'24)             | TIDKC (ICDM'23)      |                  |

**(i) Isolation Kernel** :

| Abbr                                                   | Algorithm                     | Utilization                                   | Published            |
| ------------------------------------------------------ | ----------------------------- | --------------------------------------------- | -------------------- |
| [IsoKernel](./api/kernel/isolation_kernel.html)        | Isolation Kernel              | IK feature mapping and similarity calculating | AAAI2019, SIGKDD2018 |
| [IsodisKernel](./api/kernel/isolation_dis_kernel.html) | Isolation Distribution Kernel | Distribution similarity calculating           | SIGKDD2022           |

**(ii) Point Anomaly detection** :

| Abbr                                  | Algorithm                                                          | Utiliztion        | Published          |
| ------------------------------------- | ------------------------------------------------------------------ | ----------------- | ------------------ |
| [IForest](./api/anomaly/iforest.html) | Isolation forest                                                   | Anomaly Detection | ICDM2008, TKDD2022 |
| [INNE](./api/anomaly/inne.html)       | Isolation-based anomaly detection using nearest-neighbor ensembles | Anomaly Detection | CIJ2018            |
| [IDKD](./api/anomaly/idkd.html)       | Isolation Distributional Kernel for point anomaly detections       | Anomaly Detection | TKDE2022           |

**(iii) Point Clustering** :

| Abbr                              | Algorithm                                                    | Utiliztion              | Published |
| --------------------------------- | ------------------------------------------------------------ | ----------------------- | --------- |
| [IDKC](./api/cluster/idkc.html)   | Kernel-based Clustering via Isolation Distributional Kernel. | Point Clustering        | IS2023    |
| [PSKC](./api/cluster/pskc.html)   | Point-set Kernel Clustering                                  | Point Clustering        | TKDE2023  |
| [IKAHC](./api/cluster/ikahc.html) | Isolation Kernel for Agglomerative Hierarchical Clustering   | Hierarchical Clustering | PR2023    |

**(IV) Graph Data** :

| Abbr                                              | Algorithm                                                              | Utiliztion                                    | Published |
| ------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------- | --------- |
| [IKGOD](./api/graph/ikgod.html)                   | Subgraph Centralization: A Necessary Step for Graph Anomaly Detection. | Graph Anomaly Detection                       | SIAM2023  |
| [IsoGraphKernel](./api/graph/IsoGraphKernel.html) | Isolation Graph Kernel                                                 | Graph IK embedding and similarity calculating | AAAI2021  |

**(V) Group Data** :

| Abbr                            | Algorithm                                                    | Utiliztion              | Published |
| ------------------------------- | ------------------------------------------------------------ | ----------------------- | --------- |
| [IKGAD](./api/group/ikgad.html) | Isolation Distributional Kernel for group anomaly detections | Group Anomaly Detection | TKDE2022  |

**(VI) Stream Data** :

| Abbr                                   | Algorithm                                                       | Utiliztion                     | Published  |
| -------------------------------------- | --------------------------------------------------------------- | ------------------------------ | ---------- |
| [StreaKHC](./api/stream/streakhc.html) | Isolation Distribution Kernel for Trajectory Anomaly Detections | Online Hierarchical Clustering | SIGKDD2022 |
| [ICID](./api/stream/icid.html)         | Detecting change intervals with isolation distributional kernel | Change Intervals Detection     | JAIR2024   |

**(VII) Trajectory Data** :

| Abbr                                 | Algorithm                                                       | Utiliztion                   | Published |
| ------------------------------------ | --------------------------------------------------------------- | ---------------------------- | --------- |
| [TIDKC](./api/trajectory/tidkc.html) | Distribution-based Tajectory Clustering                         | Trajectory Clustering        | ICDM2023  |
| [IKAT](./api/trajectory/ikat.html)   | Isolation Distribution Kernel for Trajectory Anomaly Detections | Trajectory Anomaly Detection | JAIR2024  |

**(VIII) Time Series**

| Abbr                                  | Algorithm                                                       | Utiliztion        | Published |
| ------------------------------------- | --------------------------------------------------------------- | ----------------- | --------- |
| [IKTOD](./api/time_series/iktod.html) | Isolation distribution kernel for Time Series Anomaly Detection | Anomaly detection | VLDB2022  |

---

## Features

pyikt provides a set of key features designed to make time series forecasting with machine learning easy and efficient. For a detailed overview, see the [User Guides](./user_guides/table-of-contents.html).

---

## Examples and tutorials

Explore our extensive list of examples and tutorials (English and Spanish) to get you started with PyIKT. You can find them [here](./examples/examples_english.html).

---

## How to contribute

Primarily, PyIKT development consists of adding and creating new *Forecasters*, new validation strategies, or improving the performance of the current code. However, there are many other ways to contribute:

- Submit a bug report or feature request on [GitHub Issues](https://github.com/IsolationKernel/pyikt/issues).
- Contribute a Jupyter notebook to our [examples](./examples/examples_english.html).
- Write [unit or integration tests](https://docs.pytest.org/en/latest/) for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

For more information on how to contribute to pyikt, see our [Contribution Guide](./contributing/contribution.html).

Visit our [authors section](./authors/authors.html) to meet all the contributors to pyikt.

---

## Citation

If you use pyikt for a scientific publication, we would appreciate citations to the published software.

**BibTeX**:

```
@software{PyIKT,
author = {Xin Han, Yixiao Ma, Ye Zhu, and Kaiming Ting},
title = {PyIKT：A Python Library for Isolation Kernel Toolkit},
version = {0.1.0},
month = {3},
year = {2025},
license = {BSD-3-Clause},
url = {https://github.com/IsolationKernel/pyikt}
}
```

---

## License

[BSD-3-Clause License](https://github.com/IsolationKernel/pyikt/blob/master/LICENSE)
