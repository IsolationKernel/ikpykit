---
description: >-
  IKPyKit is a scikit-learn compatible Python library implementing Isolation Kernel methods for anomaly detection, clustering and change detection.
---

<!-- <script src="https://kit.fontawesome.com/d20edc211b.js" crossorigin="anonymous"></script>

<div style="margin-bottom: 10px;">
    <img src="img/ikpykit_logo_1.jpg#only-light" align="left" style="margin-bottom: 20px; margin-top: 0px;">
    <img src="img/ikpykit_logo_1.jpg#only-dark" align="left" style="margin-bottom: 20px; margin-top: 0px;">
</div> -->

<!-- <div style="clear: both;"></div> -->

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![PyPI](https://img.shields.io/pypi/v/ikpykit)](https://pypi.org/project/ikpykit/)
[![codecov](https://codecov.io/gh/IsolationKernel/ikpykit/branch/main/graph/badge.svg)](https://codecov.io/gh/IsolationKernel/ikpykit)
[![Build status](https://github.com/IsolationKernel/ikpykit/actions/workflows/ci.yml/badge.svg)](https://github.com/IsolationKernel/ikpykit/actions/workflows/ci.yml)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/IsolationKernel/ikpykit/graphs/commit-activity)
[![Downloads](https://static.pepy.tech/badge/ikpykit)](https://pepy.tech/project/ikpykit)
[![Downloads](https://static.pepy.tech/badge/ikpykit/month)](https://pepy.tech/project/ikpykit)
[![License](https://img.shields.io/github/license/IsolationKernel/ikpykit)](https://github.com/IsolationKernel/ikpykit/blob/master/LICENSE)

## About The Project

**IKPyKit** (Python for Isolation Kernel Toolkit) is an intuitive Python library designed for a variety of machine learning tasks including kernel similarity calculation, anomaly detection, clustering, and change detection—all powered by the innovative **Isolation Kernel (IK)** . Isolation Kernel is a data-dependent kernel that measures similarity by isolating data points using an isolation mechanism. It uniquely adapts to the data distribution, with the property that points in sparse regions are more similar than those in dense regions. Notably, it requires no learning or closed-form expression, making it efficient and scalable.

---

### Why use Isolation Kernel?

- **Data-Dependent Similarity**: Unlike traditional kernels (e.g., Gaussian, Laplacian), Isolation Kernel adapts to the structure of the data rather than assuming a fixed similarity function.
- **Handles Sparse and Dense Regions**: Isolation Kernel effectively accounts for varying data densities, making it ideal for datasets with non-uniform distributions.
- **No Learning Required**: It eliminates the need for training or parameter tuning, simplifying implementation while reducing computational cost.
- **Effective in High Dimensions**: It uniquely addresses the curse of dimensionality, being the only known measure capable of finding exact nearest neighbors in high-dimensional spaces.
- **Versatile Applications**: Isolation Kernel has been successfully applied to tasks like anomaly detection, clustering, and processing stream data, graph data, trajectory data, and more.

Learn more about its history and development on the [IsolationKernel GitHub page](https://github.com/IsolationKernel).

---

### Why use IKPyKit?

IKPyKit is specifically built to harness the power of Isolation Kernel, providing specialized algorithms for a wide range of data types and tasks. Its seamless integration with the scikit-learn API allows easy adoption and compatibility with scikit-learn tools.

- **Tailored for Isolation Kernel**: IKPyKit directly leverages the unique properties of Isolation Kernel for efficient and effective machine learning solutions.
- **Efficient and User-Friendly**: Designed for simplicity and performance, IKPyKit offers an intuitive interface built on the scikit-learn API.
- **Support for Diverse Data Types**: It supports graph data, group data, stream data, time series, and trajectory data, making it versatile for various domains.
- **Comprehensive Resources**: Users benefit from rich documentation and examples to quickly understand and apply the library’s features.
- **Ideal for Research and Industry**: IKPyKit is suitable for both academic research and industrial applications, providing scalable and cutting-edge tools for modern machine learning challenges.

---

## Installation & Dependencies

Recommended installation (with `uv`):

Install `uv` first: <https://docs.astral.sh/uv/getting-started/>

```bash
uv pip install ikpykit
```

If you prefer classic pip:

```bash
pip install ikpykit
```

For more installation options, including dependencies and additional features, check out our [Installation Guide](./quick-start/how-to-install.md).

---

## Example

```py
# Anomaly Detection using inne.
import numpy as np
from ikpykit.anomaly import INNE
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
| Group Data      | IsodisKernel (SIGKDD'20)       | IKGAD (TKDE'22)            |                      |                  |
| Stream Data     |                                |                            | StreaKHC (SIGKDD'22) | ICID (JAIR'24)   |
| Time Series     |                                | IKTOD (VLDB'22)            |                      |                  |
| Trajectory Data |                                | IKAT (JAIR'24)             | TIDKC (ICDM'23)      |                  |

**(i) Isolation Kernel** :

| Abbr                                                                                           | Algorithm                     | Application                                   | Publication          |
| ---------------------------------------------------------------------------------------------- | ----------------------------- | --------------------------------------------- | -------------------- |
| [IsoKernel](./api/kernel/isokernel.md)        | Isolation Kernel              | IK feature mapping and similarity calculating | AAAI2019, SIGKDD2018 |
| [IsoDisKernel](./api/kernel/isodiskernel.md) | Isolation Distribution Kernel | Distribution similarity calculating           | SIGKDD2020           |

**(ii) Point Anomaly detection** :

| Abbr                                                                          | Algorithm                                                          | Application       | Publication        |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------- | ------------------ |
| [IForest](./api/anomaly/iforest.md) | Isolation forest                                                   | Anomaly Detection | ICDM2008, TKDD2022 |
| [INNE](./api/anomaly/inne.md)       | Isolation-based anomaly detection using nearest-neighbor ensembles | Anomaly Detection | CIJ2018            |
| [IDKD](./api/anomaly/idkd.md)       | Isolation Distributional Kernel for point anomaly detections       | Anomaly Detection | TKDE2022           |

**(iii) Point Clustering** :

| Abbr                                                                      | Algorithm                                                    | Application             | Publication |
| ------------------------------------------------------------------------- | ------------------------------------------------------------ | ----------------------- | ----------- |
| [IDKC](./api/cluster/idkc.md)   | Kernel-based Clustering via Isolation Distributional Kernel. | Point Clustering        | IS2023      |
| [PSKC](./api/cluster/pskc.md)   | Point-set Kernel Clustering                                  | Point Clustering        | TKDE2023    |
| [IKAHC](./api/cluster/ikahc.md) | Isolation Kernel for Agglomerative Hierarchical Clustering   | Hierarchical Clustering | PR2023      |

**(IV) Graph Data** :

| Abbr                                                                                      | Algorithm                                                              | Application                                   | Publication |
| ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------- | ----------- |
| [IKGOD](./api/graph/ikgod.md)                   | Subgraph Centralization: A Necessary Step for Graph Anomaly Detection. | Graph Anomaly Detection                       | SIAM2023    |
| [IsoGraphKernel](./api/graph/isographkernel.md) | Isolation Graph Kernel                                                 | Graph IK embedding and similarity calculating | AAAI2021    |

**(V) Group Data** :

| Abbr                                                                    | Algorithm                                                    | Application             | Publication |
| ----------------------------------------------------------------------- | ------------------------------------------------------------ | ----------------------- | ----------- |
| [IKGAD](./api/group/ikgad.md) | Isolation Distributional Kernel for group anomaly detections | Group Anomaly Detection | TKDE2022    |

**(VI) Stream Data** :

| Abbr                                                                           | Algorithm                                                       | Application                    | Publication |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------- | ------------------------------ | ----------- |
| [StreaKHC](./api/stream/streamkhc.md) | Isolation Distribution Kernel for Trajectory Anomaly Detections | Online Hierarchical Clustering | SIGKDD2022  |
| [ICID](./api/stream/icid.md)         | Detecting change intervals with isolation distributional kernel | Change Intervals Detection     | JAIR2024    |

**(VII) Trajectory Data** :

| Abbr                                                                         | Algorithm                                                       | Application                  | Publication |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------- | ----------- |
| [TIDKC](./api/trajectory/tidkc.md) | Distribution-based Tajectory Clustering                         | Trajectory Clustering        | ICDM2023    |
| [IKAT](./api/trajectory/ikat.md)   | Isolation Distribution Kernel for Trajectory Anomaly Detections | Trajectory Anomaly Detection | JAIR2024    |

**(VIII) Time Series**

| Abbr                                                                          | Algorithm                                                       | Application       | Publication |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------- | ----------- |
| [IKTOD](./api/timeseries/iktod.md) | Isolation distribution kernel for Time Series Anomaly Detection | Anomaly detection | VLDB2022    |

---

## Features

IKPyKit provides a set of key features designed to make machine learning tasks easy and efficient. For a detailed overview, see the [User Guides](./user_guides/table-of-contents.md).

---

## Examples and tutorials

Explore our extensive list of examples and tutorials to get you started with IKPyKit. You can find them [here](./examples/examples_english.md).

---

## How to contribute

Primarily, IKPyKit development consists of adding and creating new algorithms, new validation strategies, or improving the performance of the current code. However, there are many other ways to contribute:

- Submit a bug report or feature request on [GitHub Issues](https://github.com/IsolationKernel/ikpykit/issues).
- Contribute a Jupyter notebook to our [examples](./examples/examples_english.md).
- Write [unit or integration tests](https://docs.pytest.org/en/latest/) for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

For more information on how to contribute to IKPyKit, see our [Contribution Guide](./contributing/contribution.md).

Visit our [authors section](./authors/authors.md) to meet all the contributors to IKPyKit.

---

## Citation

If you use IKPyKit for a scientific Publication, we would appreciate citations to the Publication software.

**BibTeX**:

```bibtex
@software{IKPyKit,
    author = {Xin Han, Yixiao Ma, Ye Zhu, and Kaiming Ting},
    title = {IKPyKit：A Python Library for Isolation Kernel Toolkit},
    version = {0.1.0},
    month = {3},
    year = {2025},
    license = {BSD-3-Clause},
    url = {https://github.com/IsolationKernel/ikpykit}
}
```

---

## License

[BSD-3-Clause License](https://github.com/IsolationKernel/ikpykit/blob/master/LICENSE)
