<!-- <script src="https://kit.fontawesome.com/d20edc211b.js" crossorigin="anonymous"></script>

<div style="margin-bottom: 20px;">
    <img src="img/banner-landing-page-PyIKS.png#only-light" align="left" style="margin-bottom: 30px; margin-top: 0px;">
    <img src="img/banner-landing-page-dark-mode-PyIKS-no-background.png#only-dark" align="left" style="margin-bottom: 30px; margin-top: 0px;">
</div>

<div style="clear: both;"></div> -->

## About The Project

**PyIKS** (Python for Isolation Kernel Similarity) is an intuitive Python library designed for a variety of machine learning tasks including kernel similarity calculation, anomaly detection, clustering, and change detection—all powered by the innovative **Isolation Kernel (IK)** . Isolation Kernel is a data-dependent kernel that measures similarity by isolating data points using an isolation mechanism. It uniquely adapts to the data distribution, with the property that points in sparse regions are more similar than those in dense regions. Notably, it requires no learning or closed-form expression, making it efficient and scalable.

---

### Why use Isolation Kernel?

- **Data-Dependent Similarity**: Unlike traditional kernels (e.g., Gaussian, Laplacian), Isolation Kernel adapts to the structure of the data rather than assuming a fixed similarity function.
- **Handles Sparse and Dense Regions**: Isolation Kernel effectively accounts for varying data densities, making it ideal for datasets with non-uniform distributions.
- **No Learning Required**: It eliminates the need for training or parameter tuning, simplifying implementation while reducing computational cost.
- **Effective in High Dimensions**: It uniquely addresses the curse of dimensionality, being the only known measure capable of finding exact nearest neighbors in high-dimensional spaces.
- **Versatile Applications**: Isolation Kernel has been successfully applied to tasks like anomaly detection, clustering, and processing stream data, graph data, trajectory data, and more.

Learn more about its history and development on the [IsolationKernel GitHub page](https://github.com/IsolationKernel).

---

### Why use PyIKS?

PyIKS is specifically built to harness the power of Isolation Kernel, providing specialized algorithms for a wide range of data types and tasks. Its seamless integration with the scikit-learn API allows easy adoption and compatibility with scikit-learn tools.

- **Tailored for Isolation Kernel**: PyIKS directly leverages the unique properties of Isolation Kernel for efficient and effective machine learning solutions.
- **Efficient and User-Friendly**: Designed for simplicity and performance, PyIKS offers an intuitive interface built on the scikit-learn API.
- **Support for Diverse Data Types**: It supports graph data, group data, stream data, time series, and trajectory data, making it versatile for various domains.
- **Comprehensive Resources**: Users benefit from rich documentation and examples to quickly understand and apply the library’s features.
- **Ideal for Research and Industry**: PyIKS is suitable for both academic research and industrial applications, providing scalable and cutting-edge tools for modern machine learning challenges.

---

## Installation & Dependencies

To install the basic version of `PyIKS` with core dependencies, run the following:

```bash
pip install pyiks
```

For more installation options, including dependencies and additional features, check out our [Installation Guide](./quick-start/how-to-install.html).

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

| Abbr                                            | Algorithm                     | Utilization                                   | Published            |
| ----------------------------------------------- | ----------------------------- | --------------------------------------------- | -------------------- |
| [IsoKernel](./api/isolation_kernel.html)        | Isolation Kernel              | IK feature mapping and similarity calculating | AAAI2019, SIGKDD2018 |
| [IsodisKernel](./api/isolation_dis_kernel.html) | Isolation Distribution Kernel | Distribution similarity calculating           | SIGKDD2022           |

**(ii) Point Anomaly detection** :

| Abbr                          | Algorithm                                                          | Utiliztion        | Published          |
| ----------------------------- | ------------------------------------------------------------------ | ----------------- | ------------------ |
| [IForest](./api/iforest.html) | Isolation forest                                                   | Anomaly Detection | ICDM2008, TKDD2022 |
| [INNE](./api/inne.html)       | Isolation-based anomaly detection using nearest-neighbor ensembles | Anomaly Detection | CIJ2018            |
| [IDKD](./api/idkd.html)       | Isolation Distributional Kernel for point anomaly detections       | Anomaly Detection | TKDE2022           |

**(iii) Point Clustering** :

| Abbr                    | Algorithm                                                    | Utiliztion              | Published |
| ----------------------- | ------------------------------------------------------------ | ----------------------- | --------- |
| [IDKC](./api/idkc.html) | Kernel-based Clustering via Isolation Distributional Kernel. | Point Clustering        | IS2023    |
| [PSKC](./api/pskc.html) | Point-set Kernel Clustering                                  | Point Clustering        | TKDE2023  |
| IKAHC                   | Isolation Kernel for Agglomerative Hierarchical Clustering   | Hierarchical Clustering | PR2023    |

**(IV) Graph Data** :

| Abbr                                        | Algorithm                                                              | Utiliztion                                    | Published |
| ------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------- | --------- |
| [IKGOD](./api/ikgod.html)                   | Subgraph Centralization: A Necessary Step for Graph Anomaly Detection. | Graph Anomaly Detection                       | SIAM2023  |
| [IsoGraphKernel](./api/IsoGraphKernel.html) | Isolation Graph Kernel                                                 | Graph IK embedding and similarity calculating | AAAI2021  |

**(V) Group Data** :

| Abbr                      | Algorithm                                                    | Utiliztion              | Published |
| ------------------------- | ------------------------------------------------------------ | ----------------------- | --------- |
| [IKGAD](./api/ikgad.html) | Isolation Distributional Kernel for group anomaly detections | Group Anomaly Detection | TKDE2022  |

**(VI) Stream Data** :

| Abbr                            | Algorithm                                                       | Utiliztion                     | Published  |
| ------------------------------- | --------------------------------------------------------------- | ------------------------------ | ---------- |
| [StreaKHC](./api/streakhc.html) | Isolation Distribution Kernel for Trajectory Anomaly Detections | Online Hierarchical Clustering | SIGKDD2022 |
| [ICID](./api/icid.html)         | Detecting change intervals with isolation distributional kernel | Change Intervals Detection     | JAIR2024   |

**(VII) Trajectory Data** :

| Abbr                                 | Algorithm                                                       | Utiliztion                   | Published |
| ------------------------------------ | --------------------------------------------------------------- | ---------------------------- | --------- |
| [TIDKC](./api/trajectory/tidkc.html) | Distribution-based Tajectory Clustering                         | Trajectory Clustering        | ICDM2023  |
| [IKAT](./api/trajectory/ikat.html)   | Isolation Distribution Kernel for Trajectory Anomaly Detections | Trajectory Anomaly Detection | JAIR2024  |

**(VIII) Time Series**

| Abbr  | Algorithm                                                       | Utiliztion        | Published |
| ----- | --------------------------------------------------------------- | ----------------- | --------- |
| IKTOD | Isolation distribution kernel for Time Series Anomaly Detection | Anomaly detection | VLDB2022  |

---

## Features

PyIKS provides a set of key features designed to make time series forecasting with machine learning easy and efficient. For a detailed overview, see the [User Guides](./user_guides/table-of-contents.html).

---

## Examples and tutorials

Explore our extensive list of examples and tutorials (English and Spanish) to get you started with PyIKS. You can find them [here](./examples/examples_english.html).

---

## How to contribute

Primarily, PyIKS development consists of adding and creating new *Forecasters*, new validation strategies, or improving the performance of the current code. However, there are many other ways to contribute:

- Submit a bug report or feature request on [GitHub Issues](https://github.com/PyIKS/PyIKS/issues).
- Contribute a Jupyter notebook to our [examples](./examples/examples_english.html).
- Write [unit or integration tests](https://docs.pytest.org/en/latest/) for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

For more information on how to contribute to PyIKS, see our [Contribution Guide](https://github.com/PyIKS/PyIKS/blob/master/CONTRIBUTING.md).

Visit our [authors section](./authors/authors.html) to meet all the contributors to PyIKS.

---

## Citation

If you use PyIKS for a scientific publication, we would appreciate citations to the published software.

**BibTeX**:

```
@software{PyIKS,
author = {Xin Han, Yixiao Ma, Ye Zhu, and Kaiming Ting},
title = {PyIKS},
version = {0.2.0},
month = {11},
year = {2024},
license = {BSD-3-Clause},
url = {https://PyIKS.org/},
doi = {10.5281/zenodo.8382788}
}
```

---

## License

[BSD-3-Clause License](https://github.com/PyIKS/PyIKS/blob/master/LICENSE)
