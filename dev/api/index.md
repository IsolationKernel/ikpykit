# API Reference

Every estimator IKPyKit exports, grouped by the kind of data it works on. They all follow the scikit-learn API, so each one is constructed with its parameters and then fitted.

## Isolation Kernel

| Estimator                                                                                  | Description                                                                                       | Publication          |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | -------------------- |
| [IsoDisKernel](https://isolationkernel.github.io/ikpykit/dev/api/kernel/isodiskernel.html) | Isolation Distributional Kernel is a new way to measure the similarity between two distributions. | SIGKDD2020           |
| [IsoKernel](https://isolationkernel.github.io/ikpykit/dev/api/kernel/isokernel.html)       | Isolation Kernel.                                                                                 | AAAI2019, SIGKDD2018 |

## Point Anomaly Detection

| Estimator                                                                         | Description                                                         | Publication        |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------ |
| [IDKD](https://isolationkernel.github.io/ikpykit/dev/api/anomaly/idkd.html)       | Isolation Distributional Kernel for anomaly detection.              | TKDE2022           |
| [INNE](https://isolationkernel.github.io/ikpykit/dev/api/anomaly/inne.html)       | Isolation-based anomaly detection using nearest-neighbor ensembles. | CIJ2018            |
| [IForest](https://isolationkernel.github.io/ikpykit/dev/api/anomaly/iforest.html) | Wrapper of scikit-learn Isolation Forest for anomaly detection.     | ICDM2008, TKDD2012 |

## Point Clustering

| Estimator                                                                     | Description                                                                                                                                             | Publication |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| [IDKC](https://isolationkernel.github.io/ikpykit/dev/api/cluster/idkc.html)   | Isolation Distributional Kernel Clustering.                                                                                                             | IS2023      |
| [IKAHC](https://isolationkernel.github.io/ikpykit/dev/api/cluster/ikahc.html) | IKAHC is a novel hierarchical clustering algorithm. It uses a data-dependent kernel called Isolation Kernel to measure the similarity between clusters. | PR2023      |
| [PSKC](https://isolationkernel.github.io/ikpykit/dev/api/cluster/pskc.html)   | Point-Set Kernel Clustering algorithm using Isolation Kernels.                                                                                          | TKDE2023    |

## Graph Mining

| Estimator                                                                                     | Description                                                                       | Publication |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ----------- |
| [IKGOD](https://isolationkernel.github.io/ikpykit/dev/api/graph/ikgod.html)                   | Isolation-based Graph Anomaly Detection using kernel embeddings.                  | SIAM2023    |
| [IsoGraphKernel](https://isolationkernel.github.io/ikpykit/dev/api/graph/isographkernel.html) | Isolation Graph Kernel is a new way to measure the similarity between two graphs. | AAAI2021    |

## Group Mining

| Estimator                                                                   | Description                                     | Publication |
| --------------------------------------------------------------------------- | ----------------------------------------------- | ----------- |
| [IKGAD](https://isolationkernel.github.io/ikpykit/dev/api/group/ikgad.html) | Isolation Kernel-based Group Anomaly Detection. | TKDE2022    |

## Stream Mining

| Estimator                                                                            | Description                                                                        | Publication |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ----------- |
| [ICID](https://isolationkernel.github.io/ikpykit/dev/api/stream/icid.html)           | Isolate Change Interval Detection for monitoring data stream distribution changes. | JAIR2024    |
| [STREAMKHC](https://isolationkernel.github.io/ikpykit/dev/api/stream/streamkhc.html) | Streaming Hierarchical Clustering Based on Point-Set Kernel.                       | SIGKDD2022  |

## Trajectory Mining

| Estimator                                                                        | Description                                                    | Publication |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------- | ----------- |
| [IKAT](https://isolationkernel.github.io/ikpykit/dev/api/trajectory/ikat.html)   | Isolation-based anomaly detection for trajectory data.         | JAIR2024    |
| [TIDKC](https://isolationkernel.github.io/ikpykit/dev/api/trajectory/tidkc.html) | Trajectory Isolation Distributional Kernel Clustering (TIDKC). | ICDM2023    |

### DataLoader

| Estimator                                                                                           | Description                   |
| --------------------------------------------------------------------------------------------------- | ----------------------------- |
| [SheepDogs](https://isolationkernel.github.io/ikpykit/dev/api/trajectory/dataloader/sheepdogs.html) | SheepDogs trajectory dataset. |

## Time Series Mining

| Estimator                                                                        | Description                                                       | Publication |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------- |
| [IKTOD](https://isolationkernel.github.io/ikpykit/dev/api/timeseries/iktod.html) | Isolation Kernel-based Time series Subsequence Anomaly Detection. | VLDB2022    |
