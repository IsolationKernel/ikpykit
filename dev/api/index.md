# API Reference

Every estimator IKPyKit exports, grouped by the kind of data it works on. They all follow the scikit-learn API, so each one is constructed with its parameters and then fitted.

## Isolation Kernel

| Estimator                                                                                  | Description                                                                                       |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| [IsoDisKernel](https://isolationkernel.github.io/ikpykit/dev/api/kernel/isodiskernel.html) | Isolation Distributional Kernel is a new way to measure the similarity between two distributions. |
| [IsoKernel](https://isolationkernel.github.io/ikpykit/dev/api/kernel/isokernel.html)       | Isolation Kernel.                                                                                 |

## Point Anomaly Detection

| Estimator                                                                         | Description                                                         |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [IDKD](https://isolationkernel.github.io/ikpykit/dev/api/anomaly/idkd.html)       | Isolation Distributional Kernel for anomaly detection.              |
| [INNE](https://isolationkernel.github.io/ikpykit/dev/api/anomaly/inne.html)       | Isolation-based anomaly detection using nearest-neighbor ensembles. |
| [IForest](https://isolationkernel.github.io/ikpykit/dev/api/anomaly/iforest.html) | Wrapper of scikit-learn Isolation Forest for anomaly detection.     |

## Point Clustering

| Estimator                                                                     | Description                                                                                                                                             |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [IDKC](https://isolationkernel.github.io/ikpykit/dev/api/cluster/idkc.html)   | Isolation Distributional Kernel Clustering.                                                                                                             |
| [IKAHC](https://isolationkernel.github.io/ikpykit/dev/api/cluster/ikahc.html) | IKAHC is a novel hierarchical clustering algorithm. It uses a data-dependent kernel called Isolation Kernel to measure the similarity between clusters. |
| [PSKC](https://isolationkernel.github.io/ikpykit/dev/api/cluster/pskc.html)   | Point-Set Kernel Clustering algorithm using Isolation Kernels.                                                                                          |

## Graph Mining

| Estimator                                                                                     | Description                                                                       |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [IKGOD](https://isolationkernel.github.io/ikpykit/dev/api/graph/ikgod.html)                   | Isolation-based Graph Anomaly Detection using kernel embeddings.                  |
| [IsoGraphKernel](https://isolationkernel.github.io/ikpykit/dev/api/graph/isographkernel.html) | Isolation Graph Kernel is a new way to measure the similarity between two graphs. |

## Group Mining

| Estimator                                                                   | Description                                     |
| --------------------------------------------------------------------------- | ----------------------------------------------- |
| [IKGAD](https://isolationkernel.github.io/ikpykit/dev/api/group/ikgad.html) | Isolation Kernel-based Group Anomaly Detection. |

## Stream Mining

| Estimator                                                                            | Description                                                                        |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| [ICID](https://isolationkernel.github.io/ikpykit/dev/api/stream/icid.html)           | Isolate Change Interval Detection for monitoring data stream distribution changes. |
| [STREAMKHC](https://isolationkernel.github.io/ikpykit/dev/api/stream/streamkhc.html) | Streaming Hierarchical Clustering Based on Point-Set Kernel.                       |

## Trajectory Mining

| Estimator                                                                        | Description                                                    |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| [IKAT](https://isolationkernel.github.io/ikpykit/dev/api/trajectory/ikat.html)   | Isolation-based anomaly detection for trajectory data.         |
| [TIDKC](https://isolationkernel.github.io/ikpykit/dev/api/trajectory/tidkc.html) | Trajectory Isolation Distributional Kernel Clustering (TIDKC). |

### DataLoader

| Estimator                                                                                           | Description                   |
| --------------------------------------------------------------------------------------------------- | ----------------------------- |
| [SheepDogs](https://isolationkernel.github.io/ikpykit/dev/api/trajectory/dataloader/sheepdogs.html) | SheepDogs trajectory dataset. |

## Time Series Mining

| Estimator                                                                        | Description                                                       |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [IKTOD](https://isolationkernel.github.io/ikpykit/dev/api/timeseries/iktod.html) | Isolation Kernel-based Time series Subsequence Anomaly Detection. |
