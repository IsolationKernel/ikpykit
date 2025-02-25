"""
pyiks (c) by Xin Han

pyiks is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from typing import Optional, Union, Any
import numpy as np

from ._inode import INODE
from pyiks.kernel import IsoKernel
from .utils.dendrogram_purity import dendrogram_purity
from .utils.Graphviz import Graphviz
from .utils.serialize_trees import serliaze_tree_to_file


class STREAMKHC(BaseEstimator, ClusterMixin):
    """Streaming Hierarchical Clustering Based on Point-Set Kernel.

    This algorithm performs hierarchical clustering on streaming data using
    isolation kernel techniques. It builds a tree structure that adapts as new
    data points arrive, allowing for efficient online clustering.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of base estimators in the isolation kernel.

    max_samples : str, int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
        - If "auto", then `max_samples=min(8, n_samples)`.

    contamination : 'auto' or float, default='auto'
        The proportion of outliers in the data set. Used when fitting to define
        the threshold on interval scores. Must be in the range (0, 0.5].

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    max_depth : int, default=5000
        Maximum number of data points to maintain in the clustering tree.
        When exceeded, the oldest points will be removed.

    Attributes
    ----------
    tree_ : INODE
        The root node of the hierarchical clustering tree.

    iso_kernel_ : IsoKernel
        The isolation kernel used for data transformation.

    point_counter_ : int
        Counter tracking the total number of points processed.

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from pyiks.stream.cluster import STREAMKHC
    >>> import numpy as np
    >>> # Generate sample data
    >>> X = np.random.rand(100, 10)  # 100 samples with 10 features
    >>> y = np.random.randint(0, 3, size=100)  # Optional class labels
    >>>
    >>> # Initialize and fit the model with a batch
    >>> clusterer = STREAMKHC(n_estimators=100, random_state=42)
    >>> clusterer.fit_batch(X, y)
    >>>
    >>> # Process new streaming data
    >>> new_data = np.random.rand(10, 10)  # 10 new samples
    >>> new_labels = np.random.randint(0, 3, size=10)  # Optional class labels
    >>> clusterer.fit_online(new_data, new_labels)
    >>>
    >>> # Calculate clustering purity (if labels were provided)
    >>> purity = clusterer.get_purity()
    >>> print(f"Clustering purity: {purity:.2f}")
    >>>
    >>> # Visualize clustering tree
    >>> clusterer.visualize_tree("clustering_tree.png")

    References
    ----------
    .. [1] Xin Han, Ye Zhu, Kai Ming Ting, De-Chuan Zhan, Gang Li (2022)
           Streaming Hierarchical Clustering Based on Point-Set Kernel.
           Proceedings of The ACM SIGKDD Conference on Knowledge Discovery and Data Mining.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Union[str, int, float] = "auto",
        contamination: Union[str, float] = "auto",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        max_depth: int = 5000,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.contamination = contamination
        self.max_depth = max_depth

    def fit_batch(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the model with a batch of data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data points.
        y : array-like of shape (n_samples,), optional (default=None)
            The labels of the data points.
            Not used in clustering processing, just for calculating purity.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self._initialize_tree(X)
        return self

    def _initialize_tree(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Initialize the hierarchical clustering tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data points.
        y : array-like of shape (n_samples,), optional (default=None)
            The labels of the data points.
            Not used in clustering processing, just for calculating purity.
        """
        self.iso_kernel_ = IsoKernel(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        X_ikv = self.iso_kernel_.fit_transform(X, dense_output=True)
        self.tree_ = INODE()
        self.point_counter_ = 0

        for x in X_ikv:
            if self.point_counter_ >= self.max_depth:
                self.tree_ = self.tree_.delete()
            self.tree_ = self.tree_.insert(x, t=self.n_estimators)
            self.point_counter_ += 1

    def fit_online(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the model with a stream of data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data points.
        y : array-like of shape (n_samples,), optional (default=None)
            The labels of the data points.
            Not used in clustering processing, just for calculating purity.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        NotFittedError
            If the model has not been initialized with fit_batch.
        ValueError
            If X has a different number of features than seen during fit_batch.
        """
        X = check_array(X)
        check_is_fitted(self, ["tree_", "iso_kernel_"])

        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but STREAMKHC was trained with {self.n_features_in_} features."
            )

        X_ikv = self.iso_kernel_.transform(X, dense_output=True)
        for x in X_ikv:
            if self.point_counter_ >= self.max_depth:
                self.tree_ = self.tree_.delete()
            self.tree_ = self.tree_.insert(x, t=self.n_estimators)
            self.point_counter_ += 1

        return self

    def get_purity(self) -> float:
        """Calculate the purity of the clustering tree.

        Returns
        -------
        float
            The purity score of the clustering tree.

        Raises
        ------
        NotFittedError
            If the model has not been initialized with fit_batch.
        """
        check_is_fitted(self, "tree_")
        return dendrogram_purity(self.tree_)

    def serialize_tree(self, path: str) -> None:
        """Serialize the clustering tree to a file.

        Parameters
        ----------
        path : str
            The file path to save the serialized tree.

        Raises
        ------
        NotFittedError
            If the model has not been initialized with fit_batch.
        """
        check_is_fitted(self, "tree_")
        serliaze_tree_to_file(self.tree_, path)

    def visualize_tree(self, path: str) -> None:
        """Visualize the clustering tree using Graphviz.

        Parameters
        ----------
        path : str
            The file path to save the visualization.

        Raises
        ------
        NotFittedError
            If the model has not been initialized with fit_batch.
        """
        check_is_fitted(self, "tree_")
        Graphviz(self.tree_, path)
