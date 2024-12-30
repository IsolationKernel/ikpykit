"""
pyike (c) by Xin Han

pyike is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from pyike.group.anomaly._ikgad import IKGAD


def test_IKGAD_fit():
    # Create a sample dataset
    X = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    clf = IKGAD()
    clf.fit(X)
    assert clf.is_fitted_


def test_IKGAD_predict():
    X = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    clf = IKGAD()
    clf.fit(X)

    predictions = clf.predict(X)

    assert predictions.shape == (2, 3)


def test_IKGAD_decision_function():
    X = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    clf = IKGAD()
    clf.fit(X)
    decision_func = clf.decision_function(X)
    # Check if the decision function has the correct shape
    assert decision_func.shape == (2, 3)


def test_IKGAD_score_samples():
    X = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    clf = IKGAD()
    clf.fit(X)
    scores = clf.score_samples(X)
    # Check if the scores have the correct shape
    assert scores.shape == (2,)
