"""
CSP + LDA pipeline for motor imagery classification.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP


class CSP_LDA(BaseEstimator, ClassifierMixin):
    """
    Combined CSP + LDA classifier for EEG motor imagery.

    Parameters
    ----------
    n_components : int
        Number of CSP components to extract (default: 4).
    reg : float or None
        LDA regularization parameter (default: None for no regularization).
    """

    def __init__(self, n_components=4, reg=None):
        self.n_components = n_components
        self.reg = reg
        self.csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=reg)

    def fit(self, X, y):
        """
        Fit CSP + LDA pipeline.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG epoch data.
        y : ndarray, shape (n_trials,)
            Class labels.

        Returns
        -------
        self
        """
        # Extract CSP features
        self.csp.fit(X, y)
        X_csp = self.csp.transform(X)

        # Train LDA on CSP features
        self.lda.fit(X_csp, y)

        return self

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG epoch data.

        Returns
        -------
        y_pred : ndarray, shape (n_trials,)
            Predicted class labels.
        """
        # Transform using CSP
        X_csp = self.csp.transform(X)

        # Predict using LDA
        return self.lda.predict(X_csp)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG epoch data.

        Returns
        -------
        proba : ndarray, shape (n_trials, n_classes)
            Class probabilities.
        """
        X_csp = self.csp.transform(X)
        return self.lda.predict_proba(X_csp)

    def score(self, X, y):
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG epoch data.
        y : ndarray, shape (n_trials,)
            True class labels.

        Returns
        -------
        accuracy : float
            Classification accuracy.
        """
        X_csp = self.csp.transform(X)
        return self.lda.score(X_csp, y)

    def transform(self, X):
        """
        Transform data using CSP (for OTDA).

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG epoch data.

        Returns
        -------
        X_csp : ndarray, shape (n_trials, n_components)
            CSP features.
        """
        return self.csp.transform(X)

    @property
    def coef_(self):
        """LDA coefficients for distance_to_hyperplane."""
        return self.lda.coef_

    @property
    def intercept_(self):
        """LDA intercept for distance_to_hyperplane."""
        return self.lda.intercept_
