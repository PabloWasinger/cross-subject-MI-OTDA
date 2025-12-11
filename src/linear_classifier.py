"""
CSP + LDA pipeline for motor imagery classification.

This classifier intelligently handles both:
- Raw EEG data: shape (n_trials, n_channels, n_times) -> applies CSP then LDA
- CSP features: shape (n_trials, n_components) -> applies LDA directly
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP


class CSP_LDA(BaseEstimator, ClassifierMixin):
    """
    Combined CSP + LDA classifier for EEG motor imagery.

    This classifier can handle both:
    - Raw EEG data: shape (n_trials, n_channels, n_times) -> applies CSP then LDA
    - Pre-computed CSP features: shape (n_trials, n_components) -> applies LDA directly

    Parameters
    ----------
    n_components : int
        Number of CSP components to extract (default: 4).
    reg : float, str, or None
        LDA regularization parameter. Use 'auto' for automatic shrinkage.
    """

    def __init__(self, n_components=4, reg=None):
        self.n_components = n_components
        self.reg = reg
        self.csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=reg)
        self._n_eeg_channels = None  
        
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
        self._n_eeg_channels = X.shape[1]
        
        self.csp.fit(X, y)
        X_csp = self.csp.transform(X)

        self.lda.fit(X_csp, y)

        return self

    def _is_csp_features(self, X):
        """
        Check if input is already CSP features (2D) or raw EEG (3D).
        
        Parameters
        ----------
        X : ndarray
            Input data.
            
        Returns
        -------
        bool
            True if X appears to be CSP features, False if raw EEG.
        """
        if X.ndim == 3:
            return False
        
        if X.ndim == 2:
            return True
        
        raise ValueError(
            f"Cannot determine if input (shape {X.shape}) is raw EEG or CSP features. "
            f"Expected 3D for raw EEG or 2D for CSP features."
        )

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : ndarray
            Either raw EEG data (n_trials, n_channels, n_times) or
            CSP features (n_trials, n_components).

        Returns
        -------
        y_pred : ndarray, shape (n_trials,)
            Predicted class labels.
        """
        if self._is_csp_features(X):
            return self.lda.predict(X)
        else:
            X_csp = self.csp.transform(X)
            return self.lda.predict(X_csp)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : ndarray
            Either raw EEG data (n_trials, n_channels, n_times) or
            CSP features (n_trials, n_components).

        Returns
        -------
        proba : ndarray, shape (n_trials, n_classes)
            Class probabilities.
        """
        if self._is_csp_features(X):
            return self.lda.predict_proba(X)
        else:
            X_csp = self.csp.transform(X)
            return self.lda.predict_proba(X_csp)

    def score(self, X, y):
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : ndarray
            Either raw EEG data (n_trials, n_channels, n_times) or
            CSP features (n_trials, n_components).
        y : ndarray, shape (n_trials,)
            True class labels.

        Returns
        -------
        accuracy : float
            Classification accuracy.
        """
        if self._is_csp_features(X):
            return self.lda.score(X, y)
        else:
            X_csp = self.csp.transform(X)
            return self.lda.score(X_csp, y)

    def transform(self, X):
        """
        Transform data using CSP.

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