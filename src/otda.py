"""
Optimal Transport Domain Adaptation core functions.

Adapted from Peterson et al. (2022):
https://github.com/vpeterson/otda-mibci/blob/main/paper_example_samplewise.ipynb

Reference:
Peterson, V., Nieto, N., Wyser, D., Lambercy, O., Gassert, R.,
Milone, D. H., & Spies, R. D. (2022). Transfer Learning based on
Optimal Transport for Motor Imagery Brain-Computer Interfaces.
IEEE Transactions on Biomedical Engineering.

Original authors: V. Peterson & N. Nieto
"""

import numpy as np
import ot
from sklearn.metrics import accuracy_score


def forward_ot_sinkhorn(X_source, y_source, X_target, reg_e, clf, metric='sqeuclidean', norm=None):
    """
    Forward Optimal Transport with Sinkhorn regularization (FOTDA-S).

    Transports source data to target domain, then trains classifier on transported source.

    Parameters
    ----------
    X_source : ndarray, shape (n_source, n_features)
        Source data (e.g., CSP features from calibration session).
    y_source : ndarray, shape (n_source,)
        Source labels.
    X_target : ndarray, shape (n_target, n_features)
        Target data (e.g., CSP features from test session).
    reg_e : float
        Entropic regularization parameter.
    clf : sklearn classifier
        Classifier to train on transported source (e.g., LDA).
    metric : str
        Distance metric for OT cost matrix.
    norm : str or None
        Normalization for cost matrix.

    Returns
    -------
    otda : ot.da.SinkhornTransport
        Fitted OT model.
    clf : sklearn classifier
        Trained classifier on transported source.
    """
    otda = ot.da.SinkhornTransport(metric=metric, reg_e=reg_e, norm=norm, verbose=False)

    # Learn mapping: source → target
    otda.fit(Xs=X_source, Xt=X_target)

    # Transport source samples to target domain
    X_source_transported = otda.transform(Xs=X_source)

    # Train classifier on transported source
    clf.fit(X_source_transported, y_source)

    return otda, clf


def forward_ot_grouplasso(X_source, y_source, X_target, reg_e, reg_cl, clf,
                          metric='sqeuclidean', norm=None):
    """
    Forward Optimal Transport with Group-Lasso regularization (FOTDA-GL).

    Uses label information to guide the transport.

    Parameters
    ----------
    X_source : ndarray, shape (n_source, n_features)
        Source data.
    y_source : ndarray, shape (n_source,)
        Source labels.
    X_target : ndarray, shape (n_target, n_features)
        Target data.
    reg_e : float
        Entropic regularization parameter.
    reg_cl : float
        Group-Lasso regularization parameter.
    clf : sklearn classifier
        Classifier to train on transported source.
    metric : str
        Distance metric.
    norm : str or None
        Cost matrix normalization.

    Returns
    -------
    otda : ot.da.SinkhornL1l2Transport
        Fitted OT-GL model.
    clf : sklearn classifier
        Trained classifier.
    """
    otda = ot.da.SinkhornL1l2Transport(
        metric=metric, reg_e=reg_e, reg_cl=reg_cl, norm=norm, verbose=False
    )

    # Learn mapping with label information
    otda.fit(Xs=X_source, ys=y_source, Xt=X_target)

    # Transport source samples
    X_source_transported = otda.transform(Xs=X_source)

    # Train classifier
    clf.fit(X_source_transported, y_source)

    return otda, clf


def backward_ot_sinkhorn(X_source, X_target, y_target, reg_e, clf,
                         metric='sqeuclidean', norm=None):
    """
    Backward Optimal Transport with Sinkhorn regularization (BOTDA-S).

    Transports target data to source domain, uses pre-trained classifier.
    No retraining required.

    Parameters
    ----------
    X_source : ndarray, shape (n_source, n_features)
        Source data (calibration).
    X_target : ndarray, shape (n_target, n_features)
        Target data (test).
    y_target : ndarray, shape (n_target,)
        Target labels (for learning transport, not for training classifier).
    reg_e : float
        Entropic regularization parameter.
    clf : sklearn classifier
        Classifier already trained on source data.
    metric : str
        Distance metric.
    norm : str or None
        Cost matrix normalization.

    Returns
    -------
    botda : ot.da.SinkhornTransport
        Fitted backward OT model.
    X_target_transported : ndarray
        Target samples transported to source domain.
    """
    botda = ot.da.SinkhornTransport(metric=metric, reg_e=reg_e, norm=norm, verbose=False)

    # Backward: Xs=target, Xt=source
    botda.fit(Xs=X_target, Xt=X_source)

    # Transport target samples to source domain
    X_target_transported = botda.transform(Xs=X_target)

    return botda, X_target_transported


def backward_ot_grouplasso(X_source, X_target, y_target, reg_e, reg_cl, clf,
                           metric='sqeuclidean', norm=None):
    """
    Backward Optimal Transport with Group-Lasso regularization (BOTDA-GL).

    Uses target labels to guide backward transport. No classifier retraining.

    Parameters
    ----------
    X_source : ndarray, shape (n_source, n_features)
        Source data.
    X_target : ndarray, shape (n_target, n_features)
        Target data.
    y_target : ndarray, shape (n_target,)
        Target labels (for transport learning).
    reg_e : float
        Entropic regularization parameter.
    reg_cl : float
        Group-Lasso regularization parameter.
    clf : sklearn classifier
        Classifier already trained on source.
    metric : str
        Distance metric.
    norm : str or None
        Cost matrix normalization.

    Returns
    -------
    botda : ot.da.SinkhornL1l2Transport
        Fitted backward OT-GL model.
    X_target_transported : ndarray
        Target samples transported to source domain.
    """
    botda = ot.da.SinkhornL1l2Transport(
        metric=metric, reg_e=reg_e, reg_cl=reg_cl, norm=norm, verbose=False
    )

    # Backward: Xs=target (with labels), Xt=source
    botda.fit(Xs=X_target, ys=y_target, Xt=X_source)

    # Transport target samples to source domain
    X_target_transported = botda.transform(Xs=X_target)

    return botda, X_target_transported


def predict_with_botda(botda, X_target, clf):
    """
    Helper function to predict using backward OT.

    Parameters
    ----------
    botda : ot.da.SinkhornTransport or ot.da.SinkhornL1l2Transport
        Fitted backward OT model.
    X_target : ndarray
        Target samples to classify.
    clf : sklearn classifier
        Classifier trained on source.

    Returns
    -------
    y_pred : ndarray
        Predicted labels.
    """
    X_target_transported = botda.transform(Xs=X_target)
    y_pred = clf.predict(X_target_transported)
    return y_pred



