"""
Hyperparameter tuning and subset selection for OTDA.

Adapted from Peterson et al. (2022):
https://github.com/vpeterson/otda-mibci/blob/main/MIOTDAfunctions.py

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
from sklearn.model_selection import train_test_split


def distance_to_hyperplane(X, clf):
    """
    Calculates distance from samples to decision hyperplane.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data matrix.
    clf : sklearn classifier
        Trained classifier with linear decision boundary.

    Returns
    -------
    ndarray, shape (n_samples,)
        Distance of each sample to hyperplane.
    """
    b = clf.intercept_
    W = clf.coef_
    mod = np.sqrt(np.sum(np.power(W, 2)))
    d = np.abs(np.dot(X, W.T) + b) / mod
    return d[:, 0]


def wrong_classified(clf, X, y):
    """
    Returns indices of misclassified samples.

    Parameters
    ----------
    clf : sklearn classifier
        Trained classifier.
    X : ndarray, shape (n_samples, n_features)
        Data matrix.
    y : ndarray, shape (n_samples,)
        True labels.

    Returns
    -------
    ndarray
        Indices of misclassified samples.
    """
    y_pred = clf.predict(X)
    idx_wrong = np.where(y_pred != y)[0]
    return idx_wrong


def select_subset_by_distance(X_source, y_source, clf, M=20):
    """
    Selects M samples per class based on distance to decision boundary.

    Parameters
    ----------
    X_source : ndarray, shape (n_samples, n_features)
        Source data matrix.
    y_source : ndarray, shape (n_samples,)
        Source labels.
    clf : sklearn classifier
        Trained classifier on source data.
    M : int
        Number of samples to select per class.

    Returns
    -------
    X_subset : ndarray
        Selected subset.
    y_subset : ndarray
        Corresponding labels.
    """
    d = distance_to_hyperplane(X_source, clf)
    idx_wrong = wrong_classified(clf, X_source, y_source)
    d[idx_wrong] = -np.inf

    idx_sorted = np.argsort(d)[::-1]
    X_sorted = X_source[idx_sorted, :]
    y_sorted = y_source[idx_sorted]

    classes = np.unique(y_sorted)
    X_subset = []
    y_subset = []

    for c in classes:
        idx_class = np.where(y_sorted == c)[0]
        n_select = min(M, len(idx_class))
        X_subset.append(X_sorted[idx_class[:n_select], :])
        y_subset.append(y_sorted[idx_class[:n_select]])

    X_subset = np.vstack(X_subset)
    y_subset = np.hstack(y_subset)

    return X_subset, y_subset


def cv_sinkhorn(reg_e_grid, X_source, y_source, X_target, y_target, clf,
                metric='sqeuclidean', kfold=None, norm=None, verbose=False):
    """
    Grid search for Sinkhorn regularization (Forward OT).

    Parameters
    ----------
    reg_e_grid : list
        Grid of entropic regularization values.
    X_source : ndarray
        Source data.
    y_source : ndarray
        Source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier to train on transported source.
    metric : str
        Distance metric.
    kfold : dict or None
        CV config: {'nfold': int, 'train_size': float}.
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    float
        Best regularization parameter.
    """
    results = []

    for reg_e in reg_e_grid:
        ot_sinkhorn = ot.da.SinkhornTransport(metric=metric, reg_e=reg_e, norm=norm, verbose=False)

        if kfold is None:
            ot_sinkhorn.fit(Xs=X_source, Xt=X_target)
            X_source_transported = ot_sinkhorn.transform(Xs=X_source)
            clf.fit(X_source_transported, y_source)
            y_pred = clf.predict(X_target)
            acc = accuracy_score(y_target, y_pred)
            results.append(acc)
        else:
            acc_cv = []
            for k in range(kfold['nfold']):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_target, y_target, train_size=kfold['train_size'],
                    stratify=y_target, random_state=100*k
                )
                ot_sinkhorn.fit(Xs=X_source, Xt=X_train)
                X_source_transported = ot_sinkhorn.transform(Xs=X_source)
                clf.fit(X_source_transported, y_source)
                y_pred = clf.predict(X_test)
                acc_cv.append(accuracy_score(y_test, y_pred))
            results.append(np.mean(acc_cv))

    best_idx = np.argmax(results)
    best_reg = reg_e_grid[best_idx]

    if verbose:
        print(f'Best reg_e: {best_reg}')
        print(f'Accuracy grid: {results}')

    return best_reg


def cv_grouplasso(reg_e_grid, reg_cl_grid, X_source, y_source, X_target, y_target,
                  clf, metric='sqeuclidean', kfold=None, norm=None, verbose=False):
    """
    Grid search for Group-Lasso regularization (Forward OT).

    Parameters
    ----------
    reg_e_grid : list
        Entropic regularization grid.
    reg_cl_grid : list
        Group-lasso regularization grid.
    X_source : ndarray
        Source data.
    y_source : ndarray
        Source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier to train on transported source.
    metric : str
        Distance metric.
    kfold : dict or None
        CV config.
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    list [reg_e, reg_cl]
        Best regularization parameters.
    """
    results = np.empty((len(reg_e_grid), len(reg_cl_grid)))

    for i, reg_e in enumerate(reg_e_grid):
        for j, reg_cl in enumerate(reg_cl_grid):
            ot_l1l2 = ot.da.SinkhornL1l2Transport(
                metric=metric, reg_e=reg_e, reg_cl=reg_cl, norm=norm, verbose=False
            )

            if kfold is None:
                ot_l1l2.fit(Xs=X_source, ys=y_source, Xt=X_target)
                X_source_transported = ot_l1l2.transform(Xs=X_source)
                clf.fit(X_source_transported, y_source)
                y_pred = clf.predict(X_target)
                acc = accuracy_score(y_target, y_pred)
                results[i, j] = acc
            else:
                acc_cv = []
                for k in range(kfold['nfold']):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_target, y_target, train_size=kfold['train_size'],
                        stratify=y_target, random_state=100*k
                    )
                    ot_l1l2.fit(Xs=X_source, ys=y_source, Xt=X_train)
                    X_source_transported = ot_l1l2.transform(Xs=X_source)
                    clf.fit(X_source_transported, y_source)
                    y_pred = clf.predict(X_test)
                    acc_cv.append(accuracy_score(y_test, y_pred))
                results[i, j] = np.mean(acc_cv)

    best_idx = np.unravel_index(results.argmax(), results.shape)
    best_reg = [reg_e_grid[best_idx[0]], reg_cl_grid[best_idx[1]]]

    if verbose:
        print(f'Best params: reg_e={best_reg[0]}, reg_cl={best_reg[1]}')
        print(f'Accuracy matrix:\n{results}')

    return best_reg


def cv_sinkhorn_backward(reg_e_grid, X_source, y_source, X_target, y_target, clf,
                         metric='sqeuclidean', kfold=None, norm=None, verbose=False):
    """
    Grid search for Sinkhorn regularization (Backward OT).

    Parameters
    ----------
    reg_e_grid : list
        Entropic regularization grid.
    X_source : ndarray
        Source data.
    y_source : ndarray
        Source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier already trained on source.
    metric : str
        Distance metric.
    kfold : dict or None
        CV config.
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    float
        Best regularization parameter.
    """
    results = []

    for reg_e in reg_e_grid:
        bot = ot.da.SinkhornTransport(metric=metric, reg_e=reg_e, norm=norm, verbose=False)

        if kfold is None:
            bot.fit(Xs=X_target, Xt=X_source)
            X_target_transported = bot.transform(Xs=X_target)
            y_pred = clf.predict(X_target_transported)
            acc = accuracy_score(y_target, y_pred)
            results.append(acc)
        else:
            acc_cv = []
            for k in range(kfold['nfold']):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_target, y_target, train_size=kfold['train_size'],
                    stratify=y_target, random_state=100*k
                )
                bot.fit(Xs=X_train, Xt=X_source)
                X_test_transported = bot.transform(Xs=X_test)
                y_pred = clf.predict(X_test_transported)
                acc_cv.append(accuracy_score(y_test, y_pred))
            results.append(np.mean(acc_cv))

    best_idx = np.argmax(results)
    best_reg = reg_e_grid[best_idx]

    if verbose:
        print(f'Best reg_e: {best_reg}')
        print(f'Accuracy grid: {results}')

    return best_reg


def cv_grouplasso_backward(reg_e_grid, reg_cl_grid, X_source, y_source,
                           X_target, y_target, clf, metric='sqeuclidean',
                           kfold=None, norm=None, verbose=False):
    """
    Grid search for Group-Lasso regularization (Backward OT).

    Parameters
    ----------
    reg_e_grid : list
        Entropic regularization grid.
    reg_cl_grid : list
        Group-lasso regularization grid.
    X_source : ndarray
        Source data.
    y_source : ndarray
        Source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier already trained on source.
    metric : str
        Distance metric.
    kfold : dict or None
        CV config.
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    list [reg_e, reg_cl]
        Best regularization parameters.
    """
    results = np.empty((len(reg_e_grid), len(reg_cl_grid)))

    for i, reg_e in enumerate(reg_e_grid):
        for j, reg_cl in enumerate(reg_cl_grid):
            botda = ot.da.SinkhornL1l2Transport(
                metric=metric, reg_e=reg_e, reg_cl=reg_cl, norm=norm, verbose=False
            )

            if kfold is None:
                botda.fit(Xs=X_target, ys=y_target, Xt=X_source)
                X_target_transported = botda.transform(Xs=X_target)
                y_pred = clf.predict(X_target_transported)
                acc = accuracy_score(y_target, y_pred)
                results[i, j] = acc
            else:
                acc_cv = []
                for k in range(kfold['nfold']):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_target, y_target, train_size=kfold['train_size'],
                        stratify=y_target, random_state=100*k
                    )
                    botda.fit(Xs=X_train, ys=y_train, Xt=X_source)
                    X_test_transported = botda.transform(Xs=X_test)
                    y_pred = clf.predict(X_test_transported)
                    acc_cv.append(accuracy_score(y_test, y_pred))
                results[i, j] = np.mean(acc_cv)

    best_idx = np.unravel_index(results.argmax(), results.shape)
    best_reg = [reg_e_grid[best_idx[0]], reg_cl_grid[best_idx[1]]]

    if verbose:
        print(f'Best params: reg_e={best_reg[0]}, reg_cl={best_reg[1]}')
        print(f'Accuracy matrix:\n{results}')

    return best_reg
