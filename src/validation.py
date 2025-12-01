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
import json
from tqdm import tqdm
from pathlib import Path
from sklearn.base import clone


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


def select_subset_by_distance(X_source, y_source, clf, M_total=40):
    """
    Selects M_total samples total based on distance to decision boundary.
    
    Parameters
    ----------
    M_total : int
        TOTAL number of samples to select (will be balanced across classes).
    """
    d = distance_to_hyperplane(X_source, clf)
    idx_wrong = wrong_classified(clf, X_source, y_source)
    d[idx_wrong] = -np.inf

    idx_sorted = np.argsort(d)[::-1]
    X_sorted = X_source[idx_sorted, :]
    y_sorted = y_source[idx_sorted]

    classes = np.unique(y_sorted)
    n_classes = len(classes)
    M_per_class = M_total // n_classes  # Equal distribution
    
    X_subset = []
    y_subset = []

    for c in classes:
        idx_class = np.where(y_sorted == c)[0]
        n_select = min(M_per_class, len(idx_class))
        X_subset.append(X_sorted[idx_class[:n_select], :])
        y_subset.append(y_sorted[idx_class[:n_select]])

    X_subset = np.vstack(X_subset)
    y_subset = np.hstack(y_subset)

    return X_subset, y_subset


def cv_sinkhorn(reg_e_grid, X_source, y_source, X_target, y_target, clf,
                metric='sqeuclidean', outerkfold=20, innerkfold=None, M=40,
                norm=None, verbose=False):
    """
    Select subset of source data and find best reg parameter for Sinkhorn (Forward OT).

    Parameters
    ----------
    reg_e_grid : list
        Grid of entropic regularization values.
    X_source : ndarray
        FULL source data.
    y_source : ndarray
        FULL source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier to train on transported source.
    metric : str
        Distance metric.
    outerkfold : int
        Number of times to resample the subset (default: 20).
    innerkfold : dict or None
        Inner CV config for hyperparameter search: {'nfold': int, 'train_size': float}.
    M : int
        Number of samples to include in subset (default: 40).
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    tuple (X_subset, y_subset, best_reg_e)
        Best selected subset and regularization parameter.
    """
    acc_cv = []
    lista_xs = []
    lista_ys = []
    regu_ = []

    for k in range(outerkfold):
        # Random subset selection
        xs_subset, X_test, ys_subset, y_test = train_test_split(
            X_source, y_source, train_size=M, stratify=y_source, random_state=100*k
        )

        lista_xs.append(xs_subset)
        lista_ys.append(ys_subset)

        # Find best reg_e for this subset
        if len(reg_e_grid) == 1:
            regu = reg_e_grid[0]
        else:
            # Inner CV for hyperparameter tuning
            results = []
            for reg_e in reg_e_grid:
                ot_sinkhorn = ot.da.SinkhornTransport(
                    metric=metric, reg_e=reg_e, norm=norm, verbose=False
                )

                if innerkfold is None:
                    # fit with subset, transform ALL source
                    ot_sinkhorn.fit(Xs=xs_subset, Xt=X_target)
                    X_source_transported = ot_sinkhorn.transform(Xs=X_source)
                    clf_temp = clone(clf)
                    clf_temp.fit(X_source_transported, y_source)
                    y_pred = clf_temp.predict(X_target)
                    acc = accuracy_score(y_target, y_pred)
                    results.append(acc)
                else:
                    acc_inner = []
                    for kk in range(innerkfold['nfold']):
                        X_train, X_test_inner, y_train, y_test_inner = train_test_split(
                            X_target, y_target, train_size=innerkfold['train_size'],
                            stratify=y_target, random_state=100*kk
                        )
                        # fit with subset, transform ALL source
                        ot_sinkhorn.fit(Xs=xs_subset, Xt=X_train)
                        X_source_transported = ot_sinkhorn.transform(Xs=X_source)
                        clf_temp = clone(clf)
                        clf_temp.fit(X_source_transported, y_source)
                        y_pred = clf_temp.predict(X_test_inner)
                        acc_inner.append(accuracy_score(y_test_inner, y_pred))
                    results.append(np.mean(acc_inner))

            best_idx = np.argmax(results)
            regu = reg_e_grid[best_idx]

        regu_.append(regu)

        # Evaluate this subset + regu combination on full target
        # fit with subset, transform ALL source
        ot_sinkhorn = ot.da.SinkhornTransport(metric=metric, reg_e=regu, norm=norm, verbose=False)
        ot_sinkhorn.fit(Xs=xs_subset, Xt=X_target)
        X_source_transported = ot_sinkhorn.transform(Xs=X_source)
        clf_eval = clone(clf)
        clf_eval.fit(X_source_transported, y_source)
        acc_cv.append(clf_eval.score(X_target, y_target))

    # Select best subset based on accuracy
    best_idx = np.argmax(acc_cv)
    subset_xs = lista_xs[best_idx]
    subset_ys = lista_ys[best_idx]
    reg_best = regu_[best_idx]

    if verbose:
        print(f'Best reg_e: {reg_best}')
        print(f'Accuracy grid: {acc_cv}')

    return subset_xs, subset_ys, reg_best


def cv_grouplasso(reg_e_grid, reg_cl_grid, X_source, y_source, X_target, y_target,
                  clf, metric='sqeuclidean', outerkfold=20, innerkfold=None, M=40,
                  norm=None, verbose=False):
    """
    Select subset of source data and find best reg parameters for Group-Lasso (Forward OT).

    Parameters
    ----------
    reg_e_grid : list
        Entropic regularization grid.
    reg_cl_grid : list
        Group-lasso regularization grid.
    X_source : ndarray
        FULL source data.
    y_source : ndarray
        FULL source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier to train on transported source.
    metric : str
        Distance metric.
    outerkfold : int
        Number of times to resample the subset (default: 20).
    innerkfold : dict or None
        Inner CV config for hyperparameter search: {'nfold': int, 'train_size': float}.
    M : int
        Number of samples to include in subset (default: 40).
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    tuple (X_subset, y_subset, [best_reg_e, best_reg_cl])
        Best selected subset and regularization parameters.
    """
    acc_cv = []
    lista_xs = []
    lista_ys = []
    regu_ = []

    for k in range(outerkfold):
        # Random subset selection
        xs_subset, X_test, ys_subset, y_test = train_test_split(
            X_source, y_source, train_size=M, stratify=y_source, random_state=100*k
        )

        lista_xs.append(xs_subset)
        lista_ys.append(ys_subset)

        # Find best reg_e and reg_cl for this subset
        if len(reg_e_grid) == 1 and len(reg_cl_grid) == 1:
            regu = [reg_e_grid[0], reg_cl_grid[0]]
        else:
            # Inner CV for hyperparameter tuning
            results = np.empty((len(reg_e_grid), len(reg_cl_grid)))

            for i, reg_e in enumerate(reg_e_grid):
                for j, reg_cl in enumerate(reg_cl_grid):
                    ot_l1l2 = ot.da.SinkhornL1l2Transport(
                        metric=metric, reg_e=reg_e, reg_cl=reg_cl, norm=norm, verbose=False
                    )

                    if innerkfold is None:
                        # fit with subset, transform ALL source (unsupervised)
                        ot_l1l2.fit(Xs=xs_subset, ys=ys_subset, Xt=X_target)
                        X_source_transported = ot_l1l2.transform(Xs=X_source)
                        # Train with source transported only
                        clf_temp = clone(clf)
                        clf_temp.fit(X_source_transported, y_source)
                        y_pred = clf_temp.predict(X_target)
                        acc = accuracy_score(y_target, y_pred)
                        results[i, j] = acc
                    else:
                        acc_inner = []
                        for kk in range(innerkfold['nfold']):
                            X_train, X_test_inner, y_train, y_test_inner = train_test_split(
                                X_target, y_target, train_size=innerkfold['train_size'],
                                stratify=y_target, random_state=100*kk
                            )
                            # fit with subset, transform ALL source (unsupervised)
                            ot_l1l2.fit(Xs=xs_subset, ys=ys_subset, Xt=X_train)
                            X_source_transported = ot_l1l2.transform(Xs=X_source)
                            # Train with source transported only
                            clf_temp = clone(clf)
                            clf_temp.fit(X_source_transported, y_source)
                            y_pred = clf_temp.predict(X_test_inner)
                            acc_inner.append(accuracy_score(y_test_inner, y_pred))
                        results[i, j] = np.mean(acc_inner)

            best_idx = np.unravel_index(results.argmax(), results.shape)
            regu = [reg_e_grid[best_idx[0]], reg_cl_grid[best_idx[1]]]

        regu_.append(regu)

        # Evaluate this subset + regu combination on full target
        # fit with subset, transform ALL source (unsupervised)
        ot_l1l2 = ot.da.SinkhornL1l2Transport(
            metric=metric, reg_e=regu[0], reg_cl=regu[1], norm=norm, verbose=False
        )
        ot_l1l2.fit(Xs=xs_subset, ys=ys_subset, Xt=X_target)
        X_source_transported = ot_l1l2.transform(Xs=X_source)
        # Train with source transported only
        clf_eval = clone(clf)
        clf_eval.fit(X_source_transported, y_source)
        acc_cv.append(clf_eval.score(X_target, y_target))

    # Select best subset based on accuracy
    best_idx = np.argmax(acc_cv)
    subset_xs = lista_xs[best_idx]
    subset_ys = lista_ys[best_idx]
    reg_best = regu_[best_idx]

    if verbose:
        print(f'Best params: reg_e={reg_best[0]}, reg_cl={reg_best[1]}')
        print(f'Accuracy grid: {acc_cv}')

    return subset_xs, subset_ys, reg_best


def cv_sinkhorn_backward(reg_e_grid, X_source, y_source, X_target, y_target, clf,
                         metric='sqeuclidean', outerkfold=20, innerkfold=None, M=40,
                         norm=None, verbose=False):
    """
    Select subset of source data and find best reg parameter for Sinkhorn (Backward OT).

    Parameters
    ----------
    reg_e_grid : list
        Entropic regularization grid.
    X_source : ndarray
        FULL source data.
    y_source : ndarray
        FULL source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier ALREADY trained on source (before calling this function).
    metric : str
        Distance metric.
    outerkfold : int
        Number of times to resample the subset (default: 20).
    innerkfold : dict or None
        Inner CV config for hyperparameter search: {'nfold': int, 'train_size': float}.
    M : int
        Number of samples to include in subset (default: 40).
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    tuple (X_subset, y_subset, best_reg_e)
        Best selected subset and regularization parameter.
    """
    acc_cv = []
    lista_xs = []
    lista_ys = []
    regu_ = []

    for k in range(outerkfold):
        # Random subset selection from source
        xs_subset, X_test, ys_subset, y_test = train_test_split(
            X_source, y_source, train_size=M, stratify=y_source, random_state=100*k
        )

        lista_xs.append(xs_subset)
        lista_ys.append(ys_subset)

        # Find best reg_e for this subset
        if len(reg_e_grid) == 1:
            regu = reg_e_grid[0]
        else:
            # Inner CV for hyperparameter tuning
            results = []
            for reg_e in reg_e_grid:
                bot = ot.da.SinkhornTransport(
                    metric=metric, reg_e=reg_e, norm=norm, verbose=False
                )

                if innerkfold is None:
                    # fit: target → source subset
                    bot.fit(Xs=X_target, Xt=xs_subset)
                    X_target_transported = bot.transform(Xs=X_target)
                    y_pred = clf.predict(X_target_transported)
                    acc = accuracy_score(y_target, y_pred)
                    results.append(acc)
                else:
                    acc_inner = []
                    for kk in range(innerkfold['nfold']):
                        X_train, X_test_inner, y_train, y_test_inner = train_test_split(
                            X_target, y_target, train_size=innerkfold['train_size'],
                            stratify=y_target, random_state=100*kk
                        )
                        # fit: target train → source subset
                        bot.fit(Xs=X_train, Xt=xs_subset)
                        X_test_transported = bot.transform(Xs=X_test_inner)
                        y_pred = clf.predict(X_test_transported)
                        acc_inner.append(accuracy_score(y_test_inner, y_pred))
                    results.append(np.mean(acc_inner))

            best_idx = np.argmax(results)
            regu = reg_e_grid[best_idx]

        regu_.append(regu)

        # Evaluate this subset + regu combination on full target
        # fit: target → source subset
        bot = ot.da.SinkhornTransport(metric=metric, reg_e=regu, norm=norm, verbose=False)
        bot.fit(Xs=X_target, Xt=xs_subset)
        X_target_transported = bot.transform(Xs=X_target)
        y_pred = clf.predict(X_target_transported)
        acc_cv.append(accuracy_score(y_target, y_pred))

    # Select best subset based on accuracy
    best_idx = np.argmax(acc_cv)
    subset_xs = lista_xs[best_idx]
    subset_ys = lista_ys[best_idx]
    reg_best = regu_[best_idx]

    if verbose:
        print(f'Best reg_e: {reg_best}')
        print(f'Accuracy grid: {acc_cv}')

    return subset_xs, subset_ys, reg_best


def cv_grouplasso_backward(reg_e_grid, reg_cl_grid, X_source, y_source,
                           X_target, y_target, clf, metric='sqeuclidean',
                           outerkfold=20, innerkfold=None, M=40,
                           norm=None, verbose=False):
    """
    Select subset of source data and find best reg parameters for Group-Lasso (Backward OT).

    Parameters
    ----------
    reg_e_grid : list
        Entropic regularization grid.
    reg_cl_grid : list
        Group-lasso regularization grid.
    X_source : ndarray
        FULL source data.
    y_source : ndarray
        FULL source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier ALREADY trained on source (before calling this function).
    metric : str
        Distance metric.
    outerkfold : int
        Number of times to resample the subset (default: 20).
    innerkfold : dict or None
        Inner CV config for hyperparameter search: {'nfold': int, 'train_size': float}.
    M : int
        Number of samples to include in subset (default: 40).
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    tuple (X_subset, y_subset, [best_reg_e, best_reg_cl])
        Best selected subset and regularization parameters.
    """
    acc_cv = []
    lista_xs = []
    lista_ys = []
    regu_ = []

    for k in range(outerkfold):
        # Random subset selection from source
        xs_subset, X_test, ys_subset, y_test = train_test_split(
            X_source, y_source, train_size=M, stratify=y_source, random_state=100*k
        )

        lista_xs.append(xs_subset)
        lista_ys.append(ys_subset)

        # Find best reg_e and reg_cl for this subset
        if len(reg_e_grid) == 1 and len(reg_cl_grid) == 1:
            regu = [reg_e_grid[0], reg_cl_grid[0]]
        else:
            # Inner CV for hyperparameter tuning
            results = np.empty((len(reg_e_grid), len(reg_cl_grid)))

            for i, reg_e in enumerate(reg_e_grid):
                for j, reg_cl in enumerate(reg_cl_grid):
                    botda = ot.da.SinkhornL1l2Transport(
                        metric=metric, reg_e=reg_e, reg_cl=reg_cl, norm=norm, verbose=False
                    )

                    if innerkfold is None:
                        # fit: target → source subset (unsupervised)
                        botda.fit(Xs=X_target, ys=y_target, Xt=xs_subset)
                        X_target_transported = botda.transform(Xs=X_target)
                        y_pred = clf.predict(X_target_transported)
                        acc = accuracy_score(y_target, y_pred)
                        results[i, j] = acc
                    else:
                        acc_inner = []
                        for kk in range(innerkfold['nfold']):
                            X_train, X_test_inner, y_train, y_test_inner = train_test_split(
                                X_target, y_target, train_size=innerkfold['train_size'],
                                stratify=y_target, random_state=100*kk
                            )
                            # fit: target train → source subset (unsupervised)
                            botda.fit(Xs=X_train, ys=y_train, Xt=xs_subset)
                            X_test_transported = botda.transform(Xs=X_test_inner)
                            y_pred = clf.predict(X_test_transported)
                            acc_inner.append(accuracy_score(y_test_inner, y_pred))
                        results[i, j] = np.mean(acc_inner)

            best_idx = np.unravel_index(results.argmax(), results.shape)
            regu = [reg_e_grid[best_idx[0]], reg_cl_grid[best_idx[1]]]

        regu_.append(regu)

        # Evaluate this subset + regu combination on full target
        # fit: target → source subset (unsupervised)
        botda = ot.da.SinkhornL1l2Transport(
            metric=metric, reg_e=regu[0], reg_cl=regu[1], norm=norm, verbose=False
        )
        botda.fit(Xs=X_target, ys=y_target, Xt=xs_subset)
        X_target_transported = botda.transform(Xs=X_target)
        y_pred = clf.predict(X_target_transported)
        acc_cv.append(accuracy_score(y_target, y_pred))

    # Select best subset based on accuracy
    best_idx = np.argmax(acc_cv)
    subset_xs = lista_xs[best_idx]
    subset_ys = lista_ys[best_idx]
    reg_best = regu_[best_idx]

    if verbose:
        print(f'Best params: reg_e={reg_best[0]}, reg_cl={reg_best[1]}')
        print(f'Accuracy grid: {acc_cv}')

    return subset_xs, subset_ys, reg_best


def cv_sinkhorn_backward_distance(reg_e_grid, X_source, y_source, X_target, y_target, clf,
                                    metric='sqeuclidean', innerkfold=None, M=20,
                                    norm=None, verbose=False):
    """
    Select subset by distance to hyperplane and find best reg parameter for Sinkhorn (Backward OT).

    Parameters
    ----------
    reg_e_grid : list
        Entropic regularization grid.
    X_source : ndarray
        FULL source data.
    y_source : ndarray
        FULL source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier ALREADY trained on source (before calling this function).
    metric : str
        Distance metric.
    innerkfold : dict or None
        Inner CV config for hyperparameter search: {'nfold': int, 'train_size': float}.
    M : int
        Number of samples per class to include in subset (default: 20).
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    tuple (X_subset, y_subset, best_reg_e)
        Best selected subset and regularization parameter.
    """
    # Select subset using distance to hyperplane
    xs_subset, ys_subset = select_subset_by_distance(X_source, y_source, clf, M)

    # Find best reg_e for this subset
    if len(reg_e_grid) == 1:
        reg_best = reg_e_grid[0]
    else:
        # Inner CV for hyperparameter tuning
        results = []
        for reg_e in reg_e_grid:
            bot = ot.da.SinkhornTransport(
                metric=metric, reg_e=reg_e, norm=norm, verbose=False
            )

            if innerkfold is None:
                # fit: target → source subset
                bot.fit(Xs=X_target, Xt=xs_subset)
                X_target_transported = bot.transform(Xs=X_target)
                y_pred = clf.predict(X_target_transported)
                acc = accuracy_score(y_target, y_pred)
                results.append(acc)
            else:
                acc_inner = []
                for kk in range(innerkfold['nfold']):
                    X_train, X_test_inner, y_train, y_test_inner = train_test_split(
                        X_target, y_target, train_size=innerkfold['train_size'],
                        stratify=y_target, random_state=100*kk
                    )
                    # fit: target train → source subset
                    bot.fit(Xs=X_train, Xt=xs_subset)
                    X_test_transported = bot.transform(Xs=X_test_inner)
                    y_pred = clf.predict(X_test_transported)
                    acc_inner.append(accuracy_score(y_test_inner, y_pred))
                results.append(np.mean(acc_inner))

        best_idx = np.argmax(results)
        reg_best = reg_e_grid[best_idx]

    if verbose:
        print(f'Best reg_e: {reg_best}')

    return xs_subset, ys_subset, reg_best


def cv_grouplasso_backward_distance(reg_e_grid, reg_cl_grid, X_source, y_source,
                                      X_target, y_target, clf, metric='sqeuclidean',
                                      innerkfold=None, M=20, norm=None, verbose=False):
    """
    Select subset by distance to hyperplane and find best reg parameters for Group-Lasso (Backward OT).

    Parameters
    ----------
    reg_e_grid : list
        Entropic regularization grid.
    reg_cl_grid : list
        Group-lasso regularization grid.
    X_source : ndarray
        FULL source data.
    y_source : ndarray
        FULL source labels.
    X_target : ndarray
        Target data.
    y_target : ndarray
        Target labels.
    clf : sklearn classifier
        Classifier ALREADY trained on source (before calling this function).
    metric : str
        Distance metric.
    innerkfold : dict or None
        Inner CV config for hyperparameter search: {'nfold': int, 'train_size': float}.
    M : int
        Number of samples per class to include in subset (default: 20).
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print results.

    Returns
    -------
    tuple (X_subset, y_subset, [best_reg_e, best_reg_cl])
        Best selected subset and regularization parameters.
    """
    # Select subset using distance to hyperplane
    xs_subset, ys_subset = select_subset_by_distance(X_source, y_source, clf, M)

    # Find best reg_e and reg_cl for this subset
    if len(reg_e_grid) == 1 and len(reg_cl_grid) == 1:
        reg_best = [reg_e_grid[0], reg_cl_grid[0]]
    else:
        # Inner CV for hyperparameter tuning
        results = np.empty((len(reg_e_grid), len(reg_cl_grid)))

        for i, reg_e in enumerate(reg_e_grid):
            for j, reg_cl in enumerate(reg_cl_grid):
                botda = ot.da.SinkhornL1l2Transport(
                    metric=metric, reg_e=reg_e, reg_cl=reg_cl, norm=norm, verbose=False
                )

                if innerkfold is None:
                    # fit: target → source subset (unsupervised)
                    botda.fit(Xs=X_target, ys=y_target, Xt=xs_subset)
                    X_target_transported = botda.transform(Xs=X_target)
                    y_pred = clf.predict(X_target_transported)
                    acc = accuracy_score(y_target, y_pred)
                    results[i, j] = acc
                else:
                    acc_inner = []
                    for kk in range(innerkfold['nfold']):
                        X_train, X_test_inner, y_train, y_test_inner = train_test_split(
                            X_target, y_target, train_size=innerkfold['train_size'],
                            stratify=y_target, random_state=100*kk
                        )
                        # fit: target train → source subset (unsupervised)
                        botda.fit(Xs=X_train, ys=y_train, Xt=xs_subset)
                        X_test_transported = botda.transform(Xs=X_test_inner)
                        y_pred = clf.predict(X_test_transported)
                        acc_inner.append(accuracy_score(y_test_inner, y_pred))
                    results[i, j] = np.mean(acc_inner)

        best_idx = np.unravel_index(results.argmax(), results.shape)
        reg_best = [reg_e_grid[best_idx[0]], reg_cl_grid[best_idx[1]]]

    if verbose:
        print(f'Best params: reg_e={reg_best[0]}, reg_cl={reg_best[1]}')

    return xs_subset, ys_subset, reg_best


def cv_all_methods(reg_e_grid, reg_cl_grid, X_source, y_source, X_target, y_target,
                   clf, metric='sqeuclidean', outerkfold=20, innerkfold=None, M=40,
                   norm=None, verbose=False):
    """
    Perform cross-validation for all transfer learning methods and return optimal parameters and subsamples.

    Parameters
    ----------
    reg_e_grid : list
        Entropic regularization grid.
    reg_cl_grid : list
        Group-lasso regularization grid.
    X_source : ndarray
        Source domain data.
    y_source : ndarray
        Source domain labels.
    X_target : ndarray
        Target domain data.
    y_target : ndarray
        Target domain labels.
    clf : sklearn classifier
        Classifier already trained on source.
    metric : str
        Distance metric.
    outerkfold : int
        Number of outer CV folds.
    innerkfold : int or None
        Number of inner CV folds for hyperparameter search.
    M : int
        Number of samples for subset.
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print progress.

    Returns
    -------
    subsamples : dict
        Dictionary with selected subsamples for each method (keys: X and y for each method).
    reg_params : dict
        Dictionary with optimal regularization parameters for each method.
    """

    if verbose:
        print("Running Forward Sinkhorn CV...")
    X_fs, y_fs, reg_fs = cv_sinkhorn(
        reg_e_grid, X_source, y_source, X_target, y_target, clf,
        metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
        M=M, norm=norm, verbose=verbose
    )

    if verbose:
        print("Running Forward GroupLasso CV...")
    X_fg, y_fg, reg_fg = cv_grouplasso(
        reg_e_grid, reg_cl_grid, X_source, y_source, X_target, y_target, clf,
        metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
        M=M, norm=norm, verbose=verbose
    )

    if verbose:
        print("Running Backward Sinkhorn CV...")
    X_bs, y_bs, reg_bs = cv_sinkhorn_backward(
        reg_e_grid, X_source, y_source, X_target, y_target, clf,
        metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
        M=M, norm=norm, verbose=verbose
    )

    if verbose:
        print("Running Backward GroupLasso CV...")
    X_bg, y_bg, reg_bg = cv_grouplasso_backward(
        reg_e_grid, reg_cl_grid, X_source, y_source, X_target, y_target, clf,
        metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
        M=M, norm=norm, verbose=verbose
    )

    X_subsamples = {
        'forward_sinkhorn': X_fs,
        'forward_grouplasso': X_fg,
        'backward_sinkhorn': X_bs,
        'backward_grouplasso': X_bg
    }

    y_subsamples = {
        'forward_sinkhorn': y_fs,
        'forward_grouplasso': y_fg,
        'backward_sinkhorn': y_bs,
        'backward_grouplasso': y_bg
    }

    reg_params = {
        'forward_sinkhorn': reg_fs,
        'forward_grouplasso': reg_fg,
        'backward_sinkhorn': reg_bs,
        'backward_grouplasso': reg_bg
    }

    return X_subsamples, y_subsamples, reg_params


