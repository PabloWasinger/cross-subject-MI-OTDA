"""
Compare k-fold vs hyperplane subset selection for BOTDA.

Reuses functions from validation.py to evaluate:
- cv_sinkhorn_backward (k-fold)
- cv_grouplasso_backward (k-fold)  
- cv_sinkhorn_backward_distance (hyperplane)
- cv_grouplasso_backward_distance (hyperplane)
"""

import numpy as np
import json
import time
import ot
from sklearn.metrics import accuracy_score
from pathlib import Path

from validation import (
    cv_sinkhorn_backward,
    cv_grouplasso_backward,
    cv_sinkhorn_backward_distance,
    cv_grouplasso_backward_distance
)


def evaluate_botda_methods(X_source, y_source, X_target, y_target, clf,
                           reg_e_grid=None, reg_cl_grid=None,
                           M=40, outerkfold=20, innerkfold=None,
                           metric='sqeuclidean', norm=None, verbose=False):
    """
    Evaluate all 4 BOTDA subset selection methods.
    
    Parameters
    ----------
    X_source : ndarray, shape (n_source, n_features)
        Source domain features (e.g., CSP features).
    y_source : ndarray, shape (n_source,)
        Source labels.
    X_target : ndarray, shape (n_target, n_features)
        Target domain features.
    y_target : ndarray, shape (n_target,)
        Target labels.
    clf : sklearn classifier
        Classifier ALREADY TRAINED on X_source, y_source.
    reg_e_grid : list, optional
        Entropic regularization grid. Default: [0.1, 1, 10]
    reg_cl_grid : list, optional
        Group-lasso regularization grid. Default: [0.1, 1, 10]
    M : int
        Number of samples for subset (total for k-fold, per-class for hyperplane).
    outerkfold : int
        Number of random subsets to try for k-fold methods.
    innerkfold : dict or None
        Inner CV config: {'nfold': int, 'train_size': float}.
    metric : str
        Distance metric for OT.
    norm : str or None
        Cost matrix normalization.
    verbose : bool
        Print progress.
    
    Returns
    -------
    results : dict
        Dictionary with accuracies, times, and hyperparameters for each method.
    """
    if reg_e_grid is None:
        reg_e_grid = [0.1, 1, 10]
    if reg_cl_grid is None:
        reg_cl_grid = [0.1, 1, 10]
    
    results = {}
    
    # =========================================================================
    # 1. BOTDA-Sinkhorn with k-fold subset selection
    # =========================================================================
    if verbose:
        print("Running BOTDA-Sinkhorn (k-fold)...")
    
    t0 = time.time()
    X_subset, y_subset, reg_e = cv_sinkhorn_backward(
        reg_e_grid, X_source, y_source, X_target, y_target, clf,
        metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
        M=M, norm=norm, verbose=verbose
    )
    time_kfold_s = time.time() - t0
    
    # Evaluate with best subset and params
    bot = ot.da.SinkhornTransport(metric=metric, reg_e=reg_e, norm=norm, verbose=False)
    bot.fit(Xs=X_target, Xt=X_subset)
    X_target_transported = bot.transform(Xs=X_target)
    y_pred = clf.predict(X_target_transported)
    acc_kfold_s = accuracy_score(y_target, y_pred)
    
    results['botda_sinkhorn_kfold'] = {
        'accuracy': acc_kfold_s,
        'time_tuning': time_kfold_s,
        'reg_e': reg_e,
        'subset_size': len(y_subset)
    }
    
    # =========================================================================
    # 2. BOTDA-GroupLasso with k-fold subset selection
    # =========================================================================
    if verbose:
        print("Running BOTDA-GroupLasso (k-fold)...")
    
    t0 = time.time()
    X_subset, y_subset, reg_best = cv_grouplasso_backward(
        reg_e_grid, reg_cl_grid, X_source, y_source, X_target, y_target, clf,
        metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
        M=M, norm=norm, verbose=verbose
    )
    time_kfold_gl = time.time() - t0
    
    # Evaluate with best subset and params
    bot = ot.da.SinkhornL1l2Transport(
        metric=metric, reg_e=reg_best[0], reg_cl=reg_best[1], norm=norm, verbose=False
    )
    bot.fit(Xs=X_target, ys=y_target, Xt=X_subset)
    X_target_transported = bot.transform(Xs=X_target)
    y_pred = clf.predict(X_target_transported)
    acc_kfold_gl = accuracy_score(y_target, y_pred)
    
    results['botda_grouplasso_kfold'] = {
        'accuracy': acc_kfold_gl,
        'time_tuning': time_kfold_gl,
        'reg_e': reg_best[0],
        'reg_cl': reg_best[1],
        'subset_size': len(y_subset)
    }
    
    # =========================================================================
    # 3. BOTDA-Sinkhorn with hyperplane subset selection
    # =========================================================================
    if verbose:
        print("Running BOTDA-Sinkhorn (hyperplane)...")
    
    # Note: M here is per-class for hyperplane method
    M_per_class = M // 2  # For binary classification
    
    t0 = time.time()
    X_subset, y_subset, reg_e = cv_sinkhorn_backward_distance(
        reg_e_grid, X_source, y_source, X_target, y_target, clf,
        metric=metric, innerkfold=innerkfold,
        M=M_per_class, norm=norm, verbose=verbose
    )
    time_hyper_s = time.time() - t0
    
    # Evaluate with best subset and params
    bot = ot.da.SinkhornTransport(metric=metric, reg_e=reg_e, norm=norm, verbose=False)
    bot.fit(Xs=X_target, Xt=X_subset)
    X_target_transported = bot.transform(Xs=X_target)
    y_pred = clf.predict(X_target_transported)
    acc_hyper_s = accuracy_score(y_target, y_pred)
    
    results['botda_sinkhorn_hyperplane'] = {
        'accuracy': acc_hyper_s,
        'time_tuning': time_hyper_s,
        'reg_e': reg_e,
        'subset_size': len(y_subset)
    }
    
    # =========================================================================
    # 4. BOTDA-GroupLasso with hyperplane subset selection
    # =========================================================================
    if verbose:
        print("Running BOTDA-GroupLasso (hyperplane)...")
    
    t0 = time.time()
    X_subset, y_subset, reg_best = cv_grouplasso_backward_distance(
        reg_e_grid, reg_cl_grid, X_source, y_source, X_target, y_target, clf,
        metric=metric, innerkfold=innerkfold,
        M=M_per_class, norm=norm, verbose=verbose
    )
    time_hyper_gl = time.time() - t0
    
    # Evaluate with best subset and params
    bot = ot.da.SinkhornL1l2Transport(
        metric=metric, reg_e=reg_best[0], reg_cl=reg_best[1], norm=norm, verbose=False
    )
    bot.fit(Xs=X_target, ys=y_target, Xt=X_subset)
    X_target_transported = bot.transform(Xs=X_target)
    y_pred = clf.predict(X_target_transported)
    acc_hyper_gl = accuracy_score(y_target, y_pred)
    
    results['botda_grouplasso_hyperplane'] = {
        'accuracy': acc_hyper_gl,
        'time_tuning': time_hyper_gl,
        'reg_e': reg_best[0],
        'reg_cl': reg_best[1],
        'subset_size': len(y_subset)
    }
    
    # =========================================================================
    # Summary
    # =========================================================================
    results['summary'] = {
        'sinkhorn_diff': acc_hyper_s - acc_kfold_s,
        'grouplasso_diff': acc_hyper_gl - acc_kfold_gl,
        'sinkhorn_speedup': time_kfold_s / max(time_hyper_s, 1e-6),
        'grouplasso_speedup': time_kfold_gl / max(time_hyper_gl, 1e-6)
    }
    
    return results


def save_results(results, output_path):
    """Save results to JSON file."""
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results_clean = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"Results saved to {output_path}")


def print_results(results):
    """Print results in a readable format."""
    print("\n" + "="*60)
    print("BOTDA Subset Selection Comparison")
    print("="*60)

    print("\n{:<35} {:>10} {:>10}".format("Method", "Accuracy", "Time (s)"))
    print("-"*60)

    for method in ['botda_sinkhorn_kfold', 'botda_sinkhorn_hyperplane',
                   'botda_grouplasso_kfold', 'botda_grouplasso_hyperplane']:
        r = results[method]
        print("{:<35} {:>10.4f} {:>10.2f}".format(
            method, r['accuracy'], r['time_tuning']
        ))

    print("\n" + "-"*60)
    print("Summary:")
    s = results['summary']
    print(f"  Sinkhorn:   hyperplane - kfold = {s['sinkhorn_diff']:+.4f} acc, {s['sinkhorn_speedup']:.1f}x faster")
    print(f"  GroupLasso: hyperplane - kfold = {s['grouplasso_diff']:+.4f} acc, {s['grouplasso_speedup']:.1f}x faster")
    print("="*60)


if __name__ == "__main__":
    from pathlib import Path
    from structuredata import load_session_binary_mi
    from linear_classifier import CSP_LDA

    print("="*60)
    print("BOTDA Subset Selection: K-fold vs Hyperplane Distance")
    print("="*60)

    # Data paths
    data_dir = Path(__file__).parent.parent / "data"
    source_path = data_dir / "A01E.gdf"
    target_path = data_dir / "A01T.gdf"

    # EEG channel configuration (BNCI2014_001 dataset)
    eeg_channels = [
        'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
        'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9',
        'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz',
        'EEG-15', 'EEG-16'
    ]  # 22 EEG channels
    eog_channels = ['EOG-left', 'EOG-central', 'EOG-right']

    # Load source domain (subject A01, evaluation session)
    print(f"\nLoading source: {source_path.name}")
    X_source, y_source, _ = load_session_binary_mi(
        str(source_path),
        eeg_channels,
        eog_channels,
        tmin=0.5,
        tmax=2.5,
        l_freq=8,
        h_freq=30,
        verbose=True
    )

    # Load target domain (subject A01, training session)
    print(f"\nLoading target: {target_path.name}")
    X_target, y_target, _ = load_session_binary_mi(
        str(target_path),
        eeg_channels,
        eog_channels,
        tmin=0.5,
        tmax=2.5,
        l_freq=8,
        h_freq=30,
        verbose=True
    )

    # Create and train CSP+LDA classifier on source
    print("\nTraining CSP+LDA on source domain...")
    clf = CSP_LDA(n_components=4, reg='auto')
    clf.fit(X_source, y_source)

    # Baseline accuracy (no adaptation)
    baseline_acc = clf.score(X_target, y_target)
    print(f"Baseline accuracy (no BOTDA): {baseline_acc:.4f}")

    # Extract CSP features for OTDA
    print("\nExtracting CSP features...")
    X_source_csp = clf.transform(X_source)
    X_target_csp = clf.transform(X_target)

    print(f"Source CSP features: {X_source_csp.shape}")
    print(f"Target CSP features: {X_target_csp.shape}")

    # Run BOTDA comparison
    print("\n" + "="*60)
    print("Running BOTDA methods...")
    print("="*60)

    results = evaluate_botda_methods(
        X_source_csp, y_source,
        X_target_csp, y_target,
        clf,
        reg_e_grid=[0.1, 1.0, 10.0],
        reg_cl_grid=[0.1, 1.0, 10.0],
        M=40,
        outerkfold=10,
        innerkfold={'nfold': 3, 'train_size': 0.7},
        metric='sqeuclidean',
        norm='median',
        verbose=True
    )

    # Print results
    print_results(results)

    # Save results
    output_path = Path(__file__).parent.parent / "results" / "botda_comparison.json"
    save_results(results, output_path)

    print(f"\nBaseline (no BOTDA): {baseline_acc:.4f}")

