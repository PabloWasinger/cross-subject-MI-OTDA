import numpy as np
import pandas as pd    
from transfer_learning_methods import*
from validation import*

# rango_cl=[0.1, 1, 10]
# rango_e=[0.1, 1, 10] 
# metric = 'sqeuclidean'
# outerkfold = 20
# innerkfold = None
# M=20
# norm=None

def evaluate_tl_methods_blockwise(X_source, y_source, X_target, y_target, 
                                   cv_params, trials_per_run=24, verbose=True):
    """
    Evaluate transfer learning methods with block-wise adaptation.
    

    - First run R0 (trials 0:trials_per_run) is used for calibration only
    - For each subsequent run r, the transportation set Vr accumulates all prior data
    - M (source subset size) equals the size of the transportation set
    - Hyperparameters are re-optimized for each run
    
    Parameters
    ----------
    X_source : ndarray, shape (n_source_trials, n_channels, n_samples)
        Source domain EEG data (e.g., calibration session).
    y_source : ndarray, shape (n_source_trials,)
        Source domain labels.
    X_target : ndarray, shape (n_target_trials, n_channels, n_samples)
        Target domain EEG data (e.g., evaluation session).
    y_target : ndarray, shape (n_target_trials,)
        Target domain labels.
    cv_params : dict
        Cross-validation parameters with keys:
        - 'reg_e_grid': list of entropic regularization values
        - 'reg_cl_grid': list of group-lasso regularization values  
        - 'metric': distance metric (default 'sqeuclidean')
        - 'outerkfold': number of outer CV folds for subset selection
        - 'innerkfold': inner CV config dict or None
        - 'norm': cost matrix normalization or None
    trials_per_run : int, optional
        Number of trials per run/block. Default is 24.
    verbose : bool, optional
        Print progress messages. Default is True.
    
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'accuracies': dict of method_name -> list of accuracies per run
        - 'times': dict of method_name -> list of times per run
        - 'run_indices': list of run numbers that were tested
    
    Notes
    -----
    Timing includes both hyperparameter optimization (CV) and adaptation/prediction,
    which differs from Peterson's original timing that excluded CV time.
    This provides a fairer comparison for practical online scenarios.
    """
    

    # Train base model on source domain
    csp = CSP(n_components=6, reg='empirical', log=True, norm_trace=False, cov_est='epoch')
    G_source = csp.fit_transform(X_source, y_source)
    
    clf_base = LinearDiscriminantAnalysis()
    clf_base.fit(G_source, y_source)
    
    # Determine run structure
    n_target_trials = len(y_target)
    n_runs = n_target_trials // trials_per_run
    
    if verbose:
        print(f"Target session: {n_target_trials} trials -> {n_runs} runs of {trials_per_run} trials")
        print(f"R0 (first {trials_per_run} trials) used for calibration only")
        print(f"Testing runs: R1 to R{n_runs-1}")
    
    # Extract CV parameters
    reg_e_grid = cv_params['reg_e_grid']
    reg_cl_grid = cv_params['reg_cl_grid']
    metric = cv_params.get('metric', 'sqeuclidean')
    outerkfold = cv_params.get('outerkfold', 20)
    innerkfold = cv_params.get('innerkfold', None)
    norm = cv_params.get('norm', None)
    
    # Initialize results storage
    methods = ['SC', 'SR', 'Forward_Sinkhorn', 'Forward_GroupLasso',
               'Backward_Sinkhorn', 'Backward_GroupLasso', 'RPA', 'EU']
    
    predictions = {method: [] for method in methods}
    accuracies = {method: [] for method in methods}
    times = {method: [] for method in methods}
    run_indices = []
    
    # Iterate over testing runs
    for run_idx in range(1, n_runs):  # Start from run 1 (R0 is calibration)
        if verbose:
            print(f"Processing Run {run_idx} (testing trials {run_idx*trials_per_run}:{(run_idx+1)*trials_per_run})")
        
        run_indices.append(run_idx)
        
        # Define data splits for this run
        
        # Transportation set: all prior data (R0 through R_{run_idx-1})
        val_end_idx = run_idx * trials_per_run
        X_val_raw = X_target[:val_end_idx]
        y_val = y_target[:val_end_idx]
        
        # Testing run data
        test_start_idx = run_idx * trials_per_run
        test_end_idx = (run_idx + 1) * trials_per_run
        X_test_raw = X_target[test_start_idx:test_end_idx]
        y_test = y_target[test_start_idx:test_end_idx]
        
        # Transform to CSP space
        G_val = csp.transform(X_val_raw)
        G_test = csp.transform(X_test_raw)
        
        M = min(len(y_val), len(y_source))
        
        if verbose:
            print(f"Transportation set size (M): {M} trials")
            print(f"Test set size: {len(y_test)} trials")
        

        # SC (Source Classifier - no adaptation)
        y_pred_sc, time_sc = SC(G_test, clf_base)
        acc_sc = accuracy_score(y_test, y_pred_sc)

        predictions['SC'].append(y_pred_sc)
        accuracies['SC'].append(acc_sc)
        times['SC'].append(time_sc)

        # SR (Standard Recalibration)
        y_pred_sr, time_sr = SR(X_target, y_target, run_idx * trials_per_run, X_source, y_source, X_test_raw)
        acc_sr = accuracy_score(y_test, y_pred_sr)

        predictions['SR'].append(y_pred_sr)
        accuracies['SR'].append(acc_sr)
        times['SR'].append(time_sr)
        

        # Forward Sinkhorn Transport
        start_time = timeit.default_timer()

        # CV for subset selection and hyperparameter tuning
        clf_forward = clone(clf_base)
        G_fs, _, reg_fs = cv_sinkhorn(
            reg_e_grid, G_source, y_source, G_val, y_val, clf_forward,
            metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
            M=M, norm=norm, verbose=False
        )

        time_cv = timeit.default_timer() - start_time

        # Apply transport 
        y_pred_fs, time_transport = Forward_Sinkhorn_Transport(
            G_fs, reg_fs, G_source, y_source, G_val, G_test, clf_base, metric
        )
        acc_fs = accuracy_score(y_test, y_pred_fs)

        time_fs = time_cv + time_transport

        predictions['Forward_Sinkhorn'].append(y_pred_fs)
        accuracies['Forward_Sinkhorn'].append(acc_fs)
        times['Forward_Sinkhorn'].append(time_fs)
        
        # Forward GroupLasso Transport
        start_time = timeit.default_timer()

        clf_forward_gl = clone(clf_base)
        G_fg, y_fg, reg_fg = cv_grouplasso(
            reg_e_grid, reg_cl_grid, G_source, y_source, G_val, y_val, clf_forward_gl,
            metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
            M=M, norm=norm, verbose=False
        )

        time_cv = timeit.default_timer() - start_time

        # Apply transport 
        y_pred_fg, time_transport = Forward_GroupLasso_Transport(
            G_fg, y_fg, reg_fg, G_source, y_source, G_val, G_test, clf_base, metric
        )
        acc_fg = accuracy_score(y_test, y_pred_fg)
        time_fg = time_cv + time_transport

        predictions['Forward_GroupLasso'].append(y_pred_fg)
        accuracies['Forward_GroupLasso'].append(acc_fg)
        times['Forward_GroupLasso'].append(time_fg)
        
        # Backward Sinkhorn Transport
        start_time = timeit.default_timer()

        # For backward methods, we use the ORIGINAL clf_base (never modified)
        G_bs, _, reg_bs = cv_sinkhorn_backward(
            reg_e_grid, G_source, y_source, G_val, y_val, clf_base,
            metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
            M=M, norm=norm, verbose=False
        )

        time_cv = timeit.default_timer() - start_time

        # Apply transport 
        y_pred_bs, time_transport = Backward_Sinkhorn_Transport(
            G_bs, reg_bs, G_val, G_test, clf_base, metric
        )
        acc_bs = accuracy_score(y_test, y_pred_bs)
        time_bs = time_cv + time_transport

        predictions['Backward_Sinkhorn'].append(y_pred_bs)
        accuracies['Backward_Sinkhorn'].append(acc_bs)
        times['Backward_Sinkhorn'].append(time_bs)
        
        # Backward GroupLasso Transport
        start_time = timeit.default_timer()

        G_bg, _, reg_bg = cv_grouplasso_backward(
            reg_e_grid, reg_cl_grid, G_source, y_source, G_val, y_val, clf_base,
            metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
            M=M, norm=norm, verbose=False
        )

        time_cv = timeit.default_timer() - start_time

        # Apply transport 
        y_pred_bg, time_transport = Backward_GroupLasso_Transport(
            G_bg, reg_bg, G_val, y_val, G_test, clf_base, metric
        )
        acc_bg = accuracy_score(y_test, y_pred_bg)
        time_bg = time_cv + time_transport

        predictions['Backward_GroupLasso'].append(y_pred_bg)
        accuracies['Backward_GroupLasso'].append(acc_bg)
        times['Backward_GroupLasso'].append(time_bg)
        
        # RPA (Riemannian Procrustes Analysis)
        start_time = timeit.default_timer()
        y_pred_rpa, _ = RPA(X_source, X_val_raw, X_test_raw, y_source, y_val, y_test, transductive=False)
        acc_rpa = accuracy_score(y_test, y_pred_rpa)
        time_rpa = timeit.default_timer() - start_time
        
        accuracies['RPA'].append(acc_rpa)
        times['RPA'].append(time_rpa)
        
        # EU (Euclidean Alignment)
        start_time = timeit.default_timer()
        y_pred_eu, _ = EU(X_source, X_val_raw, X_test_raw, y_source, y_val, y_test)
        acc_eu = accuracy_score(y_test, y_pred_eu)
        time_eu = timeit.default_timer() - start_time
        
        accuracies['EU'].append(acc_eu)
        times['EU'].append(time_eu)
        
        predictions['RPA'].append(y_pred_rpa)
        predictions['EU'].append(y_pred_eu)
    
    results = {
        'predictions': predictions,
        'accuracies': accuracies,
        'times': times,
        'run_indices': run_indices
    }
    
    return results



def evaluate_tl_methods_samplewise(X_source, y_source, X_target, y_target, cv_params, n_calib=20,  verbose=True):
    """
    Evaluate transfer learning methods with incremental trial-by-trial calibration using optimized hyperparameters.

    Parameters
    ----------
    X_source : ndarray
        Source domain EEG data, shape (n_trials, n_channels, n_samples).
    y_source : ndarray
        Source domain labels, shape (n_trials,).
    X_target : ndarray
        Target domain EEG data, shape (n_trials, n_channels, n_samples).
    y_target : ndarray
        Target domain labels, shape (n_trials,).
    cv_params : dict
        Dictionary with cross-validation parameters.
        Expected keys: 'reg_e_grid', 'reg_cl_grid', 'metric', 'outerkfold', 'innerkfold', 'M', 'norm'.
    n_calib : int, optional
        Number of calibration trials from target domain. Default is 20.
    verbose : bool, optional
        Print progress messages. Default is True.

    Returns
    -------
    predictions : dict
        Dictionary with method names as keys and lists of predictions as values.
    times : dict
        Dictionary with method names as keys and lists of execution times as values.
    times_se : ndarray
        Matrix of execution times for each trial and method, shape (n_test_trials, 5).
    """

    # X_source on CSP space
    csp = CSP(n_components=6, reg='empirical', log=True, norm_trace=False, cov_est='epoch')
    G_source = csp.fit_transform(X_source, y_source) # We call G to features in CSP space

    # Train Base Classifier (LDA)
    clf_base = LinearDiscriminantAnalysis()
    clf_base.fit(G_source, y_source)


    # Calibration split
    X_val_raw = X_target[:n_calib]
    y_val = y_target[:n_calib]
    X_test_raw = X_target[n_calib:]
    y_test = y_target[n_calib:]

    # Transform Validation to CSP
    G_val = csp.transform(X_val_raw)


    # Hyperparameter Optimization
    reg_e_grid = cv_params['reg_e_grid']
    reg_cl_grid = cv_params['reg_cl_grid']
    metric = cv_params['metric']
    outerkfold = cv_params['outerkfold']
    innerkfold = cv_params['innerkfold']
    M=cv_params['M'] # Number of samples to transport
    norm=cv_params['norm']    

    G_subsamples, y_subsamples, reg_params = cv_all_methods(
        reg_e_grid, reg_cl_grid, G_source, y_source, G_val, y_val, clf_base,
        metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
        M=M, norm=norm, verbose=verbose
    )
    

    predictions ={
                'SC': [],
                'SR': [],
                'Forward_Sinkhorn': [],
                'Forward_GroupLasso': [],
                'Backward_Sinkhorn': [],
                'Backward_GroupLasso': [],
                'RPA': [],
                'EU': []
    }

    times ={
                'SC': [],
                'SR': [],
                'Forward_Sinkhorn': [],
                'Forward_GroupLasso': [],
                'Backward_Sinkhorn': [],
                'Backward_GroupLasso': [],
                'RPA': [],
                'EU': []
    }

    for re in range(1,len(y_test)+1):
        if np.mod(re,10)==1 : print('Running testing trial={:1.0f}'.format(re))

        #testing trial
        X_test_trial=X_test_raw[(re-1):(re)]
        y_test_trial=y_test[(re-1):(re)]

        # Transform current test trial to CSP
        G_test=csp.transform(X_test_trial)

        # Update raw validation set
        X_val_raw=np.vstack((X_val_raw, X_test_trial))
        y_val=np.hstack((y_val, y_test_trial))

        # Add transformed trial to G_val
        G_val=np.vstack((G_val, G_test))
            
        # SC  
        yt_predict, time_sc = SC(G_test, clf_base)
        predictions['SC'].append(yt_predict)
        times['SC'].append(time_sc)

        
        # SR
        if np.mod(re,10)==1 :
            print(f"Running SR for trial {re}")
        yt_predict, time_sr = SR(X_target, y_target, n_calib + (re - 1), X_source, y_source, X_test_trial)
        predictions['SR'].append(yt_predict)
        times['SR'].append(time_sr)

        # Forward Sinkhorn Transport
        if np.mod(re,10)==1 :
            print(f"Running Forward Sinkhorn Transport for trial {re}")
        yt_predict, time_fs = Forward_Sinkhorn_Transport(G_subsamples["forward_sinkhorn"], reg_params["forward_sinkhorn"], G_source, y_source, G_val, G_test, clf_base, metric)
        predictions['Forward_Sinkhorn'].append(yt_predict)
        times['Forward_Sinkhorn'].append(time_fs)

        # Forward Group-Lasso Transport
        if np.mod(re,10)==1 :
            print(f"Running Forward Group-Lasso Transport for trial {re}")
        yt_predict, time_fg = Forward_GroupLasso_Transport(G_subsamples["forward_grouplasso"], y_subsamples["forward_grouplasso"], reg_params["forward_grouplasso"], G_source, y_source, G_val, G_test, clf_base, metric)
        predictions['Forward_GroupLasso'].append(yt_predict)
        times['Forward_GroupLasso'].append(time_fg)

        # Backward Sinkhorn Transport
        if np.mod(re,10)==1 :
            print(f"Running Backward Sinkhorn Transport for trial {re}")
        yt_predict, time_bs = Backward_Sinkhorn_Transport(G_subsamples["backward_sinkhorn"], reg_params["backward_sinkhorn"], G_val, G_test, clf_base, metric)
        predictions['Backward_Sinkhorn'].append(yt_predict)
        times['Backward_Sinkhorn'].append(time_bs)


        # Backward Group-Lasso Transport
        if np.mod(re,10)==1 :
            print(f"Running Backward Group-Lasso Transport for trial {re}")
        yt_predict, time_bg = Backward_GroupLasso_Transport(G_subsamples["backward_grouplasso"], reg_params["backward_grouplasso"], G_val, y_val, G_test, clf_base, metric)
        predictions['Backward_GroupLasso'].append(yt_predict)
        times['Backward_GroupLasso'].append(time_bg)

        # Riemann
        if np.mod(re,10)==1 :
            print(f"Running Riemann for trial {re}")
        yt_predict, time_rpa = RPA(X_source, X_val_raw, X_test_trial, y_source, y_val, y_test_trial, transductive=True)
        predictions['RPA'].append(yt_predict)
        times['RPA'].append(time_rpa)

        # Euclidean
        if np.mod(re,10)==1 :
            print(f"Running Euclidean for trial {re}")
        yt_predict, time_eu = EU(X_source,X_val_raw,X_test_trial,y_source,y_val,y_test_trial)
        predictions['EU'].append(yt_predict)
        times['EU'].append(time_eu)

        

    return predictions, times


def calculate_accuracies(predictions, Y_test, print_results=True):
    """
    Calculate accuracy metrics for each Transfer Learning method.
    
    Computes the accuracy for each method by comparing predictions against
    ground truth labels. Optionally prints results to console and returns
    a pandas DataFrame with the accuracy scores.
    
    Parameters
    ----------
    predictions : dict
        Dictionary with method names as keys and lists of predictions as values.
        Expected keys: 'SC', 'SR', 'Backward_GroupLasso', 'RPA', 'EU'.
    Y_test : ndarray
        Ground truth labels for test trials, shape (n_test_trials,).
    print_results : bool, optional
        If True, prints accuracy results to console. Default is True.
    
    Returns
    -------
    df_results : pd.DataFrame
        DataFrame containing accuracy scores for each method.
        Columns: ['Method', 'Accuracy', 'Correct', 'Total'].
 
    """

    
    results = []
    
    for method, preds in predictions.items():
        preds_array = np.array(preds).flatten()
        
        # Calculate accuracy
        correct = np.sum(preds_array == Y_test)
        total = len(Y_test)
        accuracy = (correct / total) * 100
        
        
        results.append({
            'Method': method,
            'Accuracy': accuracy,
            'Correct': correct,
            'Total': total
        })
        
        
        # if print_results:
        #     print(f"Method: {method} - Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    
    df_results = pd.DataFrame(results)
    
    # Sort by accuracy (descending)
    df_results = df_results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    if print_results:
        print("\n" + "="*50)
        print("Summary:")
        print("="*50)
        print(df_results.to_string(index=False))
    
    return df_results

