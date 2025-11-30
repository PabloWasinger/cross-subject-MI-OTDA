import numpy as np
import pandas as pd    
from transfer_learning import*
from validation import*

# rango_cl=[0.1, 1, 10]
# rango_e=[0.1, 1, 10] 
# metric = 'sqeuclidean'
# outerkfold = 20
# innerkfold = None
# M=20
# norm=None


def evaluate_tl_methods_samplewise(X_source, y_source, X_target, y_target, n_calib=20, verbose=True):
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
    reg_e_grid = [0.1, 1, 10]
    reg_cl_grid = [0.1, 1, 10]
    metric = 'sqeuclidean'
    outerkfold = 20
    innerkfold = 3 
    M=40 # Number of samples to transport
    norm=None

    # BOTDAGL optimal parameters
    G_subsample, y_subsample, reg_params = cv_grouplasso_backward(
        reg_e_grid, reg_cl_grid, G_source, y_source, G_val, y_val, clf_base,
        metric=metric, outerkfold=outerkfold, innerkfold=innerkfold,
        M=M, norm=norm, verbose=verbose
    )
    

    predictions ={
                'SC': [],
                'SR': [],
                'Backward_GroupLasso': [],
                'RPA': [],
                'EU': []
    }

    times ={
                'SC': [],
                'SR': [],
                'Backward_GroupLasso': [],
                'RPA': [],
                'EU': []
    }

    for re in range(1,len(y_test)+1):
        if np.mod(re,10)==0 : print('Running testing trial={:1.0f}'.format(re))

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
        yt_predict, time_sc = SC(G_test, y_test_trial, clf_base)
        predictions['SC'].append(yt_predict)
        times['SC'].append(time_sc)

        
        # SR
        yt_predict, time_sr = SR(X_target, y_target, re, X_source, y_source, X_test_trial)
        predictions['SR'].append(yt_predict)
        times['SR'].append(time_sr)

        # Backward Group-Lasso Transport
        yt_predict, time_bg = Backward_GroupLasso_Transport(G_subsample, reg_params, G_val, y_val, G_test, clf_base, metric)
        predictions['Backward_GroupLasso'].append(yt_predict)
        times['Backward_GroupLasso'].append(time_bg)

        # Riemann
        yt_predict, time_rpa = RPA(X_source, X_val_raw, X_test_trial, y_source, y_val, y_test_trial)
        predictions['RPA'].append(yt_predict)
        times['RPA'].append(time_rpa)

        # Euclidean
        yt_predict, time_eu = EU(X_source,X_val_raw,X_test_trial,y_source,y_val,y_test_trial)
        predictions['EU'].append(yt_predict)
        times['EU'].append(time_eu)

        times_trial = [time_sc, time_sr, time_rpa, time_eu, time_bg]
    
        if re == 1:
            times_se = times_trial
        else:
            times_se = np.vstack((times_se, times_trial))
            
        
        
    return predictions, times, times_se


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

