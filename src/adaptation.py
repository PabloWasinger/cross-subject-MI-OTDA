import numpy as np
import pandas as pd    
from transfer_learning import*


# rango_cl=[0.1, 1, 10]
# rango_e=[0.1, 1, 10] 
# metric = 'sqeuclidean'
# outerkfold = 20
# innerkfold = None
# M=20
# norm=None

def set_validation(Labels_S2, Data_S2, csp):
    Labels_te=Labels_S2[20:]
    ##
    Xval=Data_S2[0:20]
    Yval=Labels_S2[0:20]
    ##
    Gval=csp.transform(Xval)
    return Gval, Yval, Labels_te



def evaluate_tl_methods(X_target, Y_target, Xt_source, Yt_source, csp, model, 
    G_BOTDA, Y_BOTDA, regu_BOTDA, metric, transform='True'):

    """
    Evaluate multiple Transfer Learning methods on target domain data.

    Performs incremental trial-by-trial evaluation of five transfer learning 
    methods: SC (Source Classifier), SR (Source Retraining), Backward Group-Lasso 
    Transport, RPA (Riemannian Procrustes Analysis), and EU (Euclidean Alignment). 
    The first 20 trials are used for validation and the rest for sequential testing.

    Parameters
    ----------
    X_target : ndarray
        Target domain input data, shape (n_trials, n_channels, n_samples).
    Y_target : ndarray
        Target domain labels, shape (n_trials,).
    Xt_source : ndarray
        Source domain input data, shape (n_trials_source, n_channels, n_samples).
    Yt_source : ndarray
        Source domain labels, shape (n_trials_source,).
    csp : object
        Pre-trained CSP (Common Spatial Patterns) object with transform() method.
    model : object
        Classification model with fit() and predict() methods.
    G_BOTDA : ndarray
        Transformed features for Backward Optimal Transport.
    Y_BOTDA : ndarray
        Labels corresponding to G_BOTDA.
    regu_BOTDA : float
        Regularization parameter for Backward Group-Lasso Transport.
    metric : str or callable
        Distance metric for transport computation.
    transform : str or bool, optional
        Whether to apply CSP transformation. Default is 'True'.


    Returns
    -------
    predictions : dict
        Dictionary with method names as keys and lists of predictions as values.
        Keys: 'SC', 'SR', 'Backward_GroupLasso', 'RPA', 'EU'.
    times : dict
        Dictionary with method names as keys and lists of execution times as values.
        Keys: 'SC', 'SR', 'Backward_GroupLasso', 'RPA', 'EU'.
    times_se : ndarray
        Matrix of execution times for each trial and method, shape (n_test_trials, 5).
        Columns: [time_sc, time_sr, time_rpa, time_eu, time_bg].
        Each row corresponds to one test trial's timing measurements.

    """

    Xval = X_target[0:20]
    Yval = Y_target[0:20]
    X_test = X_target[20:]
    Y_test = Y_target[20:]
    num_trials = len(Y_test)
    if transform: Gtr = csp.transform(Xt_source)
    else: Gtr=Xt_source

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

    for re in range(1,num_trials+1):
        if np.mod(re,10)==0 : print('Running testing trial={:1.0f}'.format(re))
    
        #testing trial
        Xte=X_test[(re-1):(re)]
        Yte=Y_test[(re-1):(re)]
        
        Xval=np.vstack((Xval, Xte))
        Yval=np.hstack((Yval, Yte))

        if transform:
            #csp estimation
            Gval=csp.transform(Xval)
            Gte=csp.transform(Xte)
            
        else: 
            Gval=Xval
            Gte=Xte
            
            
        # SC  
        yt_predict, time_sc = SC(Gte, Yte, model)
        predictions['SC'].append(yt_predict)
        times['SC'].append(time_sc)

        
        # SR
        yt_predict, time_sr = SR(X_target, Y_target, re, Xt_source, Yt_source, Xte, Yte)
        predictions['SR'].append(yt_predict)
        times['SR'].append(time_sr)

        # Backward Group-Lasso Transport
        yt_predict, time_bg = Backward_GroupLasso_Transport(G_BOTDA, Y_BOTDA, regu_BOTDA, Gtr, Yt_source, Gval, Gte, model, metric)
        predictions['Backward_GroupLasso'].append(yt_predict)
        times['Backward_GroupLasso'].append(time_bg)

        # Riemann
        yt_predict, time_rpa = RPA(Xt_source,Xval,Xte,Yt_source,Yval,Yte)
        predictions['RPA'].append(yt_predict)
        times['RPA'].append(time_rpa)

        # Euclidean
        yt_predict, time_eu = EU(Xt_source,Xval,Xte,Yt_source,Yval,Yte)
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

