import numpy as np
from transfer_learning import*
from vicky.MIOTDAfunctions import*

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


def optimize_ot_parameters(Gtr, Ytr, Gval, Yval, rango_e, rango_cl, lda, metric, outerkfold, innerkfold, M, norm):
    """
    Optimize hyperparameters and select training subsets for Optimal Transport methods.
    
    Performs nested cross-validation to find optimal regularization parameters 
    and source domain subsets for four Optimal Transport-based domain adaptation 
    methods: Sinkhorn Transport, Group-Lasso Transport, and their backward variants.
    This preprocessing step is performed once before incremental testing to ensure
    computational efficiency.
    
    Parameters
    ----------
    Gtr : ndarray
        Source domain features after CSP transformation (n_samples × n_features).
    Ytr : ndarray
        Source domain labels (n_samples,).
    Gval : ndarray
        Target domain validation features after CSP transformation.
    Yval : ndarray
        Target domain validation labels.
    rango_e : array-like
        Range of entropy regularization values to test (e.g., [0.01, 0.1, 1, 10]).
    rango_cl : array-like
        Range of class regularization values to test for Group-Lasso methods.
    lda : LinearDiscriminantAnalysis
        Pre-trained LDA classifier from source domain (used for backward methods).
    metric : str
        Distance metric for optimal transport (e.g., 'sqeuclidean').
    outerkfold : int
        Number of folds for outer cross-validation loop.
    innerkfold : int
        Number of folds for inner (nested) cross-validation loop.
    M : int
        Number of source samples to select for the subset.
    norm : str or None
        Type of normalization to apply to the data.
    
    Returns
    -------
    results : dict
        Nested dictionary containing optimized configurations for each method:
        {
            'Sinkhorn': {
                'G': ndarray - Selected source features subset,
                'Y': ndarray - Corresponding labels,
                'regu': float - Optimal entropy regularization parameter
            },
            'GroupLasso': {
                'G': ndarray - Selected source features subset,
                'Y': ndarray - Corresponding labels,
                'regu': tuple - (entropy_reg, class_reg) optimal parameters
            },
            'Backward_Sinkhorn': {
                'G': ndarray - Selected source features subset,
                'Y': ndarray - Corresponding labels,
                'regu': float - Optimal entropy regularization parameter
            },
            'Backward_GroupLasso': {
                'G': ndarray - Selected source features subset,
                'Y': ndarray - Corresponding labels,
                'regu': tuple - (entropy_reg, class_reg) optimal parameters
            }
        }
    
    Notes
    -----
    - Forward methods (Sinkhorn, GroupLasso) create a new LDA classifier internally
      and will retrain it after transport.
    - Backward methods use the pre-trained LDA classifier without retraining.
    - This function should be called once before the incremental testing loop to
      avoid redundant hyperparameter optimization.
    - The selected subsets and regularization parameters are then reused across
      all incremental test samples.
    
    """
    

    
    results = {
        'Sinkhorn' : {}, 
        'GroupLasso' : {},
        'Backward_Sinkhorn' : {}, 
        'Backward_GroupLasso' : {}
    }
    #for fotda, create a new classifier (clf)
    clf=LinearDiscriminantAnalysis()
    G_FOTDAs_, Y_FOTDAs_, regu_FOTDAs_=\
    SelectSubsetTraining_OTDAs(Gtr, Ytr, Gval, Yval, rango_e, clf, metric, outerkfold, innerkfold, M, norm)
    results['Sinkhorn']['G'], results['Sinkhorn']['Y'], results['Sinkhorn']['regu'] = G_FOTDAs_, Y_FOTDAs_, regu_FOTDAs_

    G_FOTDAl1l2_, Y_FOTDAl1l2_, regu_FOTDAl1l2_=\
        SelectSubsetTraining_OTDAl1l2(Gtr, Ytr, Gval, Yval, rango_e, rango_cl, clf, metric, outerkfold, innerkfold, M, norm)
    results['GroupLasso']['G'], results['GroupLasso']['Y'], results['GroupLasso']['regu'] = G_FOTDAl1l2_, Y_FOTDAl1l2_, regu_FOTDAl1l2_
    
    #for botda, use the already trained classifier (lda)
    G_BOTDAs_, Y_BOTDAs_, regu_BOTDAs_=\
    SelectSubsetTraining_BOTDAs(Gtr, Ytr, Gval, Yval, rango_e, lda, metric, outerkfold, innerkfold, M, norm)
    results['Backward_Sinkhorn']['G'], results['Backward_Sinkhorn']['Y'], results['Backward_Sinkhorn']['regu'] = G_BOTDAs_, Y_BOTDAs_, regu_BOTDAs_


    G_BOTDAl1l2_, Y_BOTDAl1l2_, regu_BOTDAl1l2_=\
    SelectSubsetTraining_BOTDAl1l2(Gtr, Ytr, Gval, Yval, rango_e, rango_cl, lda, metric, outerkfold, innerkfold, M, norm)
    results['Backward_GroupLasso']['G'], results['Backward_GroupLasso']['Y'], results['Backward_GroupLasso']['regu'] = G_BOTDAl1l2_, Y_BOTDAl1l2_, regu_BOTDAl1l2_

    return results



def evaluate_tl_methods(Data_S2, Labels_S2, Labels_te, Xtr, Ytr, Xval, Yval, csp, lda, Gtr, 
    G_FOTDAs_, Y_FOTDAs_, regu_FOTDAs_, G_FOTDAl1l2_, Y_FOTDAl1l2_, regu_FOTDAl1l2_, 
    G_BOTDAs_, Y_BOTDAs_, regu_BOTDAs_, G_BOTDAl1l2_, Y_BOTDAl1l2_, regu_BOTDAl1l2_, 
    metric, nc, ns, predictions):
    """
    Executes all transfer learning methods incrementally on test data.
    
    Iterates through test samples one by one, updating validation set and 
    computing predictions using multiple transfer learning approaches: 
    SC, SR, Sinkhorn Transport, Group-Lasso Transport, Backward variants, 
    RPA, and Euclidean Alignment.
    
    Parameters
    ----------
    Data_S2 : ndarray
        Target domain data (epochs × channels × time).
    Labels_S2 : ndarray
        Target domain labels.
    Labels_te : ndarray
        Test labels to determine iteration length.
    Xtr : ndarray
        Source domain training data.
    Ytr : ndarray
        Source domain training labels.
    Xval_init : ndarray
        Initial validation data from target domain.
    Yval_init : ndarray
        Initial validation labels from target domain.
    csp : CSP object
        Fitted Common Spatial Patterns transformer.
    lda : LinearDiscriminantAnalysis
        Pre-trained LDA classifier from source domain.
    Gtr : ndarray
        Source domain features after CSP transformation.
    G_FOTDAs_ : ndarray
        Source features for forward Sinkhorn OT.
    Y_FOTDAs_ : ndarray
        Source labels for forward Sinkhorn OT.
    regu_FOTDAs_ : float
        Regularization parameter for forward Sinkhorn.
    G_FOTDAl1l2_ : ndarray
        Source features for forward Group-Lasso OT.
    Y_FOTDAl1l2_ : ndarray
        Source labels for forward Group-Lasso OT.
    regu_FOTDAl1l2_ : tuple of float
        Regularization parameters for forward Group-Lasso.
    G_BOTDAs_ : ndarray
        Source features for backward Sinkhorn OT.
    Y_BOTDAs_ : ndarray
        Source labels for backward Sinkhorn OT.
    regu_BOTDAs_ : float
        Regularization parameter for backward Sinkhorn.
    G_BOTDAl1l2_ : ndarray
        Source features for backward Group-Lasso OT.
    Y_BOTDAl1l2_ : ndarray
        Source labels for backward Group-Lasso OT.
    regu_BOTDAl1l2_ : tuple of float
        Regularization parameters for backward Group-Lasso.
    metric : str
        Distance metric for optimal transport (e.g., 'sqeuclidean').
    nc : int
        Number of channels in EEG data.
    ns : int
        Number of samples (time points) in EEG data.

    Returns
    -------
    predictions : dict
        Dictionary containing predictions for each method:
        - 'SC': Source Classifier predictions
        - 'SR': Supervised Recalibration predictions
        - 'Sinkhorn': Forward Sinkhorn Transport predictions
        - 'GroupLasso': Forward Group-Lasso Transport predictions
        - 'Backward_Sinkhorn': Backward Sinkhorn Transport predictions
        - 'Backward_GroupLasso': Backward Group-Lasso Transport predictions
        - 'RPA': Riemannian Procrustes Analysis predictions
        - 'EU': Euclidean Alignment predictions
    times : ndarray
        Array of execution times (iterations × methods).
    """
    predictions ={
                'SC': [],
                'SR': [],
                'Sinkhorn': [],
                'GroupLasso': [],
                'Backward_Sinkhorn': [],
                'Backward_GroupLasso': [],
                'RPA': [],
                'EU': []
    }

    for re in range(1,len(Labels_te)+1):
        if np.mod(re,10)==0 : print('Running testing trial={:1.0f}'.format(re))
    
        #testing trial
        Xte=Data_S2[20+(re-1):20+(re)]
        Xte=Xte.reshape(1, nc, ns)
        Yte=Labels_S2[20+(re-1):20+(re)]
        
        Xval=np.vstack((Xval, Xte))
        Yval=np.hstack((Yval, Yte))

        
        #csp estimation
        Gval=csp.transform(Xval)
        Gte=csp.transform(Xte)
            
        # SC  
        yt_predict, time_sc = SC(Gte, Yte, lda)
        predictions['SC'].append(yt_predict)

        
        # SR
        yt_predict, time_sr = SR(Data_S2, Labels_S2, re, Xtr, Ytr, Xte, Yte)
        predictions['SR'].append(yt_predict)

        #%% # Sinkhorn Transport
        clf=LinearDiscriminantAnalysis()
        yt_predict, time_fs = Sinkhorn_Transport(G_FOTDAs_, Y_FOTDAs_, regu_FOTDAs_, Gtr, Ytr, Gval, Gte, clf, metric)
        predictions['Sinkhorn'].append(yt_predict)

        #%% # Group-Lasso Transport
        clf=LinearDiscriminantAnalysis()
        yt_predict, time_fg = GroupLasso_Transport(G_FOTDAl1l2_, Y_FOTDAl1l2_, regu_FOTDAl1l2_, Gtr, Ytr, Gval, Gte, clf, metric)
        predictions['GroupLasso'].append(yt_predict)

        #%% # Backward Sinkhorn Transport
        yt_predict, time_bs = Backward_Sinkhorn_Transport(G_BOTDAs_, Y_BOTDAs_, regu_BOTDAs_, Gtr, Ytr, Gval, Gte, lda, metric)
        predictions['Backward_Sinkhorn'].append(yt_predict)

        #%% # Backward Group-Lasso Transport
        yt_predict, time_bg = Backward_GroupLasso_Transport(G_BOTDAl1l2_, Y_BOTDAl1l2_, regu_BOTDAl1l2_, Gtr, Ytr, Gval, Gte, lda, metric)
        predictions['Backward_GroupLasso'].append(yt_predict)

        # Riemann
        yt_predict, time_rpa = RPA(Xtr,Xval,Xte,Ytr,Yval,Yte)
        predictions['RPA'].append(yt_predict)

        # Euclidean
        yt_predict, time_eu = EU(Xtr,Xval,Xte,Ytr,Yval,Yte)
        predictions['EU'].append(yt_predict)
        
        #save times
        times = [time_sr, time_rpa, time_eu, time_fs, time_fg, time_bs, time_bg]
            
        if re==1:
            times_se = times
        else:
            times_se = np.vstack((times_se, times))
        
        return predictions, times_se