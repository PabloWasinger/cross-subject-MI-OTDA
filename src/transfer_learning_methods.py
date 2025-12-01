import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import ot
import scipy.io
import mne          
from mne.decoding import CSP
mne.set_log_level(verbose='warning') #to avoid info at terminal
import matplotlib.pyplot as pl
np.random.seed(100)

# from MIOTDAfunctions import*

# pyRiemann transfer learning
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.transfer import TLCenter, TLRotate, TLScale, encode_domains
import timeit

#ignore warning 
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


def SC(G_te, lda):
    """
    Source Classifier (SC) - No transfer learning approach.
        
    Applies a pre-trained Linear Discriminant Analysis (LDA) classifier directly 
    to the target domain test data without any adaptation.
        
    Parameters
    ----------
    G_te : ndarray
        Test features from the target domain after CSP transformation.
    lda : LinearDiscriminantAnalysis
        Pre-trained LDA classifier from the source domain.
        
    Returns
    -------
    yt_predict : ndarray
        Predicted labels for the target domain test data.
    time : float
        Execution time in seconds.
    """
    
    start = timeit.default_timer()
    
    yt_predict = lda.predict(G_te)
    
    stop = timeit.default_timer()
    time = stop - start
    
    return yt_predict, time 


def SR(Data_S2, Labels_S2, re, Xtr, Ytr, Xte):
    """
    Standard with Recalibration (SR).
    
    Implements the sliding window approach (Ang et al.):
    Maintains a fixed training size by replacing the oldest source samples 
    with the newest target samples.
    
    Parameters
    ----------
    Data_S2, Labels_S2 : ndarray
        Full available target data (Calibration + History).
    re : int
        Current test trial index relative to calibration end.
    Xtr, Ytr : ndarray
        Original Source data.
    Xte : ndarray
        Current target test trial (raw).
    """ 

    
    start = timeit.default_timer()
    
    #Get Data
    Xtr2add = Data_S2[0:20 +re - 1] # Corrected the leakage by subtracting 1
    Ytr2add = Labels_S2[0:20 +re - 1]
    
    Xtr2 = np.vstack(((Xtr, Xtr2add)))
    Ytr2 = np.hstack(((Ytr, Ytr2add)))
        
    Ytr2 = Ytr2[len(Ytr2add):]
    Xtr2 = Xtr2[len(Ytr2add):]

    # Create a new CSP
    csp = CSP(n_components=6, reg='empirical', log=True, norm_trace=False, cov_est='epoch')
    
    #learn new csp filters
    Gtr = csp.fit_transform(Xtr2,Ytr2)
    
    #learn new lda
    lda = LinearDiscriminantAnalysis()
    lda.fit(Gtr,Ytr2)

    # Apply on new test data
    Gte = csp.transform(Xte)
    #ldatest
    yt_predict = lda.predict(Gte)
    
    # time
    stop = timeit.default_timer()
    time = stop - start
    
    return yt_predict, time 


def Forward_Sinkhorn_Transport(G_subsample, regulizers, G_source, Y_source, G_val, G_te, clf, metric):
    """
    Sinkhorn Optimal Transport - Forward adaptation of source features.
        
    Learns an optimal transport mapping from source to target domain using 
    Sinkhorn regularization, then applies this mapping to transform source 
    features before retraining the classifier.
        
    Parameters
    ----------
    G_subsample : ndarray
        Source domain features for learning the transport map.
    regulizers : tuple of float
        Regularization parameter (entropy) for Sinkhorn algorithm.
    G_source : ndarray
        Source domain training features to be transported.
    Y_source : ndarray
        Source domain training labels.
    G_val : ndarray
        Target domain validation features (unlabeled, used as transport target).
    G_te : ndarray
        Target domain test features.
    clf : classifier object
        Classifier (e.g., LDA) to be trained on transported features.
    metric : str
        Distance metric for optimal transport (e.g., 'sqeuclidean').
        
    Returns
    -------
    yt_predict : ndarray
        Predicted labels for the target domain test data.
    time : float
        Execution time in seconds.
    """


    #time
    start = timeit.default_timer()
        
    otda = ot.da.SinkhornTransport(metric=metric, reg_e=regulizers)
    #learn the map
    otda.fit(Xs=G_subsample, Xt=G_val)
    
    #apply the mapping over source data
    transp_Xs = otda.transform(Xs=G_source)

    # train a new classifier bases upon the transform source data
    clf.fit(transp_Xs, Y_source)
    
    # Compute acc
    yt_predict = clf.predict(G_te)
    
    # time
    stop = timeit.default_timer()
    time = stop - start  
    
    return yt_predict, time
    

def Forward_GroupLasso_Transport(G_subsample, Y_subsample, regulizers, G_source, Y_source, G_val, G_te, clf, metric):
    """
    Group Lasso Optimal Transport - Forward adaptation with class regularization.
        
    Similar to Sinkhorn transport but uses additional group lasso regularization 
    to encourage class-wise transport structure, promoting within-class alignment.
        
    Parameters
    ----------
    G_subsample : ndarray
        Source domain features for learning the transport map.
    Y_subsample : ndarray
        Source domain labels for learning the transport map.
    regulizers : tuple of float
        Regularization parameters: (entropy regularization, class regularization).
    G_source : ndarray
        Source domain training features to be transported.
    Y_source : ndarray
        Source domain training labels.
    G_val : ndarray
        Target domain validation features (unlabeled, used as transport target).
    G_te : ndarray
        Target domain test features.
    clf : classifier object
        Classifier (e.g., LDA) to be trained on transported features.
    metric : str
        Distance metric for optimal transport (e.g., 'sqeuclidean').
        
    Returns
    -------
    yt_predict : ndarray
        Predicted labels for the target domain test data.
    time : float
        Execution time in seconds.
    """
    #time
    start = timeit.default_timer()
    
        
    otda = ot.da.SinkhornL1l2Transport(metric = metric, reg_e = regulizers[0], reg_cl = regulizers[1])
    otda.fit(Xs=G_subsample, ys=Y_subsample, Xt=G_val)

    #transport taget samples onto source samples
    transp_Xs = otda.transform(Xs=G_source)

    # train a new classifier bases upon the transform source data
    clf.fit(transp_Xs, Y_source)

    # Compute acc
    yt_predict = clf.predict(G_te)   
    # time
    stop = timeit.default_timer()
    time = stop - start 
        
    
    return yt_predict, time

def Backward_Sinkhorn_Transport(G_source, regulizers, G_val, G_te, lda, metric):
    """ 
    Backward Sinkhorn Transport - Reverse adaptation of test features.
        
    Learns a reverse transport map from target to source domain, then transports 
    test features back to the source domain where the original classifier was trained.
    No classifier retraining is performed.
        
    Parameters
    ----------
    Gtr_daot : ndarray
        Source domain features (used as transport target).
    regulizers : tuple of float
        Regularization parameters: (entropy regularization).
    G_val : ndarray
        Target domain validation features (source for reverse transport).
    G_te : ndarray
        Target domain test features to be transported back.
    lda : LinearDiscriminantAnalysis
        Pre-trained LDA classifier from the source domain.
    metric : str
        Distance metric for optimal transport (e.g., 'sqeuclidean').
        
    Returns
    -------
    yt_predict : ndarray
        Predicted labels for the target domain test data.
    time : float
        Execution time in seconds.
    """
    # time
    start = timeit.default_timer()
      
    # Transport plan
    botda = ot.da.SinkhornTransport(metric=metric, reg_e=regulizers)
    botda.fit(Xs=G_val, Xt=G_source)
    
    #transport testing samples
    transp_Xt_backward = botda.transform(Xs=G_te)
    
    # Compute accuracy without retraining    
    yt_predict = lda.predict(transp_Xt_backward)
    
    # time
    stop = timeit.default_timer()
    time = stop - start
    
    return yt_predict, time


def Backward_GroupLasso_Transport(G_source, regulizers, G_val, Y_val, G_te, lda, metric):
    """
    Backward Group Lasso Transport - Reverse adaptation with class regularization.
        
    Reverse transport from target to source domain using group lasso regularization 
    for class-wise structure. Transports test features to source domain without 
    retraining the classifier.
        
    Parameters
    ----------
    G_source : ndarray
        Source domain features (used as transport target).
    regulizers : tuple of float
        Regularization parameters: (entropy regularization, class regularization).
    G_val : ndarray
        Target domain validation features (source for reverse transport).
    Y_val : ndarray
        Target domain validation labels.
    G_te : ndarray
        Target domain test features to be transported back.
    lda : LinearDiscriminantAnalysis
        Pre-trained LDA classifier from the source domain.
    metric : str
        Distance metric for optimal transport (e.g., 'sqeuclidean').
        
    Returns
    -------
    yt_predict : ndarray
        Predicted labels for the target domain test data.
    time : float
        Execution time in seconds.
    """

    #time
    start = timeit.default_timer()
      
    botda = ot.da.SinkhornL1l2Transport(metric=metric, reg_e=regulizers[0], reg_cl=regulizers[1])
    botda.fit(Xs=G_val, ys=Y_val, Xt=G_source)
    
    #transport testing samples
    transp_Xt_backward=botda.transform(Xs=G_te)
    
    # Compute accuracy without retraining    
    yt_predict = lda.predict(transp_Xt_backward)
    
    # time
    stop = timeit.default_timer()
    time = stop - start
    
    
    return yt_predict, time


def RPA(Xtr, Xval, Xte, Ytr, Yval, Yte):
    """
    Riemannian Procrustes Analysis (RPA) - Manifold-based domain adaptation.

    Performs domain adaptation on covariance matrices by sequential geometric
    transformations: recentering, scaling, and rotation on the Riemannian
    manifold of SPD matrices. Combines adapted source and target data for training.

    Parameters
    ----------
    Xtr : ndarray
        Source domain training data (epochs × channels × time).
    Xval : ndarray
        Target domain validation data (epochs × channels × time).
    Xte : ndarray
        Target domain test data (epochs × channels × time).
    Ytr : ndarray
        Source domain training labels.
    Yval : ndarray
        Target domain validation labels.
    Yte : ndarray
        Target domain test labels (not used in prediction).

    Returns
    -------
    yt_predict : ndarray
        Predicted labels for the target domain test data.
    time : float
        Execution time in seconds.
    """

    # time
    start = timeit.default_timer()

    # Estimate covariance matrices
    cov_tr = Covariances().transform(Xtr)
    cov_val = Covariances().transform(Xval)
    cov_te = Covariances().transform(Xte)

    # Combine all covariance matrices for transformation
    X_combined = np.concatenate([cov_tr, cov_val, cov_te])

    # Encode domains using extended labels
    # Domain encoding: 0 for source (train), 1 for target_train (val), 2 for target_test
    n_tr, n_val, n_te = len(cov_tr), len(cov_val), len(cov_te)

    # Create extended labels: [label, domain]
    y_tr_enc = encode_domains(X_combined[:n_tr], Ytr, domain='source')
    y_val_enc = encode_domains(X_combined[n_tr:n_tr+n_val], Yval, domain='target')
    y_te_enc = encode_domains(X_combined[n_tr+n_val:], Yte, domain='target')

    # Combine all data
    X_all = X_combined
    y_all_enc = np.concatenate([y_tr_enc, y_val_enc, y_te_enc])

    # Apply RPA transformations in sequence
    # 1. Recenter
    rct = TLCenter(target_domain='target')
    X_rct = rct.fit_transform(X_all, y_all_enc)

    # 2. Scale (stretch)
    scl = TLScale(target_domain='target')
    X_scl = scl.fit_transform(X_rct, y_all_enc)

    # 3. Rotate
    rot = TLRotate(target_domain='target', metric='riemann')
    X_rpa = rot.fit_transform(X_scl, y_all_enc)

    # Split back into source, target_train, target_test
    covs_source = X_rpa[:n_tr]
    covs_target_train = X_rpa[n_tr:n_tr+n_val]
    covs_target_test = X_rpa[n_tr+n_val:]

    # Combine source and target training data
    covs_train = np.concatenate([covs_source, covs_target_train])
    y_train = np.concatenate([Ytr, Yval])

    # Train MDM classifier
    clf = MDM()
    clf.fit(covs_train, y_train)

    # Predict on test data
    yt_predict = clf.predict(covs_target_test)

    # time
    stop = timeit.default_timer()
    time = stop - start

    return yt_predict, time

def EU(Xtr,Xval,Xte,Ytr,Yval,Yte):

    """
    Euclidean Alignment (EU) - Covariance-based domain normalization.
        
    Normalizes data from both domains by whitening with respect to their mean 
    covariance matrices, bringing them to a common Euclidean space. Combines 
    aligned source and target data to retrain CSP+LDA.
        
    Parameters
    ----------
    Xtr : ndarray
        Source domain training data (epochs × channels × time).
    Xval : ndarray
        Target domain validation data (epochs × channels × time).
    Xte : ndarray
        Target domain test data (epochs × channels × time).
    Ytr : ndarray
        Source domain training labels.
    Yval : ndarray
        Target domain validation labels.
    Yte : ndarray
        Target domain test labels (not used in prediction).
        
    Returns
    -------
    yt_predict : ndarray
        Predicted labels for the target domain test data.
    time : float
        Execution time in seconds.
    """
    
    # time
    start = timeit.default_timer()
    # Estimate single trial covariance
    cov_tr = Covariances().transform(Xtr)
    cov_val= Covariances().transform(Xval)
    
    Ctr = cov_tr.mean(0)
    Cval = cov_val.mean(0)
    
    # aligment
    Xtr_eu = np.asarray([np.dot(invsqrtm(Ctr), epoch) for epoch in Xtr])
    Xval_eu = np.asarray([np.dot(invsqrtm(Cval), epoch) for epoch in Xval])
    Xte_eu = np.asarray([np.dot(invsqrtm(Cval), epoch) for epoch in Xte])

    # append train and validation data
    x_train = np.concatenate([Xtr_eu, Xval_eu])
    y_train = np.concatenate([Ytr, Yval])

    # train new csp+lda
    csp = CSP(n_components=6, reg='empirical', log=True, norm_trace=False, cov_est='epoch')
    # learn csp filters
    Gtr = csp.fit_transform(x_train,y_train)
    
    # learn lda
    lda = LinearDiscriminantAnalysis()
    lda.fit(Gtr,y_train)
    
    # test
    Gte = csp.transform(Xte_eu)  
    # acc
    yt_predict = lda.predict(Gte)
    
    # time
    stop = timeit.default_timer()
    time = stop - start
        
    return yt_predict, time