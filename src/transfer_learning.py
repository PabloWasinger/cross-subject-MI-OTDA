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

# get the functions from RPA package
import rpa.transfer_learning as TL

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
import timeit

#ignore warning 
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


def SC(Gte, Yte, lda):
    """
    Source Classifier (SC) - No transfer learning approach.
        
    Applies a pre-trained Linear Discriminant Analysis (LDA) classifier directly 
    to the target domain test data without any adaptation.
        
    Parameters
    ----------
    Gte : ndarray
        Test features from the target domain after CSP transformation.
    Yte : ndarray
        Test labels from the target domain (not used in prediction).
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
    
    yt_predict = lda.predict(Gte)
    
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
    Xtr2add = Data_S2[0:20 +re]
    Ytr2add = Labels_S2[0:20 +re]
    
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
    transformations: recentering, stretching, and rotation on the Riemannian 
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
    # cov matrix estimation
    cov_tr = Covariances().transform(Xtr)
    cov_val= Covariances().transform(Xval)
    cov_te = Covariances().transform(Xte)
        
    clf = MDM()
    source={'covs':cov_tr, 'labels': Ytr}
    target_org_train={'covs':cov_val, 'labels': Yval}
    target_org_test={'covs':cov_te, 'labels': Yte}
    
    # re-centered matrices
    source_rct, target_rct_train, target_rct_test = TL.RPA_recenter(source, target_org_train, target_org_test)   
    # stretched the re-centered matrices
    source_rcs, target_rcs_train, target_rcs_test = TL.RPA_stretch(source_rct, target_rct_train, target_rct_test)
    # rotate the re-centered-stretched matrices using information from classes
    source_rpa, target_rpa_train, target_rpa_test = TL.RPA_rotate(source_rcs, target_rcs_train, target_rcs_test)
    
    # get data
    covs_source, y_source = source_rpa['covs'], source_rpa['labels']
    covs_target_train, y_target_train = target_rpa_train['covs'], target_rpa_train['labels']
    covs_target_test, y_target_test = target_rpa_test['covs'], target_rpa_test['labels']
    
    # append train and validation data
    covs_train = np.concatenate([covs_source, covs_target_train])
    y_train = np.concatenate([y_source, y_target_train])
    
    # train
    clf.fit(covs_train, y_train)
    
    # test
    covs_test = covs_target_test
    y_test = y_target_test
    yt_predict = clf.predict(covs_test)
    
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