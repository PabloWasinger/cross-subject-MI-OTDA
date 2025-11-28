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

def SR(Data_S2, Labels_S2, re, Xtr, Ytr, Xte, Yte):
    """
    Supervised Recalibration (SR) - Transfer learning with labeled target data.
        
    Retrains CSP filters and LDA classifier using only recent labeled samples 
    from the target domain, discarding source domain data.
        
    Parameters
    ----------
    Data_S2 : ndarray
        Available labeled data from the target domain (epochs).
    Labels_S2 : ndarray
        Corresponding labels for Data_S2.
    re : int
        Additional samples to use beyond the initial 20 samples.
    Xtr : ndarray
        Source domain training data (used temporarily then discarded).
    Ytr : ndarray
        Source domain training labels (used temporarily then discarded).
    Xte : ndarray
        Target domain test data.
    Yte : ndarray
        Target domain test labels (not used in prediction).
        
    Returns
    -------
    yt_predict : ndarray
        Predicted labels for the target domain test data.
    time : float
        Execution time in seconds.
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

def Sinkhorn_Transport(Gtr_daot, Ytr_daot, regu_, Gtr, Ytr, Gval, Gte, clf, metric):
    """
    Sinkhorn Optimal Transport - Forward adaptation of source features.
        
    Learns an optimal transport mapping from source to target domain using 
    Sinkhorn regularization, then applies this mapping to transform source 
    features before retraining the classifier.
        
    Parameters
    ----------
    Gtr_daot : ndarray
        Source domain features for learning the transport map.
    Ytr_daot : ndarray
        Source domain labels for learning the transport map.
    regu_ : float
        Regularization parameter (entropy) for Sinkhorn algorithm.
    Gtr : ndarray
        Source domain training features to be transported.
    Ytr : ndarray
        Source domain training labels.
    Gval : ndarray
        Target domain validation features (unlabeled, used as transport target).
    Gte : ndarray
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
        
    otda = ot.da.SinkhornTransport(metric=metric, reg_e=regu_)
    #learn the map
    otda.fit(Xs=Gtr_daot, ys=Ytr_daot, Xt=Gval)
    
    #apply the mapping over source data
    transp_Xs = otda.transform(Xs=Gtr)

    # train a new classifier bases upon the transform source data
    clf.fit(transp_Xs, Ytr)
    
    # Compute acc
    yt_predict = clf.predict(Gte)
    
    # time
    stop = timeit.default_timer()
    time = stop - start  
    
    return yt_predict, time\
    
def GroupLasso_Transport(Gtr_daot, Ytr_daot, regu_, Gtr, Ytr, Gval, Gte, clf, metric):
    """
    Group Lasso Optimal Transport - Forward adaptation with class regularization.
        
    Similar to Sinkhorn transport but uses additional group lasso regularization 
    to encourage class-wise transport structure, promoting within-class alignment.
        
    Parameters
    ----------
    Gtr_daot : ndarray
        Source domain features for learning the transport map.
    Ytr_daot : ndarray
        Source domain labels for learning the transport map.
    regu_ : tuple of float
        Regularization parameters: (entropy regularization, class regularization).
    Gtr : ndarray
        Source domain training features to be transported.
    Ytr : ndarray
        Source domain training labels.
    Gval : ndarray
        Target domain validation features (unlabeled, used as transport target).
    Gte : ndarray
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
    
        
    otda = ot.da.SinkhornL1l2Transport(metric = metric, reg_e = regu_[0], reg_cl = regu_[1])
    otda.fit(Xs=Gtr_daot, ys=Ytr_daot, Xt=Gval)

    #transport taget samples onto source samples
    transp_Xs = otda.transform(Xs=Gtr)

    # train a new classifier bases upon the transform source data
    clf.fit(transp_Xs, Ytr)

    # Compute acc
    yt_predict = clf.predict(Gte)   
    # time
    stop = timeit.default_timer()
    time = stop - start 
        
    
    return yt_predict, time

def Backward_Sinkhorn_Transport(Gtr_daot, Ytr_daot, regu_, Gtr, Ytr, Gval, Yval, Gte, lda, metric):
    """
    Backward Sinkhorn Transport - Reverse adaptation of test features.
        
    Learns a reverse transport map from target to source domain, then transports 
    test features back to the source domain where the original classifier was trained.
    No classifier retraining is performed.
        
    Parameters
    ----------
    Gtr_daot : ndarray
        Source domain features (used as transport target).
    Ytr_daot : ndarray
        Source domain labels (not used in transport).
    regu_ : float
        Regularization parameter (entropy) for Sinkhorn algorithm.
    Gtr : ndarray
        Source domain training features (not used directly).
    Ytr : ndarray
        Source domain training labels (not used directly).
    Gval : ndarray
        Target domain validation features (source for reverse transport).
    Yval : ndarray
        Target domain validation labels.
    Gte : ndarray
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
    botda = ot.da.SinkhornTransport(metric=metric, reg_e=regu_)
    botda.fit(Xs=Gval, ys=Yval, Xt=Gtr_daot)
    
    #transport testing samples
    transp_Xt_backward = botda.transform(Xs=Gte)
    
    # Compute accuracy without retraining    
    yt_predict = lda.predict(transp_Xt_backward)
    
    # time
    stop = timeit.default_timer()
    time = stop - start
    
    return yt_predict, time

def Backward_GroupLasso_Transport(Gtr_daot, Ytr_daot, regu_, Gtr, Ytr, Gval, Yval, Gte, lda, metric):
    """
    Backward Group Lasso Transport - Reverse adaptation with class regularization.
        
    Reverse transport from target to source domain using group lasso regularization 
    for class-wise structure. Transports test features to source domain without 
    retraining the classifier.
        
    Parameters
    ----------
    Gtr_daot : ndarray
        Source domain features (used as transport target).
    Ytr_daot : ndarray
        Source domain labels (not used in transport).
    regu_ : tuple of float
        Regularization parameters: (entropy regularization, class regularization).
    Gtr : ndarray
        Source domain training features (not used directly).
    Ytr : ndarray
        Source domain training labels (not used directly).
    Gval : ndarray
        Target domain validation features (source for reverse transport).
    Yval : ndarray
        Target domain validation labels.
    Gte : ndarray
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
      
    botda = ot.da.SinkhornL1l2Transport(metric=metric, reg_e=regu_[0], reg_cl=regu_[1])
    botda.fit(Xs=Gval, ys=Yval, Xt=Gtr_daot)
    
    #transport testing samples
    transp_Xt_backward=botda.transform(Xs=Gte)
    
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