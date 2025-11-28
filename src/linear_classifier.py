from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP


def linear_classifier(Xtr, Ytr):

    csp = CSP(n_components=6, reg='empirical', log=True, norm_trace=False, cov_est='epoch')
    #learn csp filters
    Gtr = csp.fit_transform(Xtr, Ytr)
    #learn lda
    lda = LinearDiscriminantAnalysis()
    lda.fit(Gtr,Ytr)
    

    return lda, Gtr, csp