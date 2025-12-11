"""
Comprehensive subject pair selection metrics for cross-subject transfer learning.

This module implements multiple criteria to select which subject pairs 
are most suitable for transfer learning evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from sklearn.preprocessing import MinMaxScaler

from structuredata import load_session_binary_mi



EEG_CHANNELS = [
    'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
    'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9',
    'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz',
    'EEG-15', 'EEG-16'
]
EOG_CHANNELS = ['EOG-left', 'EOG-central', 'EOG-right']


def load_subject_data(subject_id, session='T', data_dir='data'):
    """
    Load data for a single subject with bandpass filter (8-30 Hz).
    
    Use for CSP+LDA methods that require mu/beta band filtering.
    """
    DATA_DIR = Path(data_dir)
    GDF_DIR = DATA_DIR / "gdf"
    LABELS_DIR = DATA_DIR / "labels"
    
    gdf_file = GDF_DIR / f"A{subject_id:02d}{session}.gdf"
    labels_file = LABELS_DIR / f"A{subject_id:02d}{session}.mat"
    
    X, y, _ = load_session_binary_mi(
        str(gdf_file), EEG_CHANNELS, EOG_CHANNELS,
        tmin=0.5, tmax=2.5, l_freq=8, h_freq=30,
        reject_artifacts=True, true_labels_path=str(labels_file),
        verbose=False
    )
    
    return X, y


def load_subject_data_raw(subject_id, session='T', data_dir='data'):
    """
    Load data for a single subject WITHOUT bandpass filter.
    
    Use for EEGNet methods - the network learns its own temporal filters.
    Only applies minimal filtering (0.5-100 Hz) for DC removal and anti-aliasing.
    """
    DATA_DIR = Path(data_dir)
    GDF_DIR = DATA_DIR / "gdf"
    LABELS_DIR = DATA_DIR / "labels"
    
    gdf_file = GDF_DIR / f"A{subject_id:02d}{session}.gdf"
    labels_file = LABELS_DIR / f"A{subject_id:02d}{session}.mat"
    
    X, y, _ = load_session_binary_mi(
        str(gdf_file), EEG_CHANNELS, EOG_CHANNELS,
        tmin=0.5, tmax=2.5, l_freq=0.5, h_freq=100, 
        reject_artifacts=True, true_labels_path=str(labels_file),
        verbose=False
    )
    
    return X, y



def calculate_csp_eigenvalue_ratio(subject_id, session='T', n_components=6):
    """
    Calculate CSP eigenvalue ratio as a measure of class separability.

    Computes the ratio between largest and smallest CSP eigenvalues,
    indicating how well the spatial filters separate the two classes.

    Parameters
    ----------
    subject_id : int
        Subject identifier.
    session : str, default='T'
        Session to analyze.
    n_components : int, default=6
        Number of CSP components (must be even).

    Returns
    -------
    dict
        Dictionary containing:
        - 'ratio_max': Maximum eigenvalue ratio
        - 'ratio_mean': Mean ratio across component pairs
        - 'eigenvalues': Array of all eigenvalues
        - 'separability_score': Normalized score (0-100)
    """
    from scipy.linalg import eigh

    X, y = load_subject_data(subject_id, session)

    X_0 = X[y == 0]
    X_1 = X[y == 1]

    
    cov_0 = np.mean([np.cov(trial) for trial in X_0], axis=0)
    cov_1 = np.mean([np.cov(trial) for trial in X_1], axis=0)

    eigenvalues, _ = eigh(cov_0, cov_0 + cov_1)


    eigenvalues = np.sort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[:n_components]

    ratio_max = eigenvalues[0] / eigenvalues[-1]

    n_pairs = n_components // 2
    ratios = []
    for i in range(n_pairs):
        ratio = eigenvalues[i] / eigenvalues[-(i+1)]
        ratios.append(ratio)

    ratio_mean = np.mean(ratios)


    separability_score = np.clip(20 * np.log10(ratio_mean), 0, 100)

    return {
        'ratio_max': ratio_max,
        'ratio_mean': ratio_mean,
        'eigenvalues': eigenvalues,
        'separability_score': separability_score
    }



def calculate_riemannian_centrality(all_subjects, session='T'):
    """
    Calculate Riemannian distance of each subject to the population mean.
    
    Measures how representative each subject is of the overall population
    by computing the Riemannian distance from their mean covariance matrix
    to the grand mean covariance across all subjects.

    Parameters
    ----------
    all_subjects : list of int
        List of subject identifiers to analyze.
    session : str, default='T'
        Session to analyze ('T' for training, 'E' for evaluation).
    """
    subject_means = {}
    valid_subjects = []
    
    print("Calculando centroides Riemannianos...")
    
    for subj in all_subjects:
        try:
        
            X, _ = load_subject_data(subj, session)
            
            cov_estimator = Covariances(estimator='scm') 
            covs = cov_estimator.fit_transform(X)
            
            mean_cov = mean_riemann(covs)
            
            subject_means[subj] = mean_cov
            valid_subjects.append(subj)
        except Exception as e:
            print(f"  [WARN] Omitiendo Sujeto {subj}: {e}")
            



    population_covs = np.array([subject_means[s] for s in valid_subjects])
    grand_mean = mean_riemann(population_covs)
    
    distances = {}
    for subj in valid_subjects:
        dist = distance_riemann(subject_means[subj], grand_mean)
        distances[subj] = dist
        
    return distances


def select_best_source_subject(all_subjects, session='T', w_csp=0.4, w_riemann=0.6):
    """
    Select the best source subject for transfer learning using a weighted score.
    
    Combines two complementary criteria:
    1. Signal quality (CSP eigenvalue ratio) - higher is better
    2. Population representativeness (Riemannian centrality) - lower distance is better
    
    The final score balances individual signal quality with how well the subject
    represents the population, making them an ideal source for transfer learning.

    Parameters
    ----------
    all_subjects : list of int
        List of all subject identifiers to consider.
    session : str, default='T'
        Session to analyze ('T' for training, 'E' for evaluation).
    w_csp : float, default=0.4
        Weight for signal quality score (0-1).
    w_riemann : float, default=0.6
        Weight for centrality score (0-1).

    Returns
    -------
    best_subject : int
        Subject ID with the highest combined score.
    ranking_df : pandas.DataFrame
        Ranked table of all subjects with columns:
        
        - 'Subject' : int
            Subject identifier.
        - 'CSP_Raw' : float
            Raw CSP eigenvalue ratio.
        - 'Riemann_Dist' : float
            Riemannian distance to population mean.
        - 'Score_Quality' : float
            Normalized signal quality score (0-1).
        - 'Score_Centrality' : float
            Normalized centrality score (0-1), inverted distance.
        - 'Final_Score' : float
            Weighted combination of quality and centrality (0-1).
            
    """
    
    
    riemann_dists = calculate_riemannian_centrality(all_subjects, session)
    
    results = []
    
    print("Calculando ratios CSP y combinando scores...")
    
    for subj in riemann_dists.keys():
        
        csp_metrics = calculate_csp_eigenvalue_ratio(subj, session)
        

        csp_score_raw = csp_metrics['ratio_mean']
        
        results.append({
            'Subject': subj,
            'CSP_Raw': csp_score_raw,              
            'Riemann_Dist': riemann_dists[subj]    
        })
    
    df = pd.DataFrame(results)
    scaler = MinMaxScaler()
    
    
    
    df['Score_Quality'] = scaler.fit_transform(df[['CSP_Raw']])
    
    dist_norm = scaler.fit_transform(df[['Riemann_Dist']])
    df['Score_Centrality'] = 1 - dist_norm 
    
    
    df['Final_Score'] = (w_csp * df['Score_Quality']) + (w_riemann * df['Score_Centrality'])
    
    
    df_ranked = df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    
    best_subject = int(df_ranked.iloc[0]['Subject'])
    
    print(f"\nSujeto ganador: S{best_subject:02d} (Score: {df_ranked.iloc[0]['Final_Score']:.3f})")
    
    return best_subject, df_ranked





