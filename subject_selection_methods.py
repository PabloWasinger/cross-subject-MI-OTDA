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


def load_subject_data(subject_id, session='T', data_dir='data'):
    """Load data for a single subject and session."""
    DATA_DIR = Path(data_dir)
    GDF_DIR = DATA_DIR / "gdf"
    LABELS_DIR = DATA_DIR / "labels"
    
    EEG_CHANNELS = [
        'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
        'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9',
        'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz',
        'EEG-15', 'EEG-16'
    ]
    EOG_CHANNELS = ['EOG-left', 'EOG-central', 'EOG-right']
    
    gdf_file = GDF_DIR / f"A{subject_id:02d}{session}.gdf"
    labels_file = LABELS_DIR / f"A{subject_id:02d}{session}.mat"
    
    X, y, _ = load_session_binary_mi(
        str(gdf_file), EEG_CHANNELS, EOG_CHANNELS,
        tmin=0.5, tmax=2.5, l_freq=8, h_freq=30,
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

    # Separate data by class
    X_0 = X[y == 0]
    X_1 = X[y == 1]

    # Compute covariance matrices for each class
    cov_0 = np.mean([np.cov(trial) for trial in X_0], axis=0)
    cov_1 = np.mean([np.cov(trial) for trial in X_1], axis=0)

    # Solve generalized eigenvalue problem: cov_0 * v = lambda * (cov_0 + cov_1) * v
    # This gives the CSP eigenvalues
    eigenvalues, _ = eigh(cov_0, cov_0 + cov_1)

    # Sort eigenvalues in descending order (most discriminative first)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Select only n_components eigenvalues (matching CSP behavior)
    eigenvalues = eigenvalues[:n_components]

    # Ratio del par más discriminativo (primero y último)
    ratio_max = eigenvalues[0] / eigenvalues[-1]

    # Ratios de todos los pares
    n_pairs = n_components // 2
    ratios = []
    for i in range(n_pairs):
        ratio = eigenvalues[i] / eigenvalues[-(i+1)]
        ratios.append(ratio)

    ratio_mean = np.mean(ratios)

    # Score de separabilidad normalizado [0-100]
    # Basado en escala logarítmica (ratio=2 → 0, ratio=100 → 100)
    separability_score = np.clip(20 * np.log10(ratio_mean), 0, 100)

    return {
        'ratio_max': ratio_max,
        'ratio_mean': ratio_mean,
        'eigenvalues': eigenvalues,
        'separability_score': separability_score
    }



# --- FUNCIÓN 1: Métrica del Centro de Riemann ---
def calculate_riemannian_centrality(all_subjects, session='T'):
    """
    Calcula la distancia Riemanniana de cada sujeto al 'Gran Promedio' de la población.
    
    Pasos:
    1. Calcula la matriz de covarianza promedio de cada sujeto.
    2. Calcula el promedio de todos los sujetos (Grand Mean).
    3. Mide la distancia de cada sujeto a ese Grand Mean.
    
    Retorna:
    --------
    dict : Diccionario {subject_id: distancia}
           (Menor distancia = sujeto más representativo/central)
    """
    subject_means = {}
    valid_subjects = []
    
    print("Calculando centroides Riemannianos...")
    
    # Paso 1: Obtener la covarianza media de cada sujeto individual
    for subj in all_subjects:
        try:
            # Asume que load_subject_data está disponible en tu entorno
            X, _ = load_subject_data(subj, session)
            
            # Estimar covarianzas de los trials (matrices simétricas positivas definidas)
            cov_estimator = Covariances(estimator='scm') # Sample Covariance Matrix
            covs = cov_estimator.fit_transform(X)
            
            # Calcular el punto medio Riemanniano de este sujeto
            mean_cov = mean_riemann(covs)
            
            subject_means[subj] = mean_cov
            valid_subjects.append(subj)
        except Exception as e:
            print(f"  [WARN] Omitiendo Sujeto {subj}: {e}")
            
    if not valid_subjects:
        raise ValueError("No se pudieron cargar datos para ningún sujeto.")

    # Paso 2: Calcular el 'Gran Promedio' (El centroide de la población)
    population_covs = np.array([subject_means[s] for s in valid_subjects])
    grand_mean = mean_riemann(population_covs)
    
    # Paso 3: Calcular distancias de cada uno al Gran Promedio
    distances = {}
    for subj in valid_subjects:
        dist = distance_riemann(subject_means[subj], grand_mean)
        distances[subj] = dist
        
    return distances

# --- FUNCIÓN 2: Combinación y Selección (CSP + Riemann) ---
def select_best_source_subject(all_subjects, session='T', w_csp=0.4, w_riemann=0.6):
    """
    Selecciona el mejor sujeto 'Source' combinando:
    1. Calidad de señal (CSP Eigenvalue Ratio) -> Mayor es mejor.
    2. Representatividad (Distancia Riemanniana) -> Menor es mejor.
    
    Parámetros:
    -----------
    all_subjects : list de ints
    session : str
    w_csp : float (peso para la calidad de señal)
    w_riemann : float (peso para la centralidad)
    
    Retorna:
    --------
    best_subject : int (ID del sujeto ganador)
    ranking_df : DataFrame (Tabla con todos los resultados ordenados)
    """
    
    # 1. Obtener métricas de Riemann (usando la Función 1)
    riemann_dists = calculate_riemannian_centrality(all_subjects, session)
    
    results = []
    
    print("Calculando ratios CSP y combinando scores...")
    
    for subj in riemann_dists.keys():
        # 2. Obtener métricas de CSP (usando TU función provista)
        csp_metrics = calculate_csp_eigenvalue_ratio(subj, session)
        
        # Extraemos el valor crudo que nos interesa
        # 'ratio_mean' suele ser más robusto que 'ratio_max'
        csp_score_raw = csp_metrics['ratio_mean']
        
        results.append({
            'Subject': subj,
            'CSP_Raw': csp_score_raw,              # Mayor es mejor
            'Riemann_Dist': riemann_dists[subj]    # Menor es mejor
        })
    
    df = pd.DataFrame(results)
    scaler = MinMaxScaler()
    
    # 3. Normalización y Scoreo
    
    # A. Normalizar CSP (0 a 1)
    df['Score_Quality'] = scaler.fit_transform(df[['CSP_Raw']])
    
    # B. Normalizar e INVERTIR Riemann (para que 1 sea el "mejor" / más cercano)
    dist_norm = scaler.fit_transform(df[['Riemann_Dist']])
    df['Score_Centrality'] = 1 - dist_norm 
    
    # C. Score Final Ponderado
    df['Final_Score'] = (w_csp * df['Score_Quality']) + (w_riemann * df['Score_Centrality'])
    
    # 4. Ordenar ranking (Descendente: mejor score arriba)
    df_ranked = df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    
    best_subject = int(df_ranked.iloc[0]['Subject'])
    
    print(f"\nSujeto ganador: S{best_subject:02d} (Score: {df_ranked.iloc[0]['Final_Score']:.3f})")
    
    return best_subject, df_ranked





