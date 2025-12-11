"""
Cross-subject transfer learning evaluation for BNCI2014_001 dataset.

Methodology:
1. Select the Best Source Subject using Riemannian Centrality + CSP Quality.
2. Train the model on the Best Source Subject (Session T).
3. Evaluate on all other subjects (acting as Targets) using Sample-wise adaptation.
4. Save results to CSV files in results/cross_subject/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Custom imports
from structuredata import load_session_binary_mi
from adaptation_testing import evaluate_tl_methods_samplewise, calculate_accuracies

# Scientific imports
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from sklearn.preprocessing import MinMaxScaler
from subject_selection_methods import select_best_source_subject

warnings.filterwarnings('ignore', category=RuntimeWarning)



DATA_DIR = Path("data")
GDF_DIR = DATA_DIR / "gdf"
LABELS_DIR = DATA_DIR / "labels"
RESULTS_DIR = Path("results") / "cross_subject"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EEG_CHANNELS = [
    'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
    'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9',
    'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz',
    'EEG-15', 'EEG-16'
]
EOG_CHANNELS = ['EOG-left', 'EOG-central', 'EOG-right']

SUBJECTS = [1, 2, 3, 5, 6, 7, 8, 9] 


N_CALIB = 24 

CV_PARAMS = {
    'reg_e_grid': [0.1, 0.5, 1, 2, 5, 10, 20],
    'reg_cl_grid': [0.1, 0.5, 1, 2, 5, 10, 20],
    'metric': 'sqeuclidean',
    'outerkfold': 5, 
    'innerkfold': None, 
    'M': N_CALIB,
    'norm': None
}



def load_single_session(subject_id, session='T', verbose=False):
    """
    Load data for a single subject and session.
    """
    gdf_file = GDF_DIR / f"A{subject_id:02d}{session}.gdf"
    labels_file = LABELS_DIR / f"A{subject_id:02d}{session}.mat"
    
    if verbose:
        print(f"Loading Subject {subject_id} Session {session}...")

    X, y, _ = load_session_binary_mi(
        str(gdf_file),
        EEG_CHANNELS,
        EOG_CHANNELS,
        tmin=0.5,
        tmax=2.5,
        l_freq=8,
        h_freq=30,
        reject_artifacts=True,
        true_labels_path=str(labels_file),
        verbose=verbose
    )
    return X, y



def save_results(source_id, target_id, predictions, times, y_test):
    """Save results for a specific Source -> Target pair."""
    acc_df = calculate_accuracies(predictions, y_test, print_results=True)

    base_name = f"src_{source_id:02d}_tgt_{target_id:02d}"

    
    acc_file = RESULTS_DIR / f"{base_name}_accuracies.csv"
    acc_df.to_csv(acc_file, index=False)
    
    
    pred_data = {'trial': list(range(1, len(y_test) + 1)), 'true_label': y_test}
    for method, preds in predictions.items():
        pred_data[method] = np.array(preds).flatten()
    
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(RESULTS_DIR / f"{base_name}_predictions.csv", index=False)

    
    times_data = {'trial': list(range(1, len(y_test) + 1))}
    for method, method_times in times.items():
        times_data[method] = method_times
        
    pd.DataFrame(times_data).to_csv(RESULTS_DIR / f"{base_name}_times.csv", index=False)
    print(f"Saved results to {RESULTS_DIR}")



def main():
    print("="*60)
    print("CROSS-SUBJECT TRANSFER LEARNING EVALUATION")
    print("="*60)
    
    
    best_source_id, _ = select_best_source_subject(SUBJECTS, session='T')
    
    
    print(f"\nLoading data for Source Subject {best_source_id}...")
    X_source, y_source = load_single_session(best_source_id, session='T', verbose=True)
    
    
    
    targets = [s for s in SUBJECTS if s != best_source_id]
    
    print(f"\nStarting evaluation on {len(targets)} Target subjects: {targets}")
    
    for target_id in targets:
        print(f"\n{'-'*40}")
        print(f"Evaluating Transfer: S{best_source_id} -> S{target_id}")
        print(f"{'-'*40}")
        
        try:
            X_target, y_target = load_single_session(target_id, session='T', verbose=True)
            
            print(f"Source Data: {X_source.shape}")
            print(f"Target Data: {X_target.shape}")
            
            predictions, times = evaluate_tl_methods_samplewise(
                X_source=X_source,
                y_source=y_source,
                X_target=X_target,
                y_target=y_target,
                cv_params=CV_PARAMS,
                n_calib=N_CALIB,
                verbose=True
            )
            
            y_test_predicted = y_target[N_CALIB:]
            
            save_results(best_source_id, target_id, predictions, times, y_test_predicted)
            
        except Exception as e:
            print(f"ERROR processing target {target_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("Cross-subject evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()