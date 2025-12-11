"""
Cross-session transfer learning evaluation for BNCI2014_001 dataset.

For each subject (excluding subject 4):
- Load session T (training) and E (evaluation) data
- Train on session T, test on session E
- Evaluate all transfer learning methods with sample-wise adaptation
- Save results to CSV files in results/
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from structuredata import load_session_binary_mi
from adaptation_testing import evaluate_tl_methods_samplewise, calculate_accuracies
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


DATA_DIR = Path("data")
GDF_DIR = DATA_DIR / "gdf"
LABELS_DIR = DATA_DIR / "labels"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


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
    'outerkfold': 20,
    'innerkfold': None, 
    'M': N_CALIB,
    'norm': None
}


def load_subject_data(subject_id, verbose=False):
    """
    Load training and evaluation session data for a subject.

    Parameters
    ----------
    subject_id : int
        Subject number (1-9, excluding 4).
    verbose : bool
        Print loading info.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    
    train_gdf = GDF_DIR / f"A{subject_id:02d}T.gdf"
    train_labels = LABELS_DIR / f"A{subject_id:02d}T.mat"
    eval_gdf = GDF_DIR / f"A{subject_id:02d}E.gdf"
    eval_labels = LABELS_DIR / f"A{subject_id:02d}E.mat"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Subject {subject_id:02d}")
        print(f"{'='*60}")

    
    if verbose:
        print("Loading training session (T)...")
    X_train, y_train, _ = load_session_binary_mi(
        str(train_gdf),
        EEG_CHANNELS,
        EOG_CHANNELS,
        tmin=0.5,
        tmax=2.5,
        l_freq=8,
        h_freq=30,
        reject_artifacts=True,
        true_labels_path=str(train_labels),
        verbose=verbose
    )

    
    if verbose:
        print("\nLoading evaluation session (E)...")
    X_test, y_test, _ = load_session_binary_mi(
        str(eval_gdf),
        EEG_CHANNELS,
        EOG_CHANNELS,
        tmin=0.5,
        tmax=2.5,
        l_freq=8,
        h_freq=30,
        reject_artifacts=True,
        true_labels_path=str(eval_labels),
        verbose=verbose
    )

    return X_train, y_train, X_test, y_test


def save_results(subject_id, predictions, times, y_test):
    """
    Save predictions and timing results to CSV files.

    Parameters
    ----------
    subject_id : int
        Subject number.
    predictions : dict
        Dictionary with method names and predictions.
    times : dict
        Dictionary with method names and execution times.
    y_test : ndarray
        True labels for test set.
    """
    
    acc_df = calculate_accuracies(predictions, y_test, print_results=True)

    
    acc_filename = RESULTS_DIR / f"subject_{subject_id:02d}_accuracies.csv"
    acc_df.to_csv(acc_filename, index=False)
    print(f"\nSaved accuracies to: {acc_filename}")

  
    pred_data = {'trial': list(range(1, len(y_test) + 1)), 'true_label': y_test}
    for method, preds in predictions.items():
        pred_data[method] = np.array(preds).flatten()

    pred_df = pd.DataFrame(pred_data)
    pred_filename = RESULTS_DIR / f"subject_{subject_id:02d}_predictions.csv"
    pred_df.to_csv(pred_filename, index=False)
    print(f"Saved predictions to: {pred_filename}")

    
    times_data = {'trial': list(range(1, len(y_test) + 1))}
    for method, method_times in times.items():
        times_data[method] = method_times

    times_df = pd.DataFrame(times_data)
    times_filename = RESULTS_DIR / f"subject_{subject_id:02d}_times.csv"
    times_df.to_csv(times_filename, index=False)
    print(f"Saved timings to: {times_filename}")


def main():
    """Run cross-session evaluation for all subjects."""
    print("Cross-Session Transfer Learning Evaluation")
    print("BNCI2014_001 Dataset (Binary MI: Left vs Right Hand)")
    print(f"Processing subjects: {SUBJECTS}")

    for subject_id in SUBJECTS:
        try:
            
            X_train, y_train, X_test, y_test = load_subject_data(subject_id, verbose=True)

            print(f"\nTraining data: {X_train.shape}, Test data: {X_test.shape}")

            
            print("\nRunning transfer learning evaluation...")
            predictions, times = evaluate_tl_methods_samplewise(
                X_source=X_train,
                y_source=y_train,
                X_target=X_test,
                y_target=y_test,
                cv_params=CV_PARAMS,
                n_calib=N_CALIB,
                verbose=True
            )

            
            y_test_predicted = y_test[N_CALIB:]

            
            save_results(subject_id, predictions, times, y_test_predicted)

        except Exception as e:
            print(f"\nERROR processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("Cross-session evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
