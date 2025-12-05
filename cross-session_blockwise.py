"""
Cross-session transfer learning evaluation for BNCI2014_001 dataset (blockwise).

For each subject (excluding subject 4):
- Load session T (training) and E (evaluation) data
- Train on session T, test on session E
- Evaluate all transfer learning methods with block-wise adaptation
- Save results to CSV files in results/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from structuredata import load_session_binary_mi
from adaptation_testing import evaluate_tl_methods_blockwise
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

TRIALS_PER_RUN = 24
CV_PARAMS = {
    'reg_e_grid': [0.1, 0.5, 1, 2, 5, 10, 20],
    'reg_cl_grid': [0.1, 0.5, 1, 2, 5, 10, 20],
    'metric': 'sqeuclidean',
    'outerkfold': 10,
    'innerkfold': dict(nfold=10, train_size=0.8),
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


def save_results(subject_id, results, y_test, trials_per_run):
    """
    Save blockwise results to CSV files.

    Parameters
    ----------
    subject_id : int
        Subject number.
    results : dict
        Results dictionary from evaluate_tl_methods_blockwise.
    y_test : ndarray
        Full test labels array.
    trials_per_run : int
        Number of trials per run.
    """
    accuracies = results['accuracies']
    times = results['times']
    predictions = results['predictions']
    run_indices = results['run_indices']

    methods = list(accuracies.keys())
    
    acc_data = {'run': run_indices}
    for method in methods:
        acc_data[method] = accuracies[method]
    
    acc_df = pd.DataFrame(acc_data)
    acc_filename = RESULTS_DIR / f"subject_{subject_id:02d}_blockwise_accuracies.csv"
    acc_df.to_csv(acc_filename, index=False)
    print(f"\nSaved accuracies to: {acc_filename}")

    times_data = {'run': run_indices}
    for method in methods:
        times_data[method] = times[method]
    
    times_df = pd.DataFrame(times_data)
    times_filename = RESULTS_DIR / f"subject_{subject_id:02d}_blockwise_times.csv"
    times_df.to_csv(times_filename, index=False)
    print(f"Saved timings to: {times_filename}")

    all_predictions = []
    all_true_labels = []
    all_runs = []
    all_methods = []
    all_trials = []
    
    for run_idx, run_num in enumerate(run_indices):
        test_start_idx = run_num * trials_per_run
        test_end_idx = (run_num + 1) * trials_per_run
        y_test_run = y_test[test_start_idx:test_end_idx]
        
        for method in methods:
            preds = predictions[method][run_idx]
            all_predictions.extend(preds)
            all_true_labels.extend(y_test_run)
            all_runs.extend([run_num] * len(preds))
            all_methods.extend([method] * len(preds))
            all_trials.extend(list(range(test_start_idx, test_start_idx + len(preds))))
    
    if all_predictions:
        pred_df = pd.DataFrame({
            'run': all_runs,
            'trial': all_trials,
            'method': all_methods,
            'prediction': all_predictions,
            'true_label': all_true_labels
        })
        pred_filename = RESULTS_DIR / f"subject_{subject_id:02d}_blockwise_predictions.csv"
        pred_df.to_csv(pred_filename, index=False)
        print(f"Saved predictions to: {pred_filename}")


def main():
    """Run cross-session blockwise evaluation for all subjects."""
    print("Cross-Session Transfer Learning Evaluation (Blockwise)")
    print("BNCI2014_001 Dataset (Binary MI: Left vs Right Hand)")
    print(f"Processing subjects: {SUBJECTS}")
    print(f"Trials per run: {TRIALS_PER_RUN}")

    for subject_id in SUBJECTS:
        try:
            X_train, y_train, X_test, y_test = load_subject_data(subject_id, verbose=True)

            print(f"\nTraining data: {X_train.shape}, Test data: {X_test.shape}")

            print("\nRunning transfer learning evaluation (blockwise)...")
            results = evaluate_tl_methods_blockwise(
                X_source=X_train,
                y_source=y_train,
                X_target=X_test,
                y_target=y_test,
                cv_params=CV_PARAMS,
                trials_per_run=TRIALS_PER_RUN,
                verbose=True
            )

            save_results(subject_id, results, y_test, TRIALS_PER_RUN)

        except Exception as e:
            print(f"\nERROR processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("Cross-session blockwise evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
