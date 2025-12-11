"""
Pipeline variation: ALL subjects as source (except target and subject 4).

Source: 7 subjects × 2 sessions (T and E)
Target: 1 subject × session T only

Reuses all functions from pipeline_comparison.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'models'))

# Reutilizar todo de pipeline_comparison
from pipeline_comparison import (
    set_seed, train_eegnet_on_source, evaluate_methods_samplewise,
    calculate_accuracies, EEGNET_CONFIG, EEGNET_BOTDA_CONFIG,
    CV_PARAMS, N_CALIB, RANDOM_SEED, SUBJECTS
)
from subject_selection_methods import load_subject_data, load_subject_data_raw

# Directorios específicos para esta variante
RESULTS_DIR = Path("results") / "all_sources"
MODELS_DIR = Path("models") / "checkpoints_all_sources"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = ['T', 'E']


def load_all_sources_data(target_id, subjects=SUBJECTS, sessions=SESSIONS):
    """
    Load data from all subjects except target, using both sessions.
    
    Returns both filtered (8-30 Hz) and raw (0.5-100 Hz) data.
    """
    X_filt_all = []
    X_raw_all = []
    y_all = []
    source_subjects = [s for s in subjects if s != target_id]
    
    source_info = {'subjects': source_subjects, 'sessions': sessions, 'trials_per_subject': {}}
    
    for subj in source_subjects:
        subj_trials = 0
        for sess in sessions:
            try:
                X_filt, y = load_subject_data(subj, session=sess)      # 8-30 Hz
                X_raw, _ = load_subject_data_raw(subj, session=sess)   # 0.5-100 Hz
                X_filt_all.append(X_filt)
                X_raw_all.append(X_raw)
                y_all.append(y)
                subj_trials += len(y)
            except Exception as e:
                print(f"  Warning: Could not load S{subj}-{sess}: {e}")
        source_info['trials_per_subject'][subj] = subj_trials
    
    X_source_filt = np.concatenate(X_filt_all, axis=0)
    X_source_raw = np.concatenate(X_raw_all, axis=0)
    y_source = np.concatenate(y_all, axis=0)
    source_info['total_trials'] = len(y_source)
    
    return X_source_filt, X_source_raw, y_source, source_info


def save_results(target_id, predictions, times, y_test, source_info):
    """Save results for a target subject."""
    base_name = f"allsources_tgt_{target_id:02d}"
    
    acc_df = calculate_accuracies(predictions, y_test, print_results=True)
    acc_df.to_csv(RESULTS_DIR / f"{base_name}_accuracies.csv", index=False)
    
    pred_data = {'trial': list(range(1, len(y_test) + 1)), 'true_label': y_test}
    for method, preds in predictions.items():
        pred_data[method] = np.array(preds).flatten()
    pd.DataFrame(pred_data).to_csv(RESULTS_DIR / f"{base_name}_predictions.csv", index=False)
    
    times_data = {'trial': list(range(1, len(y_test) + 1))}
    for method, t in times.items():
        times_data[method] = t
    pd.DataFrame(times_data).to_csv(RESULTS_DIR / f"{base_name}_times.csv", index=False)
    
    source_df = pd.DataFrame([
        {'subject': s, 'trials': t} 
        for s, t in source_info['trials_per_subject'].items()
    ])
    source_df.to_csv(RESULTS_DIR / f"{base_name}_source_info.csv", index=False)
    
    print(f"Results saved to {RESULTS_DIR}")
    return acc_df


def main():
    print("="*70)
    print("CROSS-SUBJECT COMPARISON: ALL SOURCES (SAMPLEWISE)")
    print("="*70)
    print("\nConfiguration:")
    print("- Source: All subjects except target, BOTH sessions (T and E)")
    print("- Target: Single subject, session T only")
    print("="*70)
    
    set_seed(RANDOM_SEED)
    all_results = {}
    
    for target_id in SUBJECTS:
        print(f"\n{'='*70}")
        print(f"TARGET: Subject {target_id}")
        print(f"{'='*70}")
        
        try:
            # Load sources (all except target, both sessions) - filtered AND raw
            print(f"\nLoading source data (all subjects except S{target_id})...")
            X_source_filt, X_source_raw, y_source, source_info = load_all_sources_data(target_id)
            print(f"  Subjects: {source_info['subjects']}")
            print(f"  Total trials: {source_info['total_trials']}")
            print(f"  Class distribution: {np.bincount(y_source)}")
            
            # Load target (session T only) - filtered AND raw
            print(f"\nLoading target data (S{target_id}, session T)...")
            X_target_filt, y_target = load_subject_data(target_id, session='T')      # 8-30 Hz
            X_target_raw, _ = load_subject_data_raw(target_id, session='T')           # 0.5-100 Hz
            print(f"  Shape: {X_target_filt.shape}, Classes: {np.bincount(y_target)}")
            
            # Train EEGNet models on RAW data (no bandpass filter)
            print("\nTraining EEGNet models (RAW data - no bandpass)...")
            
            model_path = MODELS_DIR / f'eegnet_allsrc_tgt_{target_id:02d}.pt'
            eegnet_model, history = train_eegnet_on_source(
                X_source_raw, y_source, config=EEGNET_CONFIG,  # RAW data
                save_path=str(model_path), verbose=True
            )
            
            model_botda_path = MODELS_DIR / f'eegnet_botda_allsrc_tgt_{target_id:02d}.pt'
            eegnet_botda, history_botda = train_eegnet_on_source(
                X_source_raw, y_source, config=EEGNET_BOTDA_CONFIG,  # RAW data
                save_path=str(model_botda_path), verbose=True
            )
            
            # Evaluate - pass both filtered and raw data
            predictions, times, y_test = evaluate_methods_samplewise(
                X_source_filt, X_source_raw, y_source,    # Source: filtered + raw
                X_target_filt, X_target_raw, y_target,    # Target: filtered + raw
                eegnet_model, eegnet_botda, CV_PARAMS,
                n_calib=N_CALIB, verbose=True
            )
            
            acc_df = save_results(target_id, predictions, times, y_test, source_info)
            all_results[target_id] = acc_df
            
        except Exception as e:
            print(f"ERROR with target {target_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    summary_data = []
    for target_id, acc_df in all_results.items():
        row = {'Target': target_id}
        for _, r in acc_df.iterrows():
            row[r['Method']] = r['Accuracy']
            row[f"{r['Method']}_Kappa"] = r['Kappa']
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / 'summary_all_sources.csv', index=False)
    
    acc_cols = ['Target'] + [c for c in summary_df.columns if not c.endswith('_Kappa')]
    print(summary_df[acc_cols].to_string(index=False))
    
    print("\nMean Accuracy (%):")
    methods = [c for c in summary_df.columns if c != 'Target' and not c.endswith('_Kappa')]
    for method in methods:
        print(f"  {method:20s}: {summary_df[method].mean():.2f} ± {summary_df[method].std():.2f}")
    
    print(f"\nResults saved to {RESULTS_DIR}")
    
    return summary_df


if __name__ == "__main__":
    main()

