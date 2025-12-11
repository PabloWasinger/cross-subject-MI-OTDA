"""
Evaluation functions for cross-subject transfer learning results.

These functions aggregate and analyze results from cross-subject experiments
where a single source subject is used to transfer to multiple target subjects.
"""

import pandas as pd
import numpy as np
from pathlib import Path


METHOD_RENAME = {
    'Forward_Sinkhorn': 'FOTDA-S',
    'Forward_GroupLasso': 'FOTDA-GL',
    'Backward_Sinkhorn': 'BOTDA-S',
    'Backward_GroupLasso': 'BOTDA-GL',
    'EU': 'EA'
}

METHOD_ORDER = ['SC', 'SR', 'RPA', 'EA', 'FOTDA-S', 'FOTDA-GL', 'BOTDA-S', 'BOTDA-GL']


def calculate_cross_subject_accuracy_by_target(results_dir='results/cross_subject', 
                                                output_file='results/cross_subject/accuracy_by_target.csv', 
                                                verbose=False):
    """
    Calculate accuracy per method (rows) and target subject (columns).
    
    Args:
        results_dir: Directory containing cross-subject result files
        output_file: Path to save the output CSV
        verbose: Whether to print results
    
    Returns:
        DataFrame with methods as rows and target subjects as columns
    """
    results_path = Path(results_dir)
    acc_files = sorted(results_path.glob('src_*_tgt_*_accuracies.csv'))

    if not acc_files:
        raise FileNotFoundError(f"No cross-subject accuracy files found in {results_dir}")

    target_data = {}
    source_id = None
    
    for acc_file in acc_files:
        parts = acc_file.stem.split('_')
        src_id = int(parts[1])
        tgt_id = int(parts[3])
        
        if source_id is None:
            source_id = src_id
        
        df = pd.read_csv(acc_file)
        
        target_acc = {}
        for _, row in df.iterrows():
            method = row['Method']
            method_renamed = METHOD_RENAME.get(method, method)
            target_acc[method_renamed] = row['Accuracy']
        
        target_data[f'S{tgt_id}'] = target_acc

    result = pd.DataFrame(target_data)
    
    result = result.reindex([m for m in METHOD_ORDER if m in result.index])
    
    result['Mean'] = result.mean(axis=1)
    result['Std'] = result.iloc[:, :-1].std(axis=1)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file)

    if verbose:
        print(f"Source Subject: S{source_id}")
        print(f"Cross-subject accuracy by target saved to {output_file}")
        print(result.round(2))

    return result, source_id


def calculate_cross_subject_times_by_target(results_dir='results/cross_subject',
                                             output_file='results/cross_subject/times_by_target.csv',
                                             verbose=False):
    """
    Calculate mean execution time per method (rows) and target subject (columns).
    
    Args:
        results_dir: Directory containing cross-subject result files
        output_file: Path to save the output CSV
        verbose: Whether to print results
    
    Returns:
        DataFrame with methods as rows and target subjects as columns
    """
    results_path = Path(results_dir)
    time_files = sorted(results_path.glob('src_*_tgt_*_times.csv'))

    if not time_files:
        raise FileNotFoundError(f"No cross-subject time files found in {results_dir}")

    target_data = {}
    source_id = None
    
    for time_file in time_files:
        parts = time_file.stem.split('_')
        src_id = int(parts[1])
        tgt_id = int(parts[3])
        
        if source_id is None:
            source_id = src_id
        
        df = pd.read_csv(time_file)
        
        methods = [col for col in df.columns if col != 'trial']
        target_times = {}
        for method in methods:
            method_renamed = METHOD_RENAME.get(method, method)
            target_times[method_renamed] = df[method].mean()
        
        target_data[f'S{tgt_id}'] = target_times

    result = pd.DataFrame(target_data)
    
    result = result.reindex([m for m in METHOD_ORDER if m in result.index])
    
    result['Mean'] = result.mean(axis=1)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file)

    if verbose:
        print(f"Source Subject: S{source_id}")
        print(f"Cross-subject times by target saved to {output_file}")
        print(result.round(4))

    return result, source_id


def calculate_cross_subject_summary(results_dir='results/cross_subject',
                                     output_file='results/cross_subject/summary.csv',
                                     verbose=False):
    """
    Calculate summary statistics (Mean, Std) for each method across all target subjects.
    
    Args:
        results_dir: Directory containing cross-subject result files
        output_file: Path to save the output CSV
        verbose: Whether to print results
    
    Returns:
        DataFrame with summary statistics for each method
    """
    results_path = Path(results_dir)
    pred_files = sorted(results_path.glob('src_*_tgt_*_predictions.csv'))

    if not pred_files:
        raise FileNotFoundError(f"No cross-subject prediction files found in {results_dir}")

    all_accuracies = {}
    source_id = None
    
    for pred_file in pred_files:
        parts = pred_file.stem.split('_')
        src_id = int(parts[1])
        tgt_id = int(parts[3])
        
        if source_id is None:
            source_id = src_id
        
        df = pd.read_csv(pred_file)
        
        methods = [col for col in df.columns if col not in ['trial', 'true_label']]
        for method in methods:
            method_renamed = METHOD_RENAME.get(method, method)
            accuracy = (df[method] == df['true_label']).mean() * 100
            
            if method_renamed not in all_accuracies:
                all_accuracies[method_renamed] = []
            all_accuracies[method_renamed].append(accuracy)

    summary_data = []
    for method in METHOD_ORDER:
        if method in all_accuracies:
            accs = all_accuracies[method]
            summary_data.append({
                'Method': method,
                'Mean_Accuracy': np.mean(accs),
                'Std_Accuracy': np.std(accs),
                'Min_Accuracy': np.min(accs),
                'Max_Accuracy': np.max(accs),
                'N_Targets': len(accs)
            })

    summary_df = pd.DataFrame(summary_data)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\nCross-Subject Transfer Summary (Source: S{source_id})")
        print("=" * 60)
        print(f"Summary saved to {output_file}")
        print("\nMethod Performance:")
        for _, row in summary_df.iterrows():
            print(f"  {row['Method']:12s}: {row['Mean_Accuracy']:.2f} ± {row['Std_Accuracy']:.2f}% "
                  f"[{row['Min_Accuracy']:.1f} - {row['Max_Accuracy']:.1f}]")

    return summary_df, source_id


def calculate_cross_subject_times_summary(results_dir='results/cross_subject',
                                           output_file='results/cross_subject/times_summary.csv',
                                           verbose=False):
    """
    Calculate summary statistics for execution times per method.
    
    Args:
        results_dir: Directory containing cross-subject result files
        output_file: Path to save the output CSV
        verbose: Whether to print results
    
    Returns:
        DataFrame with time summary statistics for each method
    """
    results_path = Path(results_dir)
    time_files = sorted(results_path.glob('src_*_tgt_*_times.csv'))

    if not time_files:
        raise FileNotFoundError(f"No cross-subject time files found in {results_dir}")

    all_times = {}
    
    for time_file in time_files:
        df = pd.read_csv(time_file)
        
        methods = [col for col in df.columns if col != 'trial']
        for method in methods:
            method_renamed = METHOD_RENAME.get(method, method)
            
            if method_renamed not in all_times:
                all_times[method_renamed] = []
            all_times[method_renamed].extend(df[method].tolist())

    summary_data = []
    for method in METHOD_ORDER:
        if method in all_times:
            times = all_times[method]
            summary_data.append({
                'Method': method,
                'Mean_Time': np.mean(times),
                'Std_Time': np.std(times),
                'Min_Time': np.min(times),
                'Max_Time': np.max(times)
            })

    summary_df = pd.DataFrame(summary_data)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\nExecution Time Summary")
        print("=" * 60)
        print(f"Summary saved to {output_file}")
        print("\nMethod Times:")
        for _, row in summary_df.iterrows():
            print(f"  {row['Method']:12s}: {row['Mean_Time']:.4f} ± {row['Std_Time']:.4f} s")

    return summary_df


def run_all_evaluations(results_dir='results/cross_subject', verbose=True):
    """
    Run all cross-subject evaluation functions and save results.
    
    Args:
        results_dir: Directory containing cross-subject result files
        verbose: Whether to print results
    
    Returns:
        Dictionary with all computed DataFrames
    """
    results = {}
    
    print("=" * 60)
    print("CROSS-SUBJECT EVALUATION")
    print("=" * 60)
    
    acc_by_target, source_id = calculate_cross_subject_accuracy_by_target(
        results_dir=results_dir,
        output_file=f'{results_dir}/accuracy_by_target.csv',
        verbose=verbose
    )
    results['accuracy_by_target'] = acc_by_target
    
    times_by_target, _ = calculate_cross_subject_times_by_target(
        results_dir=results_dir,
        output_file=f'{results_dir}/times_by_target.csv',
        verbose=verbose
    )
    results['times_by_target'] = times_by_target
    
    summary, _ = calculate_cross_subject_summary(
        results_dir=results_dir,
        output_file=f'{results_dir}/summary.csv',
        verbose=verbose
    )
    results['summary'] = summary
    
    times_summary = calculate_cross_subject_times_summary(
        results_dir=results_dir,
        output_file=f'{results_dir}/times_summary.csv',
        verbose=verbose
    )
    results['times_summary'] = times_summary
    
    print("\n" + "=" * 60)
    print(f"All evaluations complete! Source Subject: S{source_id}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_all_evaluations(verbose=True)

