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


def calculate_blockwise_accuracy_by_run(results_dir='results', output_file='results/blockwise_accuracy_by_run.csv', verbose=False):
    """
    Calculate accuracy per method (rows) and run (columns), averaged across subjects.
    """
    results_path = Path(results_dir)
    acc_files = sorted(results_path.glob('subject_*_blockwise_accuracies.csv'))

    if not acc_files:
        raise FileNotFoundError(f"No blockwise accuracy files found in {results_dir}")

    all_data = []
    for acc_file in acc_files:
        subject_id = acc_file.stem.split('_')[1]
        df = pd.read_csv(acc_file)
        df['subject'] = subject_id
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    methods = [col for col in combined.columns if col not in ['run', 'subject']]
    
    result = combined.groupby('run')[methods].mean() * 100
    result = result.T
    result.index = [METHOD_RENAME.get(m, m) for m in result.index]
    result = result.reindex([m for m in METHOD_ORDER if m in result.index])
    
    result['Mean'] = result.mean(axis=1)
    result['Std'] = result.iloc[:, :-1].std(axis=1)

    result.to_csv(output_file)

    if verbose:
        print(f"Blockwise accuracy by run saved to {output_file}")
        print(result.round(2))

    return result


def calculate_blockwise_accuracy_by_subject(results_dir='results', output_file='results/blockwise_accuracy_by_subject.csv', verbose=False):
    """
    Calculate accuracy per method (rows) and subject (columns), averaged across runs.
    """
    results_path = Path(results_dir)
    acc_files = sorted(results_path.glob('subject_*_blockwise_accuracies.csv'))

    if not acc_files:
        raise FileNotFoundError(f"No blockwise accuracy files found in {results_dir}")

    subject_means = {}
    for acc_file in acc_files:
        subject_id = acc_file.stem.split('_')[1]
        df = pd.read_csv(acc_file)
        methods = [col for col in df.columns if col != 'run']
        subject_means[f'S{subject_id}'] = df[methods].mean() * 100

    result = pd.DataFrame(subject_means)
    result.index = [METHOD_RENAME.get(m, m) for m in result.index]
    result = result.reindex([m for m in METHOD_ORDER if m in result.index])

    result['Mean'] = result.mean(axis=1)
    result['Std'] = result.iloc[:, :-1].std(axis=1)

    result.to_csv(output_file)

    if verbose:
        print(f"Blockwise accuracy by subject saved to {output_file}")
        print(result.round(2))

    return result


def calculate_blockwise_times_by_run(results_dir='results', output_file='results/blockwise_times_by_run.csv', verbose=False):
    """
    Calculate mean time per method (rows) and run (columns), averaged across subjects.
    """
    results_path = Path(results_dir)
    time_files = sorted(results_path.glob('subject_*_blockwise_times.csv'))

    if not time_files:
        raise FileNotFoundError(f"No blockwise time files found in {results_dir}")

    all_data = []
    for time_file in time_files:
        subject_id = time_file.stem.split('_')[1]
        df = pd.read_csv(time_file)
        df['subject'] = subject_id
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    methods = [col for col in combined.columns if col not in ['run', 'subject']]
    
    result = combined.groupby('run')[methods].mean()
    result = result.T
    result.index = [METHOD_RENAME.get(m, m) for m in result.index]
    result = result.reindex([m for m in METHOD_ORDER if m in result.index])

    result['Mean'] = result.mean(axis=1)

    result.to_csv(output_file)

    if verbose:
        print(f"Blockwise times by run saved to {output_file}")
        print(result.round(4))

    return result


def calculate_method_accuracies(results_dir='results', output_file='results/method_accuracies.csv', verbose=False):
    """Calculate accuracy and standard deviation per method across all subjects."""
    results_path = Path(results_dir)

    prediction_files = sorted(results_path.glob('subject_*_predictions.csv'))

    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {results_dir}")

    methods = ['SC', 'SR', 'RPA', 'EU', 'Forward_Sinkhorn', 'Forward_GroupLasso',
               'Backward_Sinkhorn', 'Backward_GroupLasso']

    method_rename = {
        'Forward_Sinkhorn': 'FOTDA-S',
        'Forward_GroupLasso': 'FOTDA-GL',
        'Backward_Sinkhorn': 'BOTDA-S',
        'Backward_GroupLasso': 'BOTDA-GL',
        'EU': 'EA'
    }

    results_per_subject = {}

    for pred_file in prediction_files:
        subject_id = pred_file.stem.replace('_predictions', '')

        df = pd.read_csv(pred_file)

        subject_accuracies = {}
        for method in methods:
            if method in df.columns:
                accuracy = (df[method] == df['true_label']).mean() * 100
                subject_accuracies[method] = accuracy

        results_per_subject[subject_id] = subject_accuracies

        if verbose:
            print(f"{subject_id}: {len(df)} trials processed")

    results_df = pd.DataFrame(results_per_subject).T

    results_df.rename(columns=method_rename, inplace=True)

    mean_accuracies = results_df.mean()
    std_accuracies = results_df.std()

    summary_df = pd.DataFrame({
        'Method': mean_accuracies.index,
        'Mean_Accuracy': mean_accuracies.values,
        'Std_Accuracy': std_accuracies.values
    })

    results_df['Subject'] = results_df.index
    results_df = results_df[['Subject'] + [col for col in results_df.columns if col != 'Subject']]

    results_df.to_csv(output_file, index=False)

    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    if verbose:
        print(f"\nResults saved to {output_file}")
        print(f"Summary saved to {summary_file}")
        print("\nSummary:")
        for _, row in summary_df.iterrows():
            print(f"{row['Method']}: {row['Mean_Accuracy']:.2f} ± {row['Std_Accuracy']:.2f}%")

    return results_df, summary_df


def calculate_method_times(results_dir='results', output_file='results/method_times.csv', verbose=False):
    """Calculate mean time and standard deviation per method across all subjects."""
    results_path = Path(results_dir)

    time_files = sorted(results_path.glob('subject_*_times.csv'))

    if not time_files:
        raise FileNotFoundError(f"No time files found in {results_dir}")

    methods = ['SC', 'SR', 'RPA', 'EU', 'Forward_Sinkhorn', 'Forward_GroupLasso',
               'Backward_Sinkhorn', 'Backward_GroupLasso']

    method_rename = {
        'Forward_Sinkhorn': 'FOTDA-S',
        'Forward_GroupLasso': 'FOTDA-GL',
        'Backward_Sinkhorn': 'BOTDA-S',
        'Backward_GroupLasso': 'BOTDA-GL',
        'EU': 'EA'
    }

    all_times = {method: [] for method in methods}

    for time_file in time_files:
        df = pd.read_csv(time_file)

        for method in methods:
            if method in df.columns:
                all_times[method].extend(df[method].tolist())

        if verbose:
            subject_id = time_file.stem.replace('_times', '')
            print(f"{subject_id}: {len(df)} trials processed")

    summary_data = []
    for method in methods:
        if all_times[method]:
            times = all_times[method]
            mean_time = np.mean(times)
            std_time = np.std(times)
            summary_data.append({
                'Method': method_rename.get(method, method),
                'Mean_Time': mean_time,
                'Std_Time': std_time
            })

    summary_df = pd.DataFrame(summary_data)

    summary_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\nTime summary saved to {output_file}")
        print("\nTime Summary:")
        for _, row in summary_df.iterrows():
            print(f"{row['Method']}: {row['Mean_Time']:.4f} ± {row['Std_Time']:.4f} seconds")

    return summary_df
