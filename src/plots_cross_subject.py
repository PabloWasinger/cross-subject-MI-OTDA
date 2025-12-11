"""
Plotting functions for cross-subject transfer learning results.

These functions create visualizations for cross-subject experiments
where a single source subject is used to transfer to multiple target subjects.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


METHOD_ORDER = ['SC', 'SR', 'RPA', 'EA', 'FOTDA-S', 'FOTDA-GL', 'BOTDA-S', 'BOTDA-GL']


def plot_cross_subject_accuracy_table(csv_file='results/cross_subject/accuracy_by_target.csv',
                                       output_file='results/cross_subject/accuracy_table.png',
                                       title=None, verbose=False):
    """
    Plot accuracy table: methods (rows) × target subjects (columns).
    
    Args:
        csv_file: Path to CSV file with accuracy data
        output_file: Path to save the plot
        title: Custom title (optional)
        verbose: Whether to print progress
    
    Returns:
        matplotlib Figure object
    """
    df = pd.read_csv(csv_file, index_col=0)
    
    if df.empty:
        raise ValueError(f"CSV file '{csv_file}' contains no data.")
    
    
    df = df.reindex([m for m in METHOD_ORDER if m in df.index])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    
    cell_text = []
    for method in df.index:
        row = [method]
        for col in df.columns:
            if col == 'Std':
                row.append(f'{df.loc[method, col]:.1f}')
            else:
                row.append(f'{df.loc[method, col]:.2f}')
        cell_text.append(row)

    columns = ['Method'] + list(df.columns)

    
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    
    for i in range(len(df.index)):
        table[(i+1, 0)].set_facecolor('#E8E8E8')
        table[(i+1, 0)].set_text_props(weight='bold')

    
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4A90E2')
        table[(0, j)].set_text_props(weight='bold', color='white')

    
    subject_cols = [i for i, c in enumerate(df.columns) if c.startswith('S')]
    for j in subject_cols:
        col = df.columns[j]
        max_idx = df[col].idxmax()
        max_row_idx = list(df.index).index(max_idx)
        table[(max_row_idx+1, j+1)].set_text_props(weight='bold')
        table[(max_row_idx+1, j+1)].set_facecolor('#90EE90')

    
    for i in range(len(df.index)):
        mean_col_idx = list(df.columns).index('Mean') + 1 if 'Mean' in df.columns else None
        std_col_idx = list(df.columns).index('Std') + 1 if 'Std' in df.columns else None
        
        if mean_col_idx:
            table[(i+1, mean_col_idx)].set_facecolor('#FFE6B3')
        if std_col_idx:
            table[(i+1, std_col_idx)].set_facecolor('#FFD6D6')

    
    if 'Mean' in df.columns:
        mean_col_idx = list(df.columns).index('Mean') + 1
        best_mean_idx = list(df.index).index(df['Mean'].idxmax())
        table[(best_mean_idx+1, mean_col_idx)].set_text_props(weight='bold')
        table[(best_mean_idx+1, mean_col_idx)].set_facecolor('#90EE90')

    
    if title is None:
        title = 'Cross-Subject Transfer Learning Accuracy (%)\n(Per Target Subject)'
    plt.title(title, fontsize=14, weight='bold', pad=20)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Accuracy table saved to {output_file}")

    plt.close()
    return fig


def plot_cross_subject_accuracy_bars(csv_file='results/cross_subject/accuracy_by_target.csv',
                                      output_file='results/cross_subject/accuracy_bars.png',
                                      title=None, verbose=False):
    """
    Plot grouped bar chart showing accuracy per method across target subjects.
    
    Args:
        csv_file: Path to CSV file with accuracy data
        output_file: Path to save the plot
        title: Custom title (optional)
        verbose: Whether to print progress
    
    Returns:
        matplotlib Figure object
    """
    df = pd.read_csv(csv_file, index_col=0)
    
    if df.empty:
        raise ValueError(f"CSV file '{csv_file}' contains no data.")
    
    
    subject_cols = [c for c in df.columns if c.startswith('S')]
    df_subjects = df[subject_cols]
    
    
    df_subjects = df_subjects.reindex([m for m in METHOD_ORDER if m in df_subjects.index])

    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(subject_cols))
    width = 0.1
    n_methods = len(df_subjects.index)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    for i, method in enumerate(df_subjects.index):
        offset = (i - n_methods/2 + 0.5) * width
        bars = ax.bar(x + offset, df_subjects.loc[method], width, 
                      label=method, color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Target Subject', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(subject_cols)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim([50, 105])
    ax.grid(axis='y', alpha=0.3)
    
    if title is None:
        title = 'Cross-Subject Transfer Learning Accuracy by Target Subject'
    ax.set_title(title, fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Bar chart saved to {output_file}")

    plt.close()
    return fig


def plot_cross_subject_summary(csv_file='results/cross_subject/summary.csv',
                                output_file='results/cross_subject/summary_plot.png',
                                title=None, verbose=False):
    """
    Plot summary bar chart with mean accuracy and error bars for each method.
    
    Args:
        csv_file: Path to CSV file with summary data
        output_file: Path to save the plot
        title: Custom title (optional)
        verbose: Whether to print progress
    
    Returns:
        matplotlib Figure object
    """
    df = pd.read_csv(csv_file)
    
    if df.empty:
        raise ValueError(f"CSV file '{csv_file}' contains no data.")
    
    
    df['Method_order'] = df['Method'].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 999)
    df = df.sort_values('Method_order').drop('Method_order', axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    
    
    best_idx = df['Mean_Accuracy'].idxmax()
    bar_colors = [colors[i] if i != best_idx else '#2ecc71' for i in range(len(df))]
    
    bars = ax.bar(x, df['Mean_Accuracy'], yerr=df['Std_Accuracy'], 
                  capsize=5, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    
    for i, (bar, val, std) in enumerate(zip(bars, df['Mean_Accuracy'], df['Std_Accuracy'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax.set_ylim([50, 110])
    ax.grid(axis='y', alpha=0.3)
    
    if title is None:
        title = 'Cross-Subject Transfer Learning - Method Comparison'
    ax.set_title(title, fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Summary plot saved to {output_file}")

    plt.close()
    return fig


def plot_cross_subject_heatmap(csv_file='results/cross_subject/accuracy_by_target.csv',
                                output_file='results/cross_subject/accuracy_heatmap.png',
                                title=None, verbose=False):
    """
    Plot heatmap of accuracy: methods × target subjects.
    
    Args:
        csv_file: Path to CSV file with accuracy data
        output_file: Path to save the plot
        title: Custom title (optional)
        verbose: Whether to print progress
    
    Returns:
        matplotlib Figure object
    """
    df = pd.read_csv(csv_file, index_col=0)
    
    if df.empty:
        raise ValueError(f"CSV file '{csv_file}' contains no data.")
    
    
    subject_cols = [c for c in df.columns if c.startswith('S')]
    df_subjects = df[subject_cols]
    
    
    df_subjects = df_subjects.reindex([m for m in METHOD_ORDER if m in df_subjects.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(df_subjects.values, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)
    
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom', fontsize=10)
    
    
    ax.set_xticks(np.arange(len(subject_cols)))
    ax.set_yticks(np.arange(len(df_subjects.index)))
    ax.set_xticklabels(subject_cols)
    ax.set_yticklabels(df_subjects.index)
    
    
    for i in range(len(df_subjects.index)):
        for j in range(len(subject_cols)):
            text = ax.text(j, i, f'{df_subjects.iloc[i, j]:.1f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_xlabel('Target Subject', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    
    if title is None:
        title = 'Cross-Subject Transfer Learning Accuracy Heatmap'
    ax.set_title(title, fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Heatmap saved to {output_file}")

    plt.close()
    return fig


def plot_cross_subject_times_table(csv_file='results/cross_subject/times_summary.csv',
                                    output_file='results/cross_subject/times_table.png',
                                    title=None, verbose=False):
    """
    Plot table of execution times per method.
    
    Args:
        csv_file: Path to CSV file with time summary data
        output_file: Path to save the plot
        title: Custom title (optional)
        verbose: Whether to print progress
    
    Returns:
        matplotlib Figure object
    """
    df = pd.read_csv(csv_file)
    
    if df.empty:
        raise ValueError(f"CSV file '{csv_file}' contains no data.")
    
    # Reorder methods
    df['Method_order'] = df['Method'].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 999)
    df = df.sort_values('Method_order').drop('Method_order', axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')

    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([
            row['Method'],
            f"{row['Mean_Time']:.4f}",
            f"{row['Std_Time']:.4f}",
            f"{row['Mean_Time']:.4f} ± {row['Std_Time']:.4f}"
        ])

    columns = ['Method', 'Mean Time (s)', 'Std Dev (s)', 'Summary']

    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4A90E2')
        table[(0, j)].set_text_props(weight='bold', color='white')

    
    for i in range(len(df)):
        table[(i+1, 0)].set_facecolor('#E8E8E8')
        table[(i+1, 0)].set_text_props(weight='bold')

    
    min_time_idx = df['Mean_Time'].idxmin()
    row_idx = df.index.get_loc(min_time_idx) + 1
    table[(row_idx, 1)].set_facecolor('#90EE90')
    table[(row_idx, 1)].set_text_props(weight='bold')

    if title is None:
        title = 'Cross-Subject Transfer - Execution Time by Method'
    plt.title(title, fontsize=14, weight='bold', pad=20)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"Times table saved to {output_file}")

    plt.close()
    return fig


def plot_all_cross_subject(results_dir='results/cross_subject', verbose=True):
    """
    Generate all cross-subject plots.
    
    Args:
        results_dir: Directory containing cross-subject result files
        verbose: Whether to print progress
    
    Returns:
        Dictionary with all generated figures
    """
    figures = {}
    
    print("=" * 60)
    print("GENERATING CROSS-SUBJECT PLOTS")
    print("=" * 60)
    
    try:
        figures['accuracy_table'] = plot_cross_subject_accuracy_table(
            csv_file=f'{results_dir}/accuracy_by_target.csv',
            output_file=f'{results_dir}/accuracy_table.png',
            verbose=verbose
        )
    except Exception as e:
        print(f"Warning: Could not generate accuracy table: {e}")
    
    try:
        figures['accuracy_bars'] = plot_cross_subject_accuracy_bars(
            csv_file=f'{results_dir}/accuracy_by_target.csv',
            output_file=f'{results_dir}/accuracy_bars.png',
            verbose=verbose
        )
    except Exception as e:
        print(f"Warning: Could not generate accuracy bars: {e}")
    
    try:
        figures['summary_plot'] = plot_cross_subject_summary(
            csv_file=f'{results_dir}/summary.csv',
            output_file=f'{results_dir}/summary_plot.png',
            verbose=verbose
        )
    except Exception as e:
        print(f"Warning: Could not generate summary plot: {e}")
    
    try:
        figures['heatmap'] = plot_cross_subject_heatmap(
            csv_file=f'{results_dir}/accuracy_by_target.csv',
            output_file=f'{results_dir}/accuracy_heatmap.png',
            verbose=verbose
        )
    except Exception as e:
        print(f"Warning: Could not generate heatmap: {e}")
    
    try:
        figures['times_table'] = plot_cross_subject_times_table(
            csv_file=f'{results_dir}/times_summary.csv',
            output_file=f'{results_dir}/times_table.png',
            verbose=verbose
        )
    except Exception as e:
        print(f"Warning: Could not generate times table: {e}")
    
    print("\n" + "=" * 60)
    print("All plots generated!")
    print("=" * 60)
    
    return figures


if __name__ == "__main__":
    
    from evaluation_cross_subject import run_all_evaluations
    run_all_evaluations(verbose=True)
    plot_all_cross_subject(verbose=True)

